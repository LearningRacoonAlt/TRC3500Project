import threading

import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports


# ========== CONFIGURATION ==========

BAUD_RATE = 230400

# Packet protocol constants: Every valid data packet starts with two 0xAA bytes,
# followed by 1 byte for the chest band and 1 byte for the thermistor (4 bytes total).
HEADER_BYTES = bytes([0xAA, 0xAA])
HEADER_BYTE = 0xAA
PACKET_SIZE_BYTES = 4

SAMPLE_RATE_HZ = 500 # Microcontroller sends data 500 times per second
ADC_MAX = 255        # 8 bit ADC resolution

# Window size for the moving average filter (250 samples at 500Hz = 0.5 seconds of smoothing)
SMOOTHING_WINDOW_SAMPLES = 250

MIN_EVENT_SPACING_S = 0.5 # Minimum time (seconds) required between consecutive peaks

# Thresholds (on normalized 0-1 scale) that a peak must cross to be considered valid
CHEST_BAND_INHALE_PEAK_THRESHOLD = 0.10
THERMISTOR_EXHALE_PEAK_THRESHOLD = 0.10

# Expected timing window for an exhalation to occur immediately after an inhalation
MIN_EXHALE_LAG_S = 0.25
MAX_EXHALE_LAG_S = 6.0

# ===================================


# ========== SERIAL READING ==========

def find_stm32_port():
    """Finds STM32 port"""
    ports = serial.tools.list_ports.comports()

    for port in ports:
        if "STLink" in port.description or "STM" in port.description:
            return port.device

        if "0483:5740" in port.hwid.upper():
            return port.device

    return None


def create_recording_buffer():
    """Initializes empty lists to store raw data points from both sensors"""
    return {
        "chest_band": [],
        "thermistor": [],
    }


def align_buffer_to_header(serial_buffer):
    """
    Ensures packet synchronization. Looks for the 0xAA header.
    Trashes any leading junk data so the buffer always starts precisely at a packet boundary.
    """
    header_index = serial_buffer.find(HEADER_BYTES)

    if header_index >= 0:
        if header_index > 0:
            # Drop everything prior to the valid header sequence
            del serial_buffer[:header_index]
        return True

    if len(serial_buffer) > 0:
        if serial_buffer[-1] == HEADER_BYTE:
            del serial_buffer[:-1]
        else:
            serial_buffer.clear()

    return False


def record_sensor_samples(ser, stop_event, recording_buffer):
    """
    Target function run in a separate background thread. 
    Continuously polls the serial interface and extracts raw data values.
    """
    ser.reset_input_buffer() # Flush old data
    serial_buffer = bytearray()

    # Main collection loop active until main thread signals to stop
    while not stop_event.is_set():
        chunk = ser.read(ser.in_waiting or 1)

        if chunk:
            serial_buffer.extend(chunk)

        # Process all possible packets
        while len(serial_buffer) >= 2:
            if not align_buffer_to_header(serial_buffer):
                break

            # Wait for more bytes to complete the packet
            if len(serial_buffer) < PACKET_SIZE_BYTES:
                break

            # Extract data values from explicit byte offsets (Index 2 and 3)
            recording_buffer["chest_band"].append(serial_buffer[2])
            recording_buffer["thermistor"].append(serial_buffer[3])

            # Remove processed packet
            del serial_buffer[:PACKET_SIZE_BYTES]

    # Clean up loop
    while len(serial_buffer) >= PACKET_SIZE_BYTES:
        if not align_buffer_to_header(serial_buffer):
            break

        if len(serial_buffer) < PACKET_SIZE_BYTES:
            break

        recording_buffer["chest_band"].append(serial_buffer[2])
        recording_buffer["thermistor"].append(serial_buffer[3])

        del serial_buffer[:PACKET_SIZE_BYTES]


# ========== SIGNAL PROCESSING ==========

def moving_average(signal, window_size):
    """
    Applies a low-pass moving average filter to reduce high-frequency electrical or motion noise.
    Uses an optimized running-sum approach to maintain efficiency.
    """
    if not signal:
        return []

    if window_size <= 1:
        return list(signal)

    smoothed = []
    running_sum = 0

    for i, value in enumerate(signal):
        running_sum += value

        # When window size is reached, subtract the oldest sample falling out of the window frame
        if i >= window_size:
            running_sum -= signal[i - window_size]
            average = running_sum / window_size
        # If full size not reached
        else:
            average = running_sum / (i + 1)

        smoothed.append(average)

    return smoothed


def normalize(signal):
    """
    Performs Min-Max normalization to map all data points strictly between 0.0 and 1.0.
    This calibrates varying sensor amplitudes to a standard scale.
    """
    if not signal:
        return []

    min_value = min(signal)
    max_value = max(signal)

    if max_value == min_value:
        return [0.0 for _ in signal] # Prevent division-by-zero if signal is flat

    return [
        (value - min_value) / (max_value - min_value)
        for value in signal
    ]


# ========== EVENT DETECTION ==========

def detect_peak_times(signal, sample_rate_hz, min_spacing_s, peak_threshold):
    """
    Finds local maxima (peaks) in a 1D signal that exceed a designated threshold,
    using Non-Maximum Suppression to prevent closely spaced false triggers.
    """
    if len(signal) < 3:
        return []

    # Convert timing constraint from seconds to equivalent index/sample count
    min_spacing_samples = int(min_spacing_s * sample_rate_hz)
    candidate_peaks = []

    # Standard derivative check (value must be greater than neighbors)
    for i in range(1, len(signal) - 1):
        previous_value = signal[i - 1]
        current_value = signal[i]
        next_value = signal[i + 1]

        is_peak = previous_value < current_value and next_value <= current_value
        is_above_threshold = current_value > peak_threshold

        if is_peak and is_above_threshold:
            candidate_peaks.append((i, current_value))

    if not candidate_peaks:
        return []

    # Sort candidate peaks from highest amplitude to lowest
    candidate_peaks.sort(key=lambda item: item[1], reverse=True)
    accepted_peak_indices = []

    # Enforce minimum distance separation constraint
    for peak_index, _peak_value in candidate_peaks:
        too_close = False

        for accepted_index in accepted_peak_indices:
            if abs(peak_index - accepted_index) < min_spacing_samples:
                too_close = True
                break

        if not too_close:
            accepted_peak_indices.append(peak_index)

    # Re-sort chronologically by sequence order rather than amplitude
    accepted_peak_indices.sort()

    # Map indices back into timestamp measurements (seconds)
    return [
        peak_index / sample_rate_hz
        for peak_index in accepted_peak_indices
    ]


def detect_breathing_cycles(
    inhale_estimates, 
    exhale_estimates, 
    min_exhale_lag_s, 
    max_exhale_lag_s,
):
    """
    Matches inhalation events to subsequent exhalation events based on time delays.
    Validates true breathing cycles by making sure an exhale follows an inhale within physiological reason.
    """
    breathing_cycles = []
    exhale_index = 0

    for inhale_index, inhale_time_s in enumerate(inhale_estimates):
        next_inhale_time_s = None

        # Look ahead to see when the next breath starts to bound the current search window
        if inhale_index + 1 < len(inhale_estimates):
            next_inhale_time_s = inhale_estimates[inhale_index + 1]

        # Define valid time boundaries where the matching exhale peak should reside
        window_start_s = inhale_time_s + min_exhale_lag_s
        window_end_s = inhale_time_s + max_exhale_lag_s

        # Cut off the window early if a new inhalation begins before the max lag expires
        if next_inhale_time_s is not None:
            window_end_s = min(window_end_s, next_inhale_time_s)

        # Skip over historical exhale peaks that happened prior to our search window
        while (
            exhale_index < len(exhale_estimates)
            and exhale_estimates[exhale_index] < window_start_s
        ):
            exhale_index += 1

        if exhale_index >= len(exhale_estimates):
            break

        exhale_time_s = exhale_estimates[exhale_index]

        # If the valid exhale falls inside our bounds, record a confirmed breath cycle
        if exhale_time_s <= window_end_s:
            exhale_lag_s = exhale_time_s - inhale_time_s
            # Use the midpoint between inhale and exhale peaks
            detected_time_s = (inhale_time_s + exhale_time_s) / 2.0

            breathing_cycles.append({
                "cycle_index": len(breathing_cycles) + 1,
                "inhale_time_s": inhale_time_s,
                "exhale_time_s": exhale_time_s,
                "exhale_lag_s": exhale_lag_s,
                "detected_time_s": detected_time_s,
            })

            exhale_index += 1

    return breathing_cycles


def process_data(final_signals):
    """Wrapper coordinating feature extraction and cycle tracking across both processed signals"""
    # Find timestamps for all peaks
    inhale_estimates = detect_peak_times(
        final_signals["chest_band_normalized"],
        sample_rate_hz=SAMPLE_RATE_HZ,
        min_spacing_s=MIN_EVENT_SPACING_S,
        peak_threshold=CHEST_BAND_INHALE_PEAK_THRESHOLD,
    )

    exhale_estimates = detect_peak_times(
        final_signals["thermistor_normalized"],
        sample_rate_hz=SAMPLE_RATE_HZ,
        min_spacing_s=MIN_EVENT_SPACING_S,
        peak_threshold=THERMISTOR_EXHALE_PEAK_THRESHOLD,
    )

    sensor_estimates = {
        "chest_band_normalized": inhale_estimates,
        "thermistor_normalized": exhale_estimates,
    }

    # Match peaks together into complete cycles
    breathing_cycles = detect_breathing_cycles(
        inhale_estimates,
        exhale_estimates,
        min_exhale_lag_s=MIN_EXHALE_LAG_S,
        max_exhale_lag_s=MAX_EXHALE_LAG_S,
    )

    return sensor_estimates, breathing_cycles


# ========== PLOTTING ==========

def signal_display_name(signal_name):
    """Convert dictionary tracking keys into readable legend labels"""
    display_names = {
        "chest_band_raw": "Chest Band",
        "thermistor_raw": "Thermistor",
        "chest_band_normalized": "Chest Band, Normalized",
        "thermistor_normalized": "Thermistor, Normalized",
    }

    return display_names.get(signal_name, signal_name)


def estimate_display_name(signal_name):
    """Helper to identify peak classifications for visual legend titles"""
    if signal_name == "chest_band_normalized":
        return "estimated inhale peak"

    if signal_name == "thermistor_normalized":
        return "estimated exhale peak"

    return "estimated peak"


def plot_signals(
    signals,
    plot_filename,
    title,
    y_label,
    y_limits=None,
    sensor_estimates=None,
    breathing_cycle_times=None,
):
    """Generates stacked subplots of tracking streams and overlays discovered analytical events"""
    if not signals:
        return None

    signal_names = list(signals.keys())
    num_samples = len(signals[signal_names[0]])

    if num_samples == 0:
        return None

    # Error checking to prevent crashes during index iteration matching
    for name in signal_names:
        if len(signals[name]) != num_samples:
            raise ValueError(
                f"Signal length mismatch: {name} has {len(signals[name])} samples, "
                f"expected {num_samples}."
            )

    # Generate matching timeline vector scaled in seconds
    time_s = [i / SAMPLE_RATE_HZ for i in range(num_samples)]

    # Dynamic subplot matrix layout depending on how many signals are mapped in argument dictionary
    fig, axes = plt.subplots(
        len(signal_names),
        1,
        figsize=(12, 3.5 * len(signal_names)),
        sharex=True,
    )

    if len(signal_names) == 1:
        axes = [axes]

    fig.suptitle(title)

    for ax, name in zip(axes, signal_names):
        # Draw the main continuous raw or filtered waveform line
        ax.plot(
            time_s,
            signals[name],
            linewidth=0.8,
            label=signal_display_name(name),
        )

        # Draw vertical blue dashed lines representing isolated event peaks
        if sensor_estimates is not None:
            estimate_times = sensor_estimates.get(name, [])
            first_estimate = True

            for estimate_time_s in estimate_times:
                ax.axvline(
                    estimate_time_s,
                    linewidth=0.8,
                    linestyle="--",
                    color="blue",
                    label=estimate_display_name(name) if first_estimate else None,
                )
                first_estimate = False # Avoid duplicate labels

        # Draw solid red lines for verified combined breathing cycles
        if breathing_cycle_times is not None:
            first_cycle = True

            for cycle_time_s in breathing_cycle_times:
                ax.axvline(
                    cycle_time_s,
                    linewidth=1.3,
                    linestyle="-",
                    color="red",
                    label="detected breathing cycle" if first_cycle else None,
                )
                first_cycle = False

        ax.set_ylabel(y_label)

        if y_limits is not None:
            ax.set_ylim(*y_limits)

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150) # Save artifact export image

    # Render non-blocking display window
    plt.show(block=False)
    plt.pause(0.1)

    return plot_filename


# ========== PRINTING ==========

def print_generated_plots(plot_files):
    print("\nGenerated plots:")

    for filename in plot_files:
        print(f"  {filename}")


def print_breathing_cycle_summary(breathing_cycles):
    """Calculates statistical metrics including Breaths Per Minute (BPM)"""
    print("\nCycles:")

    if not breathing_cycles:
        print("  None")
        print("\nConfirmed breathing cycles: 0")
        return

    # Print out descriptive attributes of individual breath cycles
    for cycle in breathing_cycles:
        print(
            f"  Cycle {cycle['cycle_index']}: "
            f"detected={cycle['detected_time_s']:.3f} s, "
            f"inhale={cycle['inhale_time_s']:.3f} s, "
            f"exhale={cycle['exhale_time_s']:.3f} s, "
            f"exhale_lag={cycle['exhale_lag_s']:.3f} s"
        )

    print(f"\nConfirmed breathing cycles: {len(breathing_cycles)}")

    detected_times = [
        cycle["detected_time_s"]
        for cycle in breathing_cycles
    ]

    # Compute breathing speed frequency if at least two intervals exist
    if len(detected_times) >= 2:
        # Calculate time passed between successive breath intervals
        cycle_spacings = [
            detected_times[i] - detected_times[i - 1]
            for i in range(1, len(detected_times))
        ]

        average_cycle_spacing_s = sum(cycle_spacings) / len(cycle_spacings)

        if average_cycle_spacing_s > 0:
            # Conversion math from time gap (seconds) to rate frequency (Breaths / Minute)
            breathing_rate_bpm = 60.0 / average_cycle_spacing_s
            print(f"Average detected-cycle spacing: {average_cycle_spacing_s:.2f} s")
            print(f"Estimated breathing rate: {breathing_rate_bpm:.1f} breaths/min")


# ========== RECORDING SESSION ==========

def handle_recording_finished(recording_buffer, recording_number):
    """Executes the data processing, filtering, algorithmic evaluation, and plotting pipeline"""
    chest_band_raw = recording_buffer["chest_band"]
    thermistor_raw = recording_buffer["thermistor"]

    num_samples = len(chest_band_raw)
    duration_s = num_samples / SAMPLE_RATE_HZ

    print(f"\nCaptured {num_samples} sample pairs ({duration_s:.2f} s).")

    if num_samples == 0:
        print("No data received.")
        return

    plot_files = []

    # Visualize unprocessed raw streams
    raw_signals = {
        "chest_band_raw": chest_band_raw,
        "thermistor_raw": thermistor_raw,
    }

    plot_files.append(
        plot_signals(
            raw_signals,
            f"breath_raw_plot_{recording_number:03d}.png",
            title="Raw Sensor Signals",
            y_label="ADC Value (8-bit)",
            y_limits=(0, ADC_MAX),
        )
    )

    # Filter noise and normalize values
    chest_band_smoothed = moving_average(
        chest_band_raw,
        SMOOTHING_WINDOW_SAMPLES,
    )

    thermistor_smoothed = moving_average(
        thermistor_raw,
        SMOOTHING_WINDOW_SAMPLES,
    )

    final_signals = {
        "chest_band_normalized": normalize(chest_band_smoothed),
        "thermistor_normalized": normalize(thermistor_smoothed),
    }

    # Run algorithmic peak analysis
    sensor_estimates, breathing_cycles = process_data(final_signals)

    breathing_cycle_times = [
        cycle["detected_time_s"]
        for cycle in breathing_cycles
    ]

    # Output Final Processed Graph Results
    plot_files.append(
        plot_signals(
            final_signals,
            f"breath_final_plot_{recording_number:03d}.png",
            title="Breathing Cycle Detection from Chest Band and Thermistor Peaks",
            y_label="Normalized Amplitude",
            y_limits=(0, 1),
            sensor_estimates=sensor_estimates,
            breathing_cycle_times=breathing_cycle_times,
        )
    )

    # Filter out None values
    plot_files = [
        filename
        for filename in plot_files
        if filename is not None
    ]

    print_generated_plots(plot_files)
    print_breathing_cycle_summary(breathing_cycles)


# ========== MAIN PROGRAM ==========

def main():
    """Main program"""
    port = find_stm32_port()

    # Manual fallback prompt if auto-discovery fails
    if port is None:
        port = input("Enter COM port, e.g. COM3 or /dev/ttyACM0: ").strip()

    print(f"Opening {port} at {BAUD_RATE} baud...")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
    except serial.SerialException as e:
        print(f"Could not open serial port: {e}")
        return

    print("Serial port opened.")
    print("Press 's' then Enter to start recording.")
    print("Press 's' then Enter again to stop recording.")
    print("Press Ctrl+C to quit.")

    is_recording = False
    stop_event = threading.Event() # Command the thread to stop
    recording_thread = None
    recording_buffer = None
    recording_number = 1

    try:
        while True:
            command = input("> ").strip().lower()

            if command != "s":
                print("Use 's' to start/stop recording, or Ctrl+C to quit.")
                continue

            # State machine toggle to start recording
            if not is_recording:
                recording_buffer = create_recording_buffer()
                stop_event.clear()

                # Run the function in the background
                recording_thread = threading.Thread(
                    target=record_sensor_samples,
                    args=(ser, stop_event, recording_buffer),
                )

                recording_thread.start()
                is_recording = True

                print("Recording started.")

            # State machine toggle to stop recording
            else:
                stop_event.set()        # Alert background loop to exit
                recording_thread.join() # Synchronize threads

                is_recording = False

                print("Recording stopped.")
                handle_recording_finished(recording_buffer, recording_number)

                recording_number += 1

                print("\nPress 's' then Enter to start another recording.")
                print("Press Ctrl+C to quit.")

    # Exit program
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting...")

        if is_recording:
            stop_event.set()
            recording_thread.join()

            print("Recording stopped.")
            handle_recording_finished(recording_buffer, recording_number)

    # Close serial port
    finally:
        ser.close()
        print("Serial port closed.")


if __name__ == "__main__":
    main()
