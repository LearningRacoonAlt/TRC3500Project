import serial
import numpy as np

# ==========================================
# SETTINGS
# ==========================================
PORT = 'COM5'
BAUD = 230400
THRESHOLD = 10          
RECORD_SAMPLES = 20000  # 1 second of data at 20kHz

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    ser = serial.Serial(PORT, BAUD)
    ser.reset_input_buffer()
    
    print(f"Coin Drop Classifier Live on {PORT}...")
    print("Listening for impacts... (Press Ctrl+C to stop)")

    is_recording = False
    buffer = bytearray() # bytearray is 100x faster than python lists

    try:
        while True:
            if ser.in_waiting:
                # Read all available raw bytes directly
                chunk = ser.read(ser.in_waiting)
                
                if not is_recording:
                    # Check if threshold crossed to trigger recording
                    if max(chunk) > THRESHOLD:
                        print("\nImpact Detected! Analyzing...")

                        # Starts listening upon impact
                        is_recording = True
                        buffer.extend(chunk)
                        
                else:
                    buffer.extend(chunk)

                    # Once we have enough samples, process them
                    if len(buffer) >= RECORD_SAMPLES:
                        # Convert to numpy array for max processing speed
                        data = np.frombuffer(buffer[:RECORD_SAMPLES], dtype=np.uint8).astype(np.float32)
                        
                        # Signal alignment
                        start_idx = np.argmax(data > THRESHOLD) + 50 # moment when coin hits the surface, skips noise of strike
                        wave = data[start_idx:]
                        
                        # ensure sufficient length
                        if len(wave) >= 3000:
                            # Extract rms and peaks
                            rms = np.sqrt(np.mean(wave[:3000]**2))
                            peak = np.max(wave)
                            p_idx = np.argmax(wave)
                            secondary = np.max(wave[p_idx + 100:]) if p_idx + 100 < len(wave) else 0

                            # Classify based on average power and peaks
                            if rms > 0.10:
                                height = "HIGH"
                                dist = "NEAR" if peak > 29.5 else "FAR"
                            else:
                                height = "SHORT"
                                dist = "NEAR" if (peak + secondary) > 25.5 else "FAR"

                            # Print output
                            print(f"PREDICTION: {height} & {dist}")
                            print(f"[Peak: {peak:.0f} | Bounce: {secondary:.0f} | RMS: {rms:.3f}]")
                        
                        # Reset for the next drop
                        buffer.clear()
                        is_recording = False
                        ser.reset_input_buffer()

    # Stop program on interrupt
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
