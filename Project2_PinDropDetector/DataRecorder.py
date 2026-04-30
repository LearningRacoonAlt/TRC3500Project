import serial
import numpy as np
import csv
import os

# ==========================================
# SETTINGS
# ==========================================
PORT = 'COM5'
BAUD = 230400
THRESHOLD = 15         
RECORD_SAMPLES = 20000 
CSV_FILENAME = "all_coin_drops_wide.csv" 

drop_count = 1
is_recording = False

# ==========================================
# FILE SETUP & AUTO-RESUME
# ==========================================
# Use existing file, else create new
if os.path.exists(CSV_FILENAME):
    print(f"Found existing '{CSV_FILENAME}'. Resuming...")
    try:
        with open(CSV_FILENAME, "r") as f:
            header = next(csv.reader(f))
            drop_count = len(header)
    except StopIteration:
        pass 
else:
    print(f"Creating new file '{CSV_FILENAME}'...")

# Continue from last drop count if possible
print(f"Next Drop ID: {drop_count}\n")
print("Listening for data collection... (Press Ctrl+C to stop)")

# Setup Serial
ser = serial.Serial(PORT, BAUD)
ser.reset_input_buffer()

# Using a bytearray is significantly faster for data buffering
buffer = bytearray()

try:
    while True:
        if ser.in_waiting:
            # Read all available bytes in one go
            chunk = ser.read(ser.in_waiting)

            if not is_recording:
                # Check if threshold crossed to trigger recording
                if max(chunk) > THRESHOLD:
                    print(f"Detected! Recording Drop #{drop_count}...")
                    
                    # Starts listening upon impact
                    is_recording = True
                    buffer.extend(chunk)
            else:
                buffer.extend(chunk)

                # Once we have enough samples, process them
                if len(buffer) >= RECORD_SAMPLES:
                    # Convert to numpy array for max processing speed
                    final_wave = np.frombuffer(buffer[:RECORD_SAMPLES], dtype=np.uint8)
                    
                    # Save to csv as new column
                    rows = []
                    if os.path.exists(CSV_FILENAME):
                        with open(CSV_FILENAME, "r") as f:
                            rows = list(csv.reader(f))

                    # Initialize file structure
                    if not rows:
                        rows.append(["Sample_Index"])
                        for i in range(len(final_wave)):
                            rows.append([i])

                    # Header
                    rows[0].append(f"Drop_{drop_count}")

                    for i, val in enumerate(final_wave):
                        # Inject data
                        if i + 1 < len(rows):
                            rows[i+1].append(val)

                        # If we have mismatched lengths, create new rows
                        else:
                            pad_length = len(rows[0]) - 2
                            rows.append([i] + [""] * pad_length + [val])

                    # Overwrite and save
                    with open(CSV_FILENAME, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(rows)
                    
                    # Feedback and reset
                    print(f"Saved Drop #{drop_count} to CSV")
                    print(f"{'-'*30}\n")
                    
                    buffer.clear()
                    is_recording = False
                    drop_count += 1
                    ser.reset_input_buffer()

# Close program if interrupted
except KeyboardInterrupt:
    print("\nData collection stopped.")
finally:
    ser.close()
