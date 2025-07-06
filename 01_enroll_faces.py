import face_recognition
import numpy as np
import os
import csv
# No need for datetime in this specific script, but keep if other parts of your system use it.

# --- Configuration ---
EMPLOYEE_DATA_DIR = "employee_data" # Folder where your employee images are stored
DATA_DIR = "data"                   # Folder for CSV files
EMPLOYEE_FACES_CSV = os.path.join(DATA_DIR, "employee_faces.csv")

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMPLOYEE_DATA_DIR, exist_ok=True) # Ensure this exists if you put images here

print("Starting face enrollment process...")
print(f"Reading images from: {EMPLOYEE_DATA_DIR}")

known_face_encodings = []
known_face_names = []
known_employee_ids = []
# We don't need a list for workstations here as we write directly to CSV

# Open CSV file in write mode to create/overwrite it with the correct header
with open(EMPLOYEE_FACES_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # MODIFIED: Add 'workstation' to the CSV header
    writer.writerow(['employee_id', 'name', 'face_encoding', 'workstation'])

    for filename in os.listdir(EMPLOYEE_DATA_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")): # Use .lower() for case-insensitivity
            file_path = os.path.join(EMPLOYEE_DATA_DIR, filename)

            employee_id = ""
            employee_name = ""
            employee_workstation = "N/A" # Default in case parsing fails

            try:
                # Extract filename without extension (e.g., '101_JohnDoe_WS01')
                base_filename = os.path.splitext(filename)[0]
                parts = base_filename.split('_')

                if len(parts) >= 3:
                    # Assuming format: ID_Name_Workstation
                    employee_id = parts[0].strip()
                    employee_name = parts[1].replace('_', ' ').title().strip()
                    employee_workstation = '_'.join(parts[2:]).strip() # Join remaining for workstation
                elif len(parts) == 2 and parts[0].isdigit(): # If only ID_Name format
                    employee_id = parts[0].strip()
                    employee_name = parts[1].replace('_', ' ').title().strip()
                    employee_workstation = "Default" # Assign a default or leave blank if no workstation in name
                else: # Fallback for other formats
                    employee_id = base_filename.strip()
                    employee_name = base_filename.replace('_', ' ').title().strip()
                    employee_workstation = "Unknown" # Assign unknown if not following format

                # Add a check to ensure ID is actually a digit if that's a strict requirement
                if not employee_id.isdigit():
                    print(f"  Warning: Employee ID '{employee_id}' (from {filename}) is not numeric. Using as is.")

                print(f"Processing {filename} (ID: {employee_id}, Name: {employee_name}, Workstation: {employee_workstation})...")

                image = face_recognition.load_image_file(file_path)
                face_encodings = face_recognition.face_encodings(image)

                if len(face_encodings) > 1:
                    print(f"  Warning: Multiple faces found in {filename}. Using the first one.")
                elif len(face_encodings) == 0:
                    print(f"  Error: No face found in {filename}. Skipping.")
                    continue # Skip to the next file

                face_encoding = face_encodings[0]
                
                # Convert face encoding to a string for CSV
                encoding_str = ",".join(map(str, face_encoding.tolist()))

                # MODIFIED: Write employee_workstation to the CSV row
                writer.writerow([employee_id, employee_name, encoding_str, employee_workstation])

                # These lists are for in-memory use during this script's execution,
                # not strictly needed for the CSV writing, but can be kept.
                known_face_encodings.append(face_encoding)
                known_face_names.append(employee_name)
                known_employee_ids.append(employee_id)

                print(f"  Successfully encoded and saved {employee_name}.")

            except Exception as e:
                print(f"  Failed to process {filename}: {e}")

print("\nFace enrollment complete!")
print(f"Total employees enrolled: {len(known_face_encodings)}")
print(f"Face data saved to: {EMPLOYEE_FACES_CSV}")

