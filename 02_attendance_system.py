import face_recognition
import cv2
import numpy as np
import os
import csv
from datetime import datetime, date, timedelta
import time
import dlib
from screeninfo import get_monitors
import faiss # Import Faiss
import pandas as pd # Import pandas for Excel operations

# --- Configuration ---
DATA_DIR = "data"
EMPLOYEE_FACES_CSV = os.path.join(DATA_DIR, "employee_faces.csv")
MONTHLY_ATTENDANCE_DIR = os.path.join(DATA_DIR, "monthly_attendance_reports")

FACE_RECOGNITION_TOLERANCE = 0.5 # This is used for compare_faces
COOLDOWN_SECONDS = 10  
last_recognized_time = {}

FAISS_K_NEIGHBORS = 15

MESSAGE_DISPLAY_DURATION_SECONDS = 10
successful_attendance_message_lines = []
successful_attendance_message_until = datetime.now()

# --- Initialize Dlib models ---
# Ensure these paths are correct for your environment
predictor_path = r"C:\Users\krush\AppData\Local\Programs\Python\Python310\Lib\site-packages\face_recognition_models\models\shape_predictor_68_face_landmarks.dat"
face_encoder_path = r"C:\Users\krush\AppData\Local\Programs\Python\Python310\Lib\site-packages\face_recognition_models\models\dlib_face_recognition_resnet_model_v1.dat"

if not os.path.exists(predictor_path):
    print(f"Fatal Error: 'shape_predictor_68_face_landmarks.dat' not found at {predictor_path}")
    print("Please ensure 'face_recognition_models' is correctly installed, or manually provide the path.")
    exit()
if not os.path.exists(face_encoder_path):
    print(f"Fatal Error: 'dlib_face_recognition_resnet_model_v1.dat' not found at {face_encoder_path}")
    print("Please ensure 'face_recognition_models' is correctly installed, or manually provide the path.")
    exit()

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_encoder = dlib.face_recognition_model_v1(face_encoder_path)
# --- END Dlib models initialization ---

# --- Ensure data directory exists ---
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MONTHLY_ATTENDANCE_DIR, exist_ok=True)

# --- Shift Configuration ---
SHIFT_TIMES = {
    "9AM-3PM Shift": datetime.strptime("09:00:00", "%H:%M:%S").time(),
    "12PM-6PM Shift": datetime.strptime("12:00:00", "%H:%M:%S").time(), # Changed from 13:00:00 to 12:00:00
    "3PM-9PM Shift": datetime.strptime("15:00:00", "%H:%M:%S").time(),
    "Overnight Shift": datetime.strptime("21:00:00", "%H:%M:%S").time(),
}

CLOCK_IN_WINDOW_MINUTES = 30
CLOCK_OUT_MIN_DURATION_MINUTES = 30 # Clock out allowed after this duration from shift start

# --- Function to load known faces from CSV (Includes Workstation) and build Faiss index ---
def load_known_faces(csv_path):
    known_face_encodings = []
    known_face_names = []
    known_employee_ids = []
    known_workstations = []

    if not os.path.exists(csv_path):
        print(f"Error: Employee faces CSV not found at {csv_path}. Please run 01_enroll_faces.py or 01b_batch_enroll_faces.py first.")
        return [], [], [], [], None

    print(f"Loading known faces from {csv_path}...")
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        try:
            id_idx = header.index('employee_id')
            name_idx = header.index('name')
            encoding_idx = header.index('face_encoding')
            workstation_idx = header.index('workstation')
        except ValueError as e:
            print(f"Error: Missing expected column in {csv_path}. {e}")
            print("Please ensure the CSV has 'employee_id', 'name', 'face_encoding', and 'workstation' columns.")
            return [], [], [], [], None

        for row in reader:
            if len(row) > workstation_idx:
                employee_id = row[id_idx]
                name = row[name_idx]
                encoding_str = row[encoding_idx]
                workstation = row[workstation_idx]

                try:
                    face_encoding = np.fromstring(encoding_str, sep=',', dtype=float)
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)
                    known_employee_ids.append(employee_id)
                    known_workstations.append(workstation)
                except ValueError:
                    print(f"Warning: Skipping malformed encoding for {name} (ID: {employee_id}).")
            else:
                print(f"Warning: Skipping malformed row in CSV: {row}")

    print(f"Successfully loaded {len(known_face_encodings)} known faces.")

    faiss_index = None
    if known_face_encodings:
        embedding_dim = known_face_encodings[0].shape[0]
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index.add(np.array(known_face_encodings).astype('float32'))
        print(f"Faiss index created with {faiss_index.ntotal} vectors.")
    else:
        print("No face encodings to build Faiss index from.")

    return known_face_encodings, known_face_names, known_employee_ids, known_workstations, faiss_index

# --- Function to determine the nominal shift based on time ---
def get_shift_from_time(dt):
    hour = dt.hour

    if 9 <= hour < 12:
        return "9AM-3PM Shift"
    elif 12 <= hour < 15: # This range now correctly covers 12 PM onwards
        return "12PM-6PM Shift"
    elif 15 <= hour < 21:
        return "3PM-9PM Shift"
    elif 21 <= hour <= 23 or 0 <= hour < 9:
        return "Overnight Shift"
    else:
        return "Unassigned Shift"

# --- Function to check if current time is within clock-in window ---
def is_within_clock_in_window(current_dt, shift_name):
    if shift_name not in SHIFT_TIMES:
        return False, "Invalid Shift"

    shift_start_time = SHIFT_TIMES[shift_name]
    shift_start_dt = datetime.combine(current_dt.date(), shift_start_time)

    window_start = shift_start_dt - timedelta(minutes=CLOCK_IN_WINDOW_MINUTES)
    window_end = shift_start_dt + timedelta(minutes=CLOCK_IN_WINDOW_MINUTES)
    
    if window_start <= current_dt <= window_end:
        return True, "On Time"
    elif current_dt < window_start:
        return False, "Too Early"
    else: # current_dt > window_end
        return False, "Too Late"

# --- Function to calculate working hours ---
def calculate_working_hours(punch_in_str, punch_out_str):
    try:
        if not punch_out_str or pd.isna(punch_out_str):
            return ""

        punch_in = datetime.strptime(punch_in_str, "%H:%M:%S").time()
        punch_out = datetime.strptime(punch_out_str, "%H:%M:%S").time()

        dummy_date = date(1, 1, 1)
        dt_punch_in = datetime.combine(dummy_date, punch_in)
        dt_punch_out = datetime.combine(dummy_date, punch_out)

        if dt_punch_out < dt_punch_in:
            dt_punch_out += timedelta(days=1)

        duration = dt_punch_out - dt_punch_in
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return "N/A"

def log_attendance(employee_id, employee_name, workstation, shift, is_in_clock_in_window, clock_in_window_status):
    current_dt = datetime.now()
    current_date_str = current_dt.strftime("%Y-%m-%d")
    current_time_str = current_dt.strftime("%H:%M:%S")
    month_year_str = current_dt.strftime("%Y-%m")
    
    output_filename = os.path.join(MONTHLY_ATTENDANCE_DIR, f"attendance_report_{month_year_str}.xlsx")

    excel_columns = ['emp id', 'emp name', 'date', 'punch in', 'punch out', 'working hours', 'shift', 'workstation']
    
    df_attendance = pd.DataFrame(columns=excel_columns) # Initialize as empty DataFrame with correct columns

    # Load existing attendance data if the file exists
    if os.path.exists(output_filename):
        try:
            df_attendance = pd.read_excel(output_filename)
            # Ensure 'date' column is explicitly converted to string for reliable comparison
            # This is crucial to avoid mismatches due to date object vs string comparison
            df_attendance['date'] = df_attendance['date'].astype(str).str.slice(0, 10) # Get YYYY-MM-DD
        except Exception as e:
            print(f"Error reading existing Excel file {output_filename}: {e}. Starting with a new (empty) DataFrame.")
            df_attendance = pd.DataFrame(columns=excel_columns) # Re-initialize to empty if read fails

    # Debugging: Print initial DataFrame state and target values
    print(f"\n--- LOG_ATTENDANCE DEBUG ---")
    print(f"Current emp_id: {employee_id}, Current date: {current_date_str}")
    # print(f"df_attendance before query:\n{df_attendance.to_string()}") # Commented out for less verbose output

    # Find the existing record for this employee for the current day
    existing_record_query = df_attendance[
        (df_attendance['emp id'].astype(str) == str(employee_id)) & # Ensure string comparison
        (df_attendance['date'] == current_date_str)
    ]
    
    # Debugging: Print the result of the query
    # print(f"Existing record query result (count: {len(existing_record_query)}):\n{existing_record_query.to_string()}") # Commented out for less verbose output

    message_lines = []
    log_action = ""
    should_save_df = False # Flag to control when to save the DataFrame

    if existing_record_query.empty:
        # Case 1: No record found for this employee for today. This is a potential NEW Clock In.
        print(f"DEBUG: No existing record found for {employee_id} on {current_date_str}. Proceeding with NEW record creation.")
        if is_in_clock_in_window:
            new_record = {
                'emp id': employee_id,
                'emp name': employee_name,
                'date': current_date_str,
                'punch in': current_time_str,
                'punch out': '', # Initial clock in, punch out is empty
                'working hours': '',
                'shift': shift,
                'workstation': workstation
            }
            # Add the new record. This is the ONLY place a new row is explicitly added.
            df_attendance = pd.concat([df_attendance, pd.DataFrame([new_record], columns=excel_columns)], ignore_index=True)
            log_action = "Clock In"
            message_lines = [f"Welcome, {employee_name}!", "Clocked In. ✅"]
            should_save_df = True
            print(f"DEBUG: Successfully added new clock-in record for {employee_id}.")
        else:
            # Attempt to clock in outside the allowed window (too early/late)
            message_lines = [f"Hello, {employee_name},"]
            if clock_in_window_status == "Too Early":
                if shift in SHIFT_TIMES:
                    shift_start_time_obj = SHIFT_TIMES[shift]
                    clock_in_start = datetime.combine(current_dt.date(), shift_start_time_obj) - timedelta(minutes=CLOCK_IN_WINDOW_MINUTES)
                    clock_in_end = datetime.combine(current_dt.date(), shift_start_time_obj) + timedelta(minutes=CLOCK_IN_WINDOW_MINUTES)
                    
                    message_lines.append("You cannot clock in yet.")
                    message_lines.append(f"Your {shift} starts at {shift_start_time_obj.strftime('%I:%M %p')}.")
                    message_lines.append(f"You can clock in from {clock_in_start.strftime('%I:%M %p')} to {clock_in_end.strftime('%I:%M %p')}. ⏰")
                else:
                    message_lines.append("Shift not recognized for current time.")
                    message_lines.append("Please contact support. ❓")
            elif clock_in_window_status == "Too Late":
                message_lines.append("You are outside the allowed clock-in window. ⏳")
                message_lines.append("Please contact your supervisor for manual attendance.")
            else:
                message_lines.append("Shift timing not defined or recognized for this moment.")
                message_lines.append("Please contact support. ❓")
            log_action = f"Clock In Attempt ({clock_in_window_status})"
            should_save_df = False # No DataFrame change, so no save needed.
            print(f"DEBUG: Clock-in attempt outside window for {employee_id}.")
    else:
        # Case 2: An existing record for this employee for today already exists.
        # Get the index of the first (and ideally only) matching record.
        # This will be the row we UPDATE.
        idx = existing_record_query.index[0]  
        existing_punch_in = df_attendance.at[idx, 'punch in']
        existing_punch_out = df_attendance.at[idx, 'punch out']

        print(f"DEBUG: Existing record found at index {idx} for {employee_id}.")
        print(f"DEBUG: Existing punch in: {existing_punch_in}, Existing punch out: {existing_punch_out}")

        if pd.isna(existing_punch_in) or not existing_punch_in:
            # This scenario indicates an incomplete or malformed existing record.
            # It should ideally be prevented by the initial clock-in logic.
            log_action = "Error: Incomplete existing record (no Punch In). Contact support."
            message_lines = ["Error: Incomplete record.", "Please contact support. ❌"]
            should_save_df = False
            print(f"DEBUG: Error: Malformed existing record for {employee_id}.")
        elif pd.isna(existing_punch_out) or not existing_punch_out:
            # Employee has punched in, but not yet out. This is a potential Clock Out.

            nominal_shift_start_time_obj = SHIFT_TIMES.get(shift)
            if nominal_shift_start_time_obj:
                # Use the date from the existing record to ensure consistency for shift start time.
                # Convert the date string from the DataFrame back to a date object for timedelta calculations
                record_date = datetime.strptime(df_attendance.at[idx, 'date'], "%Y-%m-%d").date()
                
                shift_start_on_record_date = datetime.combine(record_date, nominal_shift_start_time_obj)
                earliest_clock_out_datetime = shift_start_on_record_date + timedelta(minutes=CLOCK_OUT_MIN_DURATION_MINUTES)
                
                # Special handling for overnight shifts if current time crosses midnight.
                # Adjust earliest_clock_out_datetime to the next day if the shift spans midnight
                if nominal_shift_start_time_obj.hour >= 20 and current_dt.hour < 9: # Roughly 8 PM or later start, and current time is next morning
                        earliest_clock_out_datetime = earliest_clock_out_datetime + timedelta(days=1)
                elif current_dt.date() > record_date and nominal_shift_start_time_obj.hour < 9: # if shift starts before 9 AM and current date is next day for an overnight shift
                        earliest_clock_out_datetime = earliest_clock_out_datetime + timedelta(days=1)


                # Check if current time is before the earliest allowed clock-out time.
                if current_dt < earliest_clock_out_datetime:
                    log_action = "Clock Out Attempt (Too Early)"
                    message_lines = [
                        f"Hello, {employee_name}!",
                        "You've already clocked in for today.",
                        f"You can only clock out after {earliest_clock_out_datetime.strftime('%I:%M %p')}. ⏰"
                    ]
                    should_save_df = False
                    print(f"DEBUG: Clock-out attempt too early for {employee_id}.")
                else:
                    # Valid Clock Out: Update the existing row directly.
                    df_attendance.at[idx, 'punch out'] = current_time_str
                    df_attendance.at[idx, 'working hours'] = calculate_working_hours(
                        existing_punch_in, current_time_str
                    )
                    log_action = "Clock Out"
                    message_lines = [f"Goodbye, {employee_name}!", "Clocked Out. ✅"]
                    should_save_df = True
                    print(f"DEBUG: Successfully updated clock-out for {employee_id} at index {idx}.")
            else:
                # Fallback if shift time isn't found for the existing record (unlikely if initial logic is sound).
                log_action = "Clock Out (Shift Info Missing)"
                message_lines = [f"Goodbye, {employee_name}!", "Clocked Out (Shift time unknown). ✅"]
                df_attendance.at[idx, 'punch out'] = current_time_str
                df_attendance.at[idx, 'working hours'] = calculate_working_hours(
                    existing_punch_in, current_time_str
                )
                should_save_df = True
                print(f"DEBUG: Clock-out for {employee_id} (shift info missing).")
        else:
            # Employee has already punched both in and out for today. No action needed.
            log_action = "Already Logged (In & Out)"
            message_lines = [
                f"Hello, {employee_name}!",
                "You've already clocked in & out for today.",
                "No further action needed. 👍"
            ]
            should_save_df = False # No DataFrame change, so no save needed.
            print(f"DEBUG: {employee_id} already clocked in and out for today.")

    # Save the DataFrame ONLY if a modification was made (new clock-in or clock-out update).
    if should_save_df:
        try:
            df_attendance.to_excel(output_filename, index=False)
            print(f"\n--- ATTENDANCE LOGGED & EXCEL UPDATED ---")
            print(f"Timestamp: {current_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Employee ID: {employee_id}")
            print(f"Name: {employee_name}")
            print(f"Action: {log_action}")
            print(f"Workstation: {workstation}")
            print(f"Shift: {shift}")
            print(f"Excel file updated: {output_filename}")
            print(f"-----------------------------------------\n")
            # print(f"DEBUG: df_attendance AFTER save:\n{df_attendance.to_string()}") # Commented out for less verbose output

        except Exception as e:
            print(f"ERROR: Could not save attendance to Excel file {output_filename}: {e}")
            message_lines = ["ERROR: Failed to save attendance!", "Please contact support. ❌"]
    else:
        # If no save occurred, still print the action to the console for debugging/monitoring.
        print(f"\n--- ATTENDANCE ACTION ---")
        print(f"Timestamp: {current_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Employee ID: {employee_id}")
        print(f"Name: {employee_name}")
        print(f"Action: {log_action} (No Excel update needed)")
        print(f"-------------------------\n")

    print(f"--- END LOG_ATTENDANCE DEBUG ---")
    return message_lines, log_action


# --- Function to get today's attendance for a specific employee (MODIFIED for Excel) ---
# This function is now less critical as log_attendance manages state, but useful for display logic.
def get_todays_attendance_for_employee(employee_id):
    current_dt = datetime.now()
    current_date_str = current_dt.strftime("%Y-%m-%d")
    month_year_str = current_dt.strftime("%Y-%m")
    
    output_filename = os.path.join(MONTHLY_ATTENDANCE_DIR, f"attendance_report_{month_year_str}.xlsx")

    last_clock_in_shift = "N/A"
    last_status = None # Can be "Clock In" or "Clock Out" (if punch_out is filled)

    if not os.path.exists(output_filename):
        return last_clock_in_shift, last_status

    try:
        df = pd.read_excel(output_filename)
        # Ensure 'date' column is string formatted for direct comparison
        df['date'] = df['date'].astype(str).str.slice(0, 10)  

        employee_today_record = df[
            (df['emp id'].astype(str) == str(employee_id)) &
            (df['date'] == current_date_str)
        ]

        if not employee_today_record.empty:
            record = employee_today_record.iloc[0] # Get the first (and should be only) record for today
            last_clock_in_shift = record['shift']
            if pd.isna(record['punch out']) or record['punch out'] == '':
                last_status = "Clock In" # Means they punched in but not out
            else:
                last_status = "Clock Out" # Means they punched in and out
    except Exception as e:
        print(f"Warning: Could not read attendance from {output_filename} for status check: {e}")

    return last_clock_in_shift, last_status


# Function to detect and return coordinates of an extended screen
def get_extended_screen_coordinates():
    monitors = get_monitors()
    if len(monitors) <= 1:
        print("Only one monitor detected. Window will remain on primary screen.")
        return None, None
    for m in monitors:
        if m.x != 0 or m.y != 0:
            print(f"Detected extended screen '{m.name}' at X:{m.x}, Y:{m.y} with size {m.width}x{m.height}")
            return m.x, m.y
    print("Multiple monitors detected, but no obvious extended screen found (all start at 0,0 or similar). Window will remain on primary.")
    return None, None


# --- Main Attendance System Logic ---
def run_attendance_system():
    global successful_attendance_message_lines, successful_attendance_message_until
    global last_recognized_time

    known_face_encodings, known_face_names, known_employee_ids, known_workstations, faiss_index = load_known_faces(EMPLOYEE_FACES_CSV)

    if not known_face_encodings or faiss_index is None:
        print("No known faces or Faiss index could be loaded. Exiting.")
        return

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open webcam. Please ensure it's connected and not in use.")
        return

    # Create the window before setting properties
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)  
    # Set the window to full screen
    cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n--- Attendance System Started ---")
    print("Look at the camera to mark your attendance.")
    print("Press 'q' to quit.")

    window_moved = False

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        current_frame_faces_to_draw = {}
        detected_faces = face_detector(rgb_small_frame, 1)

        is_any_recognized_face_in_frame = False
        message_to_render_this_frame = []

        if datetime.now() < successful_attendance_message_until:
            message_to_render_this_frame = successful_attendance_message_lines
        else:
            if detected_faces:
                face_locations = []
                face_encodings = []

                for dlib_face in detected_faces:
                    top = dlib_face.top() * 4
                    right = dlib_face.right() * 4
                    bottom = dlib_face.bottom() * 4
                    left = dlib_face.left() * 4
                    face_locations.append((top, right, bottom, left))

                    shape = shape_predictor(rgb_small_frame, dlib_face)
                    face_encoding = np.array(face_encoder.compute_face_descriptor(rgb_small_frame, shape))
                    face_encodings.append(face_encoding)

                for i, current_face_encoding in enumerate(face_encodings):
                    query_encoding = current_face_encoding.astype('float32').reshape(1, -1)
                    distances, indices = faiss_index.search(query_encoding, FAISS_K_NEIGHBORS)
                    valid_indices = indices[0][indices[0] != -1]

                    name = "Unknown"
                    employee_id = "N/A"
                    workstation = "N/A"

                    if len(valid_indices) > 0:
                        candidate_encodings = [known_face_encodings[j] for j in valid_indices]
                        face_distances_candidates = face_recognition.face_distance(candidate_encodings, current_face_encoding)
                        matches = face_recognition.compare_faces(candidate_encodings, current_face_encoding, tolerance=FACE_RECOGNITION_TOLERANCE)

                        if True in matches:
                            best_match_candidate_index = np.argmin(face_distances_candidates)
                            original_index = valid_indices[best_match_candidate_index]

                            employee_id = known_employee_ids[original_index]
                            name = known_face_names[original_index]
                            workstation = known_workstations[original_index]

                            current_dt = datetime.now()
                            is_any_recognized_face_in_frame = True

                            if employee_id in last_recognized_time and \
                               (current_dt - last_recognized_time[employee_id]).total_seconds() < COOLDOWN_SECONDS:
                                display_name = f"{name} (Cooldown)"
                            else:
                                last_recognized_time[employee_id] = current_dt

                                current_shift_name = get_shift_from_time(current_dt)
                                is_in_window, clock_in_window_status = is_within_clock_in_window(current_dt, current_shift_name)

                                # Pass clock-in window status directly to log_attendance for decision making
                                message_lines_to_set, current_status_action = log_attendance(
                                    employee_id, name, workstation, current_shift_name, is_in_window, clock_in_window_status
                                )
                                successful_attendance_message_lines = message_lines_to_set
                                successful_attendance_message_until = datetime.now() + timedelta(seconds=MESSAGE_DISPLAY_DURATION_SECONDS)
                                
                                display_name = name # Display actual name after attempt

                        else:
                            display_name = "Unknown"
                            if datetime.now() >= successful_attendance_message_until:
                                message_to_render_this_frame = ["Face Not Recognized.", "Please try again. 🚫"]
                    else:
                        display_name = "Unknown"
                        if datetime.now() >= successful_attendance_message_until:
                            message_to_render_this_frame = ["Face Not Recognized.", "Please try again. 🚫"]

                    current_frame_faces_to_draw[face_locations[i]] = display_name

                if not successful_attendance_message_lines and not is_any_recognized_face_in_frame and datetime.now() >= successful_attendance_message_until:
                    message_to_render_this_frame = ["Ready for Scan.", "Please step in front of the camera."]
                elif not successful_attendance_message_lines and is_any_recognized_face_in_frame and datetime.now() >= successful_attendance_message_until and not message_to_render_this_frame:
                    message_to_render_this_frame = [] # Clear message if recognized and no specific action triggered
            else:
                if datetime.now() >= successful_attendance_message_until:
                    message_to_render_this_frame = ["Ready for Scan.", "Please step in front of the camera."]
                else:
                    message_to_render_this_frame = successful_attendance_message_lines

        # --- Drawing Bounding Boxes and Names ---
        for (top, right, bottom, left), display_name in current_frame_faces_to_draw.items():
            color = (0, 0, 255) # Red for unknown/detecting
            if display_name != "Unknown" and "Cooldown" not in display_name:
                color = (0, 255, 0) # Green for recognized and active
            elif "Cooldown" in display_name:
                color = (255, 255, 0) # Yellow for recognized but in cooldown

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, display_name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

        # --- MODIFIED: Display on-screen messages ---
        if message_to_render_this_frame:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7 # Reduced font size
            font_thickness = 2
            initial_y_pos = 50 # Start from 50 pixels from the top

            y_offset = 0
            for line_text in message_to_render_this_frame:
                text_size = cv2.getTextSize(line_text, font, font_scale, font_thickness)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2 # Center horizontally
                
                # Calculate vertical position based on initial_y_pos and accumulated offset
                text_y = initial_y_pos + text_size[1] + y_offset

                # Add a small padding between lines based on font height
                line_spacing = int(text_size[1] * 0.4) # 40% of font height as spacing

                cv2.putText(frame, line_text, (text_x, text_y), font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
                y_offset += text_size[1] + line_spacing # Increment offset for the next line

        # Show the resulting image
        cv2.imshow('Video', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    print("\n--- Attendance System Stopped ---")

if __name__ == "__main__":
    run_attendance_system()