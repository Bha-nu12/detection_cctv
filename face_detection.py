import cv2
import numpy as np
import time
import os
from test_face_detection import FaceDatabase, FaceDetector

# Suppress OpenCV warnings
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

def open_camera():
    """Open camera with DirectShow backend on Windows"""
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap
    except:
        return None

print("Initializing face detection model...")
try:
    face_db = FaceDatabase('face_database.json')
    face_detector = FaceDetector(confidence_threshold=0.5)
    print("Face detection model loaded successfully!")
except Exception as e:
    print(f"Error loading face detection model: {str(e)}")
    print("Please make sure all model files are present in the models directory")
    exit(1)

# Initialize camera
print("\nInitializing camera...")
cap = open_camera()
if cap is None:
    print("No camera found. Please check your camera connection.")
    exit(1)

print("Camera initialized successfully!")

# Initialize variables
is_registration_mode = True
registration_samples = []
last_sample_time = 0
SAMPLE_INTERVAL = 0.5
REQUIRED_SAMPLES = 5
last_unknown_time = 0
UNKNOWN_INTERVAL = 5  # Ask for unknown person's name every 5 seconds

print("\nControls:")
print("  - Press '1' to switch to Registration mode")
print("  - Press '2' to switch to Recognition mode")
print("  - Press 's' to save face (only in Registration mode)")
print("  - Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break

        # Add face detection guide
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        guide_size = min(width, height) // 3
        cv2.rectangle(frame, 
                    (center_x - guide_size//2, center_y - guide_size//2),
                    (center_x + guide_size//2, center_y + guide_size//2),
                    (0, 255, 255), 2)  # Yellow guide box

        try:
            faces = face_detector.detect_faces(frame)
            for face in faces:
                bbox = face.bbox
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Thicker green line
                
                if is_registration_mode:  # Registration mode
                    current_time = time.time()
                    if len(registration_samples) < REQUIRED_SAMPLES and current_time - last_sample_time >= SAMPLE_INTERVAL:
                        registration_samples.append(face.embeddings)
                        last_sample_time = current_time
                        print(f"Sample collected: {len(registration_samples)}/{REQUIRED_SAMPLES}")
                    
                    if len(registration_samples) < REQUIRED_SAMPLES:
                        cv2.putText(frame, f"Position face in yellow box ({len(registration_samples)}/{REQUIRED_SAMPLES})", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, "Press 's' to save face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:  # Recognition mode
                    match = face_db.find_match(face.embeddings)
                    current_time = time.time()
                    if match:
                        _, name, similarity = match
                        if similarity > 0.6:
                            label = f"{name} ({similarity:.2f})"
                            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            # Ask for unknown person's name
                            if current_time - last_unknown_time >= UNKNOWN_INTERVAL:
                                name = input("Unknown person detected. Enter name (or press Enter to skip): ")
                                if name:
                                    face_id = str(int(time.time() * 1000))
                                    face_db.add_face(face_id, name, face.embeddings)
                                    print(f"Face saved for {name}")
                                last_unknown_time = current_time
                    else:
                        cv2.putText(frame, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            continue

        # Show mode indicator
        mode_text = "Registration Mode" if is_registration_mode else "Recognition Mode"
        cv2.putText(frame, mode_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            is_registration_mode = True
            registration_samples = []
            print("Switched to Registration mode")
        elif key == ord('2'):
            is_registration_mode = False
            print("Switched to Recognition mode")
        elif key == ord('s'):
            if is_registration_mode:
                print(f"Debug: Samples collected: {len(registration_samples)}")
                if len(registration_samples) >= REQUIRED_SAMPLES:
                    name = input("Enter name for this face (or press Enter to skip): ")
                    if name:
                        try:
                            avg_embedding = np.mean(registration_samples, axis=0)
                            face_id = str(int(time.time() * 1000))
                            face_db.add_face(face_id, name, avg_embedding)
                            print(f"Face saved for {name}")
                            registration_samples = []
                        except Exception as e:
                            print(f"Error saving face: {str(e)}")
                    else:
                        print("Registration skipped")
                else:
                    print("Please wait for face registration to complete")
                    print(f"Current samples: {len(registration_samples)}/{REQUIRED_SAMPLES}")
            else:
                print("You must be in Registration mode to save faces")
        elif key == ord('q'):
            print("Quitting...")
            break

except KeyboardInterrupt:
    
    print("\nProgram interrupted by user")
except Exception as e:
    print(f"\nAn error occurred: {str(e)}")
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows() 