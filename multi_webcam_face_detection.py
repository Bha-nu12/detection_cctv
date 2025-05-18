import cv2
import numpy as np
import json
import os
import pyttsx3
import time
from test_face_detection import FaceDatabase, FaceDetector

def open_camera(index):
    """Try to open camera with different backends"""
    # Try DirectShow first
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Try default backend
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            # Try MSMF backend
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            if not cap.isOpened():
                return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def draw_controls(frame, current_camera):
    """Draw control instructions on the frame"""
    controls = [
        "Controls:",
        "Press '1': Registration Camera",
        "Press '2': Recognition Camera 1",
        "Press '3': Recognition Camera 2",
        "Press 's': Save face (in Registration Camera)",
        "Press 'q': Quit"
    ]
    
    # Draw semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw text
    for i, text in enumerate(controls):
        y = 40 + i * 25
        color = (0, 255, 0) if text.startswith(f"Press '{current_camera}'") else (255, 255, 255)
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw current mode
    mode_text = "Current Mode: Registration" if current_camera == '1' else f"Current Mode: Recognition Camera {current_camera}"
    cv2.putText(frame, mode_text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def main():
    # Camera indices
    CAMERA_INDICES = [0, 1, 3]
    CAMERA_NAMES = [f"Webcam {i}" for i in CAMERA_INDICES]

    # Initialize face database and detector with higher confidence threshold
    face_db = FaceDatabase('face_database.json')
    face_detector = FaceDetector(confidence_threshold=0.7)  # Increased confidence threshold
    engine = pyttsx3.init()

    # Open all cameras
    print("\nAttempting to open cameras...")
    caps = []
    for idx in CAMERA_INDICES:
        print(f"Trying to open camera {idx}...")
        cap = open_camera(idx)
        if cap is None:
            print(f"Warning: Could not open camera {idx}")
            caps.append(None)
        else:
            print(f"Successfully opened camera {idx}")
            caps.append(cap)

    # Check if at least one camera is working
    if all(cap is None for cap in caps):
        print("\nError: No cameras could be opened. Please check your camera connections and try again.")
        exit(1)

    current_camera = '1'  # Start with registration camera
    print("\nControls:")
    print("  - Press '1' for Registration Camera")
    print("  - Press '2' for Recognition Camera 1")
    print("  - Press '3' for Recognition Camera 2")
    print("  - Press 's' to save face (only in Registration Camera)")
    print("  - Press 'q' to quit\n")
    print("Starting in Registration mode (Camera 1)")

    # Variables for face registration
    registration_samples = []
    last_sample_time = 0
    SAMPLE_INTERVAL = 0.5  # Time between samples in seconds
    REQUIRED_SAMPLES = 5   # Number of samples needed for registration

    while True:
        for i, (cap, name) in enumerate(zip(caps, CAMERA_NAMES)):
            if cap is None:
                continue
            ret, frame = cap.read()
            if not ret:
                # Show black frame if camera fails
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Camera {i} not available", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(name, frame)
                continue
                
            # Draw controls on the frame
            draw_controls(frame, current_camera)
            
            faces = face_detector.detect_faces(frame)
            for face in faces:
                # Get face coordinates from the face detection object
                x, y, w, h = face.bbox
                
                # Draw face rectangle
                color = (0, 255, 0)  # Green for detected faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                if current_camera == '1':
                    # Registration camera
                    current_time = time.time()
                    if len(registration_samples) < REQUIRED_SAMPLES and current_time - last_sample_time >= SAMPLE_INTERVAL:
                        registration_samples.append(face.embeddings)
                        last_sample_time = current_time
                        print(f"Sample {len(registration_samples)}/{REQUIRED_SAMPLES} collected")
                    
                    # Show registration progress
                    progress = f"Registration: {len(registration_samples)}/{REQUIRED_SAMPLES} samples"
                    cv2.putText(frame, progress, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                else:
                    # Recognition cameras
                    match = face_db.find_match(face.embeddings)
                    if match:
                        face_id, name, similarity = match
                        # Only show high confidence matches
                        if similarity > 0.6:  # Increased similarity threshold
                            label = f"{name} ({similarity:.2f})"
                            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            # Announce name
                            engine.say(f"{name} detected on {name}")
                            engine.runAndWait()
                        else:
                            label = "Unknown"
                            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        label = "Unknown"
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow(name, frame)

        key = cv2.waitKey(1) & 0xFF
        # Handle camera switching
        if key == ord('1'):
            current_camera = '1'
            registration_samples = []  # Reset samples when switching to registration
            print("\nSwitched to Registration mode")
        elif key == ord('2'):
            current_camera = '2'
            print("\nSwitched to Recognition Camera 1")
        elif key == ord('3'):
            current_camera = '3'
            print("\nSwitched to Recognition Camera 2")
        # Save face from registration camera
        elif key == ord('s'):
            if current_camera == '1':
                if caps[0] is not None:
                    ret, frame = caps[0].read()
                    if ret:
                        faces = face_detector.detect_faces(frame)
                        if faces and len(registration_samples) >= REQUIRED_SAMPLES:
                            print("\nEnter name for this face (or press Enter to skip): ", end='', flush=True)
                            name = input()
                            if name:
                                # Use average of all samples for better accuracy
                                avg_embedding = np.mean(registration_samples, axis=0)
                                face_id = str(int(time.time() * 1000))
                                face_db.add_face(face_id, name, avg_embedding)
                                print(f"Saved face for {name} using {len(registration_samples)} samples")
                                registration_samples = []  # Reset samples after saving
                            else:
                                print("Skipped registration.")
                        else:
                            print(f"Need {REQUIRED_SAMPLES} samples for registration. Current samples: {len(registration_samples)}")
            else:
                print("\nYou must be in Registration mode (Camera 1) to save faces")
        elif key == ord('q'):
            break

    # Cleanup
    for cap in caps:
        if cap is not None:
            cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 