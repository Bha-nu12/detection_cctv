import streamlit as st
import cv2
import numpy as np
from test_face_detection import FaceDatabase, FaceDetector
import time
import pyttsx3
import threading
from sklearn.metrics import f1_score

# Initialize session state
if 'last_announcement' not in st.session_state:
    st.session_state.last_announcement = {}
if 'announcement_cooldown' not in st.session_state:
    st.session_state.announcement_cooldown = 5  # seconds between announcements
if 'true_labels' not in st.session_state:
    st.session_state.true_labels = []
if 'pred_labels' not in st.session_state:
    st.session_state.pred_labels = []

# Constants
CAMERA_INDICES = [0]  # Only use camera 0 since it's the only one available
CAMERA_NAMES = [f"Webcam {i}" for i in CAMERA_INDICES]

def open_camera(index):
    """Try to open camera with different backends"""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            if not cap.isOpened():
                return None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def announce_detection(name, camera_name):
    """Announce face detection using text-to-speech"""
    current_time = time.time()
    if name not in st.session_state.last_announcement or \
            current_time - st.session_state.last_announcement[name] >= st.session_state.announcement_cooldown:
        engine = pyttsx3.init()
        engine.say(f"{name} detected on {camera_name}")
        engine.runAndWait()
        st.session_state.last_announcement[name] = current_time

def process_frame(frame, face_detector, face_db, camera_name):
    """Process a single frame for face detection and recognition"""
    faces = face_detector.detect_faces(frame)
    processed_frame = frame.copy()
    detected_names = []
    
    for face in faces:
        x, y, w, h = face.bbox
        color = (0, 255, 0)
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
        
        match = face_db.find_match(face.embeddings)
        if match:
            face_id, name, similarity = match
            if similarity > 0.6:
                label = f"{name}"
                cv2.putText(processed_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                detected_names.append(name)
                announce_detection(name, camera_name)

                # Update true and predicted labels for F1 score calculation
                st.session_state.true_labels.append(name)  # Assuming 'name' is the true label
                st.session_state.pred_labels.append(name)
            else:
                label = "Unknown"
                cv2.putText(processed_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                st.session_state.true_labels.append(name)
                st.session_state.pred_labels.append("Unknown")
        else:
            label = "Unknown"
            cv2.putText(processed_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            st.session_state.true_labels.append("Unknown")
            st.session_state.pred_labels.append("Unknown")
            
    return processed_frame, detected_names

def main():
    st.set_page_config(page_title="Face Detection Dashboard", layout="wide")
    st.title("Face Detection and Recognition Dashboard")
    
    # Initialize face detection components
    face_db = FaceDatabase('face_database.json')
    face_detector = FaceDetector(confidence_threshold=0.7)
    
    # Sidebar controls
    st.sidebar.title("Controls")
    camera_mode = st.sidebar.radio(
        "Select Camera",
        ["Camera 1"],
        index=0
    )
    
    # Map camera mode to index
    camera_index = CAMERA_INDICES[int(camera_mode.split()[-1]) - 1]
    camera_name = f"Camera {camera_mode.split()[-1]}"
    
    # Open selected camera
    cap = open_camera(camera_index)
    
    if cap is None:
        st.error(f"Could not open camera {camera_index}")
        return
    
    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Camera Feed")
        frame_placeholder = st.empty()
    
    with col2:
        st.subheader("Detection Status")
        status_placeholder = st.empty()
        detection_history = st.empty()
        f1_placeholder = st.empty()  # Add a placeholder for the F1 score
    
    # Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera")
            break
        
        # Process frame
        processed_frame, detected_names = process_frame(
            frame, 
            face_detector, 
            face_db,
            camera_name
        )
        
        # Convert to RGB for display
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Update the frame
        frame_placeholder.image(processed_frame, channels="RGB")
        
        # Update status
        if detected_names:
            status_placeholder.success(f"Detected: {', '.join(detected_names)}")
            # Update detection history
            detection_history.markdown(f"""
            ### Recent Detections
            {', '.join(detected_names)} detected on {camera_name}
            """)
        else:
            status_placeholder.info("No faces detected")
            detection_history.empty()
        
        # Calculate and display F1 score
        if st.session_state.true_labels and st.session_state.pred_labels:
            f1 = f1_score(st.session_state.true_labels, st.session_state.pred_labels, average='weighted')
            f1_placeholder.metric("F1 Score", f1)
        
        time.sleep(0.1)  # Small delay to prevent overwhelming the system

if __name__ == "__main__":
    main()
