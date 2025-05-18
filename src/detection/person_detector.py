import cv2
import numpy as np
from typing import Tuple, Dict, List

class PersonDetector:
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize HOG person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Tracking data
        self.tracked_persons = {}
        
    def detect_and_track(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Detect and track persons in the frame."""
        display_frame = frame.copy()
        detections = []
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process face detections
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(display_frame, 'Face', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            detections.append({
                'type': 'face',
                'bbox': (x, y, w, h)
            })
        
        return display_frame, detections 