import cv2
import numpy as np
import face_recognition
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_metadata = {}
        
    def add_known_face(self, image: np.ndarray, name: str, metadata: Dict = None):
        """Add a known face to the recognition system."""
        # Get face encoding
        face_encoding = face_recognition.face_encodings(image)
        
        if face_encoding:
            self.known_face_encodings.append(face_encoding[0])
            self.known_face_names.append(name)
            self.known_face_metadata[name] = metadata or {}
            return True
        return False
    
    def recognize_face(self, face_image: np.ndarray) -> Optional[Dict]:
        """Recognize a face and return its information."""
        # Get face encoding
        face_encoding = face_recognition.face_encodings(face_image)
        
        if not face_encoding:
            return None
            
        # Compare with known faces
        matches = face_recognition.compare_faces(
            self.known_face_encodings, 
            face_encoding[0]
        )
        
        if True in matches:
            match_index = matches.index(True)
            name = self.known_face_names[match_index]
            metadata = self.known_face_metadata[name]
            
            return {
                'name': name,
                'metadata': metadata,
                'confidence': self._calculate_confidence(
                    face_encoding[0],
                    self.known_face_encodings[match_index]
                )
            }
        
        return None
    
    def _calculate_confidence(self, encoding1, encoding2) -> float:
        """Calculate confidence score between two face encodings."""
        distance = np.linalg.norm(encoding1 - encoding2)
        return max(0, min(100, 100 * (1 - distance))) 