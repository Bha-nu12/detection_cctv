import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import uuid

@dataclass
class FaceDetection:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    embeddings: np.ndarray
    face_id: str = None  # UUID for face identification
    pose: str = "frontal"  # frontal, left_profile, right_profile

class FaceDetector:
    def __init__(self, confidence_threshold: float = 0.5):
        # Load face detection model
        self.face_detector = cv2.dnn.readNetFromCaffe(
            'models/deploy.prototxt',
            'models/res10_300x300_ssd_iter_140000.caffemodel'
        )
        self.confidence_threshold = confidence_threshold
        
        # Load face recognition model
        self.face_recognizer = cv2.dnn.readNetFromTorch('models/nn4.small2.v1.t7')
        
        # Store known faces with their IDs
        self.known_faces: Dict[str, np.ndarray] = {}
        
    def add_known_face(self, face_id: str, embedding: np.ndarray):
        """Add a known face to the database."""
        self.known_faces[face_id] = embedding
        
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a frame and compute their embeddings.
        
        Args:
            frame: Input image in BGR format
        
        Returns:
            List of FaceDetection objects containing bbox, confidence, and embeddings
        """
        height, width = frame.shape[:2]
        
        # Prepare input blob for face detection
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300),
            (104, 177, 123)
        )
        
        # Detect faces
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # Get bounding box
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Ensure box is within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # Extract face ROI
                face_roi = frame[y1:y2, x1:x2]
                
                # Compute face embeddings
                face_blob = cv2.dnn.blobFromImage(
                    face_roi, 
                    1.0/255, 
                    (96, 96), 
                    (0, 0, 0), 
                    swapRB=True, 
                    crop=False
                )
                self.face_recognizer.setInput(face_blob)
                embeddings = self.face_recognizer.forward()
                
                # Generate new face ID if not found in known faces
                face_id = None
                embedding = embeddings.flatten()
                
                # Check if this face matches any known face
                for known_id, known_embedding in self.known_faces.items():
                    similarity = self.compute_similarity(known_embedding, embedding)
                    if similarity > 0.6:  # Threshold for matching
                        face_id = known_id
                        break
                
                # If no match found, generate new ID
                if face_id is None:
                    face_id = str(uuid.uuid4())
                    self.add_known_face(face_id, embedding)
                
                faces.append(FaceDetection(
                    bbox=(x1, y1, x2-x1, y2-y1),
                    confidence=float(confidence),
                    embeddings=embedding,
                    face_id=face_id,
                    pose="frontal"  # Simplified pose detection
                ))
                
        return faces
        
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity) 