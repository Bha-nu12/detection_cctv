import time
from typing import Dict, List
import numpy as np
from ..streaming.camera_manager import CameraManager
from ..detection.face_detector import FaceDetector
from ..database.redis_client import RedisClient

class FrameProcessor:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.camera_manager = CameraManager()
        self.face_detector = FaceDetector()
        self.redis_client = RedisClient(redis_host, redis_port)
        
    def process_frame(self, camera_id: str) -> Dict:
        """
        Process a single frame from a camera feed.
        
        Returns:
            Dictionary containing detections and matches
        """
        frame = self.camera_manager.get_frame(camera_id)
        if frame is None:
            return {'error': 'No frame available'}
            
        # Detect faces in the frame
        detections = self.face_detector.detect_faces(frame)
        
        results = []
        timestamp = time.time()
        
        for detection in detections:
            # Find best match among known faces
            best_match = self._find_best_match(detection.embeddings)
            
            if best_match:
                person_id, similarity = best_match
                self.redis_client.update_last_seen(person_id, camera_id, timestamp)
                
                metadata = self.redis_client.get_metadata(person_id)
                results.append({
                    'person_id': person_id,
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'similarity': similarity,
                    'metadata': metadata
                })
            else:
                # Store new face if no match found
                new_person_id = f"person_{int(timestamp)}_{len(results)}"
                self.redis_client.store_face_embedding(
                    new_person_id,
                    detection.embeddings,
                    {'first_seen': timestamp,
                     'first_camera': camera_id}
                )
                results.append({
                    'person_id': new_person_id,
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'new_face': True
                })
                
        return {
            'camera_id': camera_id,
            'timestamp': timestamp,
            'detections': results
        }
    
    def _find_best_match(self, 
                        query_embedding: np.ndarray,
                        min_similarity: float = 0.6) -> tuple:
        """Find the best matching face in the database."""
        best_match = None
        best_similarity = 0
        
        # TODO: Implement efficient similarity search using Redis
        # For now, this is a simple linear search
        # In production, consider using vector similarity search
        
        return best_match 