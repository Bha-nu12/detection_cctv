import cv2
import numpy as np
import json
import os
from src.detection.face_detector import FaceDetector
from src.camera.multi_camera_manager import MultiCameraManager
import time

class FaceRecognitionSystem:
    def __init__(self):
        self.face_detector = FaceDetector(confidence_threshold=0.5)
        self.camera_manager = MultiCameraManager()
        self.face_database = {}
        self.load_face_database()
        
    def load_face_database(self):
        """Load face database from JSON file"""
        if os.path.exists('face_database.json'):
            with open('face_database.json', 'r') as f:
                self.face_database = json.load(f)
                
    def save_face_database(self):
        """Save face database to JSON file"""
        with open('face_database.json', 'w') as f:
            json.dump(self.face_database, f)
            
    def add_new_face(self, face_id, embedding, name):
        """Add a new face to the database"""
        self.face_database[face_id] = {
            'name': name,
            'embedding': embedding.tolist()
        }
        self.save_face_database()
        
    def find_matching_face(self, embedding):
        """Find the best matching face in the database"""
        best_match = None
        best_similarity = 0.6  # Minimum similarity threshold
        
        for face_id, data in self.face_database.items():
            stored_embedding = np.array(data['embedding'])
            similarity = np.dot(embedding, stored_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (face_id, data['name'])
                
        return best_match
        
    def process_frame(self, frame, camera_name):
        """Process a single frame for face detection and recognition"""
        if frame is None:
            return frame
            
        faces = self.face_detector.detect_faces(frame)
        
        for face in faces:
            x, y, w, h = face.bbox
            
            # Try to find a match in the database
            match = self.find_matching_face(face.embeddings)
            
            if match:
                face_id, name = match
                # Draw green box for known face    
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"Name: {name}"
            else:
                # Draw yellow box for unknown face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                text = "Unknown Face"
                
            # Add text
            cv2.putText(frame, text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add confidence score
            conf_text = f"Conf: {face.confidence:.2f}"
            cv2.putText(frame, conf_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                       
            # If unknown face, prompt for name
            if not match:
                print(f"\nNew face detected on camera {camera_name}!")
                name = input("Enter name for this face (or press Enter to skip): ")
                if name:
                    self.add_new_face(face.face_id, face.embeddings, name)
                    print(f"Saved face for {name}")
                    
        return frame

def main():
    # Initialize the system
    system = FaceRecognitionSystem()
    
    # Add local webcam (using the working configuration)
    print("Connecting to webcam...")
    if system.camera_manager.add_camera(0, "webcam"):
        print("Successfully connected to webcam")
    else:
        print("Failed to connect to webcam")
        return
    
    print("\nPress 'q' to quit")
    print("Press 's' to save current frame")
    
    try:
        while True:
            # Read frames from all cameras
            frames = system.camera_manager.read_all()
            
            if not frames:
                print("No frames received from camera")
                time.sleep(1)
                continue
            
            for camera_name, frame in frames.items():
                if frame is None:
                    continue
                    
                # Process frame
                processed_frame = system.process_frame(frame, camera_name)
                
                if processed_frame is not None:
                    # Display frame
                    cv2.imshow(f'Camera: {camera_name}', processed_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frames
                for camera_name, frame in frames.items():
                    if frame is not None:
                        filename = f"frame_{camera_name}_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"Saved frame from {camera_name} to {filename}")
                        
    finally:
        # Cleanup
        system.camera_manager.release_all()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 