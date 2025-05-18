import cv2
import numpy as np
from src.detection.face_detector import FaceDetector
import time
import json
import os
import pyttsx3

class FaceDatabase:
    def __init__(self, db_file='face_database.json'):
        self.db_file = db_file
        self.faces = self.load_database()
        self.engine = pyttsx3.init()
        self.min_quality_score = 0.7  # Minimum quality score for face samples
        self.similarity_threshold = 0.7  # Increased similarity threshold
        
    def load_database(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r') as f:
                return json.load(f)
        return {}
        
    def save_database(self):
        with open(self.db_file, 'w') as f:
            json.dump(self.faces, f)
            
    def add_face(self, face_id, name, embedding, quality_score=None):
        if quality_score is not None and quality_score < self.min_quality_score:
            print(f"Face quality too low ({quality_score:.2f}), skipping...")
            return False
            
        if face_id not in self.faces:
            self.faces[face_id] = {
                'name': name,
                'embeddings': [embedding.tolist()],
                'last_updated': time.time()
            }
        else:
            # Add new embedding to existing person
            self.faces[face_id]['embeddings'].append(embedding.tolist())
            self.faces[face_id]['last_updated'] = time.time()
            
        self.save_database()
        return True
        
    def find_match(self, embedding, threshold=None):
        if threshold is None:
            threshold = self.similarity_threshold
            
        best_match = None
        best_similarity = 0
        
        for face_id, data in self.faces.items():
            # Compare against all stored embeddings for this person
            similarities = []
            for stored_embedding in data['embeddings']:
                stored_embedding = np.array(stored_embedding)
                similarity = np.dot(embedding, stored_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                )
                similarities.append(similarity)
            
            # Use the average of top 3 similarities
            avg_similarity = np.mean(sorted(similarities, reverse=True)[:3])
            
            if avg_similarity > threshold and avg_similarity > best_similarity:
                best_match = (face_id, data['name'], avg_similarity)
                best_similarity = avg_similarity
                
        return best_match
        
    def get_face_quality(self, face_img):
        """Calculate face quality score based on various factors."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Calculate blur score using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, blur_score / 500.0)  # Normalize blur score
            
            # Calculate brightness score
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            
            # Calculate contrast score
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 128.0)
            
            # Combine scores with weights
            quality_score = (
                0.4 * blur_score +
                0.3 * brightness_score +
                0.3 * contrast_score
            )
            
            return quality_score
        except Exception as e:
            print(f"Error calculating face quality: {str(e)}")
            return 0.0

def draw_landmarks(frame, landmarks, color=(0, 255, 0)):
    """Draw facial landmarks on the frame."""
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 1, color, -1)

def draw_face_mesh(frame, landmarks, color=(0, 255, 0)):
    """Draw face mesh connections."""
    # Define connections between landmarks for face mesh
    connections = [
        # Face outline
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
        # Left eyebrow
        (17, 18), (18, 19), (19, 20), (20, 21),
        # Right eyebrow
        (22, 23), (23, 24), (24, 25), (25, 26),
        # Nose bridge
        (27, 28), (28, 29), (29, 30),
        # Lower nose
        (31, 32), (32, 33), (33, 34), (34, 35),
        # Left eye
        (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
        # Right eye
        (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
        # Outer lip
        (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
        (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
        # Inner lip
        (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66),
        (66, 67), (67, 60)
    ]
    
    # Draw connections
    for connection in connections:
        start_point = landmarks[connection[0]]
        end_point = landmarks[connection[1]]
        cv2.line(frame, tuple(start_point), tuple(end_point), color, 1)

def main():
    # Initialize face detector and database
    detector = FaceDetector(confidence_threshold=0.5)
    db = FaceDatabase()
    
    # Available cameras
    cameras = {
        '1': {'index': 0, 'name': 'Registration Camera'},
        '2': {'index': 1, 'name': 'Recognition Camera 1'},
        '3': {'index': 2, 'name': 'Recognition Camera 2'}
    }
    
    current_camera = None
    cap = None
    
    print("\nCamera Controls:")
    print("Press '1' for Registration Camera")
    print("Press '2' for Recognition Camera 1")
    print("Press '3' for Recognition Camera 2")
    print("Press 'q' to quit")
    print("Press 's' to save face (only in Registration Camera)")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Handle camera switching
        if key in [ord('1'), ord('2'), ord('3')]:
            camera_key = chr(key)
            if current_camera != camera_key:
                if cap is not None:
                    cap.release()
                print(f"\nSwitching to {cameras[camera_key]['name']}...")
                cap = cv2.VideoCapture(cameras[camera_key]['index'], cv2.CAP_DSHOW)
                if not cap.isOpened():
                    print(f"Failed to open camera {camera_key}")
                    continue
                current_camera = camera_key
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
        
        elif key == ord('q'):
            break
            
        if cap is None or not cap.isOpened():
            continue
            
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            continue
            
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Process faces based on current camera
        for face in faces:
            x, y, w, h = face.bbox
            color = (0, 255, 0)  # Green for recognized faces
            
            if current_camera == '1':  # Registration Camera
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "Press 's' to register"
                cv2.putText(frame, text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Handle face registration
                if key == ord('s'):
                    name = input("Enter person's name: ")
                    face_id = str(int(time.time() * 1000))
                    db.add_face(face_id, name, face.embedding)
                    print(f"Registered {name}")
                    
            else:  # Recognition Cameras
                # Find matching face
                match = db.find_match(face.embedding)
                if match:
                    face_id, name, similarity = match
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    # Display name and similarity
                    text = f"{name} ({similarity:.2f})"
                    cv2.putText(frame, text, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # Announce person
                    db.engine.say(f"{name} detected")
                    db.engine.runAndWait()
                else:
                    # Unknown face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow(f'Camera {current_camera}', frame)
    
    # Cleanup
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 