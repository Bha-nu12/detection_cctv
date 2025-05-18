import cv2
import time
import numpy as np
from datetime import datetime
import os
from typing import Dict, List
import json

class AdvancedCCTVSystem:
    def __init__(self, config_path: str = 'config.json'):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Setup output directories
        self.base_dir = self.config.get('output_dir', 'output')
        self.setup_directories()
        
        # Statistics
        self.stats = {
            'start_time': time.time(),
            'frames_processed': 0,
            'faces_detected': 0
        }
    
    def setup_directories(self):
        """Create necessary directories."""
        dirs = ['recordings', 'snapshots', 'logs']
        for dir_name in dirs:
            path = os.path.join(self.base_dir, dir_name)
            os.makedirs(path, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load system configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def detect_faces(self, frame):
        """Detect faces in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def process_frame(self, frame):
        """Process a single frame."""
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Add label
            cv2.putText(frame, 'Face', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw detection area
            cv2.circle(frame, (x + w//2, y + h//2), 2, (0, 255, 0), 2)
        
        # Update statistics
        self.stats['frames_processed'] += 1
        self.stats['faces_detected'] += len(faces)
        
        return frame, len(faces)
    
    def add_stats_overlay(self, frame):
        """Add statistics overlay to frame."""
        runtime = time.time() - self.stats['start_time']
        fps = self.stats['frames_processed'] / runtime if runtime > 0 else 0
        
        stats_text = [
            f"Runtime: {runtime:.1f}s",
            f"FPS: {fps:.1f}",
            f"Faces Detected: {self.stats['faces_detected']}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        # Add semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Add text
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (10, 25 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        """Run the CCTV monitoring system."""
        print("Starting Advanced CCTV System...")
        print("Press 'q' to quit, 's' to save snapshot")
        
        if not self.camera.isOpened():
            print("Error: Could not open camera")
            return
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None
        
        try:
            while True:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame
                processed_frame, num_faces = self.process_frame(frame)
                
                # Add statistics overlay
                self.add_stats_overlay(processed_frame)
                
                # Initialize video writer if not done yet
                if out is None:
                    height, width = processed_frame.shape[:2]
                    output_path = os.path.join(
                        self.base_dir,
                        'recordings',
                        f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi'
                    )
                    out = cv2.VideoWriter(
                        output_path,
                        fourcc,
                        20.0,
                        (width, height)
                    )
                
                # Write frame to video
                out.write(processed_frame)
                
                # Display the frame
                cv2.imshow('CCTV Monitor', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    # Save snapshot
                    snapshot_path = os.path.join(
                        self.base_dir,
                        'snapshots',
                        f'snapshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                    )
                    cv2.imwrite(snapshot_path, processed_frame)
                    print(f"Saved snapshot: {snapshot_path}")
        
        finally:
            # Cleanup
            print("\nCleaning up...")
            self.camera.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            self._save_statistics()
    
    def _save_statistics(self):
        """Save system statistics to log file."""
        stats = {
            'end_time': time.time(),
            'runtime': time.time() - self.stats['start_time'],
            'frames_processed': self.stats['frames_processed'],
            'faces_detected': self.stats['faces_detected'],
            'average_fps': self.stats['frames_processed'] / (time.time() - self.stats['start_time'])
        }
        
        log_path = os.path.join(
            self.base_dir,
            'logs',
            f'stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(log_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Statistics saved to: {log_path}") 