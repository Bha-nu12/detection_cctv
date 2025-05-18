import cv2
import threading
import queue
from typing import Dict, List
import time
import numpy as np

class CameraStream:
    def __init__(self, source, name: str, buffer_size: int = 30):
        self.source = source
        self.name = name
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        self.cap = None
        
    def start(self):
        # Try to open camera without specific backend first
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            # If that fails, try with DirectShow backend
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.name}")
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
            
        # Start frame capture thread
        thread = threading.Thread(target=self._capture_frames)
        thread.daemon = True
        thread.start()
        return self
        
    def _capture_frames(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
                
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
                
            # Clear queue if full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            self.frame_queue.put(frame)
            
    def read(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
            
    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

class MultiCameraManager:
    def __init__(self):
        self.cameras: Dict[str, CameraStream] = {}
        self.recorders: Dict[str, cv2.VideoWriter] = {}
        
    def add_camera(self, source, name: str) -> bool:
        """Add a new camera stream."""
        try:
            camera = CameraStream(source, name)
            camera.start()
            self.cameras[name] = camera
            return True
        except Exception as e:
            print(f"Error adding camera {name}: {str(e)}")
            return False
            
    def start_recording(self, camera_name: str, output_path: str):
        """Start recording from a specific camera."""
        if camera_name not in self.cameras:
            return False
            
        # Get a test frame to determine size
        test_frame = self.cameras[camera_name].read()
        if test_frame is None:
            return False
            
        h, w = test_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.recorders[camera_name] = cv2.VideoWriter(
            output_path, fourcc, 30.0, (w, h)
        )
        return True
        
    def stop_recording(self, camera_name: str):
        """Stop recording from a specific camera."""
        if camera_name in self.recorders:
            self.recorders[camera_name].release()
            del self.recorders[camera_name]
            
    def read_all(self) -> Dict[str, np.ndarray]:
        """Read frames from all cameras."""
        frames = {}
        for name, camera in self.cameras.items():
            frame = camera.read()
            if frame is not None:
                frames[name] = frame
                
                # Record if active
                if name in self.recorders:
                    self.recorders[name].write(frame)
        return frames
        
    def release_all(self):
        """Release all cameras and recorders."""
        for camera in self.cameras.values():
            camera.stop()
        for recorder in self.recorders.values():
            recorder.release()
        self.cameras.clear()
        self.recorders.clear() 