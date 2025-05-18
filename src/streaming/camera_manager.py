import cv2
from typing import Dict, Optional
import logging
from threading import Thread
import queue
import numpy as np

class CameraManager:
    def __init__(self):
        self.cameras: Dict[str, dict] = {}
        self.frame_queues: Dict[str, queue.Queue] = {}
        self.logger = logging.getLogger(__name__)

    def add_camera(self, camera_id: str, stream_url: str) -> bool:
        """
        Add a new camera stream to the manager.
        
        Args:
            camera_id: Unique identifier for the camera
            stream_url: RTSP/HTTP URL or camera index
        """
        try:
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                self.logger.error(f"Failed to open camera stream: {stream_url}")
                return False
            
            self.cameras[camera_id] = {
                'stream_url': stream_url,
                'capture': cap,
                'active': True
            }
            self.frame_queues[camera_id] = queue.Queue(maxsize=30)  # Buffer 30 frames
            
            # Start frame capture thread
            thread = Thread(target=self._capture_frames, args=(camera_id,))
            thread.daemon = True
            thread.start()
            
            return True
        except Exception as e:
            self.logger.error(f"Error adding camera {camera_id}: {str(e)}")
            return False

    def _capture_frames(self, camera_id: str):
        """Background thread for continuous frame capture."""
        while self.cameras[camera_id]['active']:
            cap = self.cameras[camera_id]['capture']
            ret, frame = cap.read()
            
            if ret:
                if self.frame_queues[camera_id].full():
                    # Remove oldest frame if queue is full
                    try:
                        self.frame_queues[camera_id].get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queues[camera_id].put(frame)
                cv2.imshow('Camera Test', frame)
            else:
                self.logger.warning(f"Failed to read frame from camera {camera_id}")
                break

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get the latest frame from a specific camera."""
        try:
            return self.frame_queues[camera_id].get_nowait()
        except queue.Empty:
            return None

    def release_camera(self, camera_id: str):
        """Release resources for a specific camera."""
        if camera_id in self.cameras:
            self.cameras[camera_id]['active'] = False
            self.cameras[camera_id]['capture'].release()
            del self.cameras[camera_id]
            del self.frame_queues[camera_id]

    def release_all(self):
        """Release all camera resources."""
        for camera_id in list(self.cameras.keys()):
            self.release_camera(camera_id)