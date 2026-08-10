"""
Video loader module for MemoryTrack system.
Implements multi-camera video stream management.
"""

import cv2
import numpy as np
from typing import Iterator, Optional, List, Tuple, Dict
from pathlib import Path
from threading import Thread, Lock
import queue
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CameraFrame:
    """Frame from a camera with metadata."""
    frame: np.ndarray
    camera_id: str
    timestamp: float
    frame_number: int


class VideoLoader:
    """Multi-camera video stream loader."""
    
    def __init__(self, sources: List[str], camera_ids: Optional[List[str]] = None,
                 buffer_size: int = 10, resize: Optional[Tuple[int, int]] = None):
        """
        Initialize video loader.
        
        Args:
            sources: List of video sources (file paths, RTSP URLs, or webcam indices)
            camera_ids: Optional list of camera IDs (auto-generated if not provided)
            buffer_size: Frame buffer size for each camera
            resize: Optional resize dimensions (width, height)
        """
        self.sources = sources
        self.camera_ids = camera_ids or [f"cam_{i}" for i in range(len(sources))]
        self.buffer_size = buffer_size
        self.resize = resize
        
        if len(self.camera_ids) != len(self.sources):
            raise ValueError("Number of camera IDs must match number of sources")
        
        # Video captures
        self.captures: Dict[str, cv2.VideoCapture] = {}
        
        # Frame buffers
        self.frame_buffers: Dict[str, queue.Queue] = {}
        
        # Thread control
        self.threads: Dict[str, Thread] = {}
        self.running = False
        self.lock = Lock()
        
        # Frame counters
        self.frame_counters: Dict[str, int] = {}
        
        logger.info(f"VideoLoader initialized with {len(sources)} sources")
    
    def _start_camera_thread(self, camera_id: str, source: str) -> None:
        """
        Start thread for a single camera.
        
        Args:
            camera_id: Camera ID
            source: Video source
        """
        def capture_loop():
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}: {source}")
                return
            
            with self.lock:
                self.captures[camera_id] = cap
                self.frame_counters[camera_id] = 0
            
            logger.info(f"Started capture thread for {camera_id}")
            
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame from {camera_id}")
                    # Try to reopen
                    cap.release()
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        logger.error(f"Failed to reopen {camera_id}")
                        break
                    continue
                
                # Resize if specified
                if self.resize:
                    frame = cv2.resize(frame, self.resize)
                
                # Create camera frame
                camera_frame = CameraFrame(
                    frame=frame,
                    camera_id=camera_id,
                    timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
                    frame_number=self.frame_counters[camera_id]
                )
                
                # Add to buffer
                try:
                    self.frame_buffers[camera_id].put(camera_frame, timeout=0.1)
                except queue.Full:
                    # Remove oldest frame
                    try:
                        self.frame_buffers[camera_id].get_nowait()
                        self.frame_buffers[camera_id].put(camera_frame, timeout=0.1)
                    except queue.Empty:
                        pass
                
                with self.lock:
                    self.frame_counters[camera_id] += 1
            
            cap.release()
            logger.info(f"Capture thread stopped for {camera_id}")
        
        # Start thread
        thread = Thread(target=capture_loop, daemon=True)
        thread.start()
        self.threads[camera_id] = thread
    
    def start(self) -> None:
        """Start all camera capture threads."""
        self.running = True
        
        for camera_id, source in zip(self.camera_ids, self.sources):
            # Initialize buffer
            self.frame_buffers[camera_id] = queue.Queue(maxsize=self.buffer_size)
            
            # Start capture thread
            self._start_camera_thread(camera_id, source)
        
        logger.info("All camera capture threads started")
    
    def stop(self) -> None:
        """Stop all camera capture threads."""
        self.running = False
        
        # Wait for threads to finish
        for camera_id, thread in self.threads.items():
            thread.join(timeout=2.0)
        
        # Release captures
        for camera_id, cap in self.captures.items():
            cap.release()
        
        self.captures.clear()
        self.threads.clear()
        
        logger.info("All camera capture threads stopped")
    
    def get_frame(self, camera_id: str) -> Optional[CameraFrame]:
        """
        Get latest frame from a specific camera.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            CameraFrame or None if no frame available
        """
        if camera_id not in self.frame_buffers:
            logger.warning(f"Camera {camera_id} not found")
            return None
        
        try:
            return self.frame_buffers[camera_id].get_nowait()
        except queue.Empty:
            return None
    
    def get_frames(self) -> Dict[str, Optional[CameraFrame]]:
        """
        Get latest frames from all cameras.
        
        Returns:
            Dictionary of camera_id -> CameraFrame
        """
        frames = {}
        for camera_id in self.camera_ids:
            frames[camera_id] = self.get_frame(camera_id)
        return frames
    
    def __iter__(self) -> Iterator[Dict[str, Optional[CameraFrame]]]:
        """
        Iterate over synchronized frames from all cameras.
        
        Yields:
            Dictionary of camera_id -> CameraFrame
        """
        self.start()
        
        try:
            while self.running:
                yield self.get_frames()
        finally:
            self.stop()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def get_camera_ids(self) -> List[str]:
        """Get list of camera IDs."""
        return self.camera_ids.copy()
    
    def get_frame_count(self, camera_id: str) -> int:
        """
        Get total frame count for a camera.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Frame count
        """
        with self.lock:
            return self.frame_counters.get(camera_id, 0)
    
    def is_running(self) -> bool:
        """Check if video loader is running."""
        return self.running


class SingleVideoLoader:
    """Simple loader for a single video file."""
    
    def __init__(self, source: str, camera_id: str = "cam_0",
                 resize: Optional[Tuple[int, int]] = None):
        """
        Initialize single video loader.
        
        Args:
            source: Video source (file path or webcam index)
            camera_id: Camera ID
            resize: Optional resize dimensions
        """
        self.source = source
        self.camera_id = camera_id
        self.resize = resize
        self.cap = None
        self.frame_number = 0
        
        logger.info(f"SingleVideoLoader initialized for {source}")
    
    def open(self) -> bool:
        """
        Open video source.
        
        Returns:
            True if successful
        """
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False
        
        self.frame_number = 0
        logger.info(f"Opened video source: {self.source}")
        return True
    
    def read(self) -> Optional[CameraFrame]:
        """
        Read next frame.
        
        Returns:
            CameraFrame or None if end of video
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        # Resize if specified
        if self.resize:
            frame = cv2.resize(frame, self.resize)
        
        camera_frame = CameraFrame(
            frame=frame,
            camera_id=self.camera_id,
            timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
            frame_number=self.frame_number
        )
        
        self.frame_number += 1
        
        return camera_frame
    
    def release(self) -> None:
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Released video capture")
    
    def __iter__(self) -> Iterator[CameraFrame]:
        """Iterate over frames."""
        if not self.open():
            return
        
        try:
            while True:
                frame = self.read()
                if frame is None:
                    break
                yield frame
        finally:
            self.release()
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def get_fps(self) -> float:
        """Get video FPS."""
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_frame_count(self) -> int:
        """Get total frame count."""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get video resolution."""
        if self.cap is None:
            return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)


class ImageSequenceLoader:
    """Loader for image sequences."""
    
    def __init__(self, image_dir: str, pattern: str = "*.jpg",
                 camera_id: str = "cam_0"):
        """
        Initialize image sequence loader.
        
        Args:
            image_dir: Directory containing images
            pattern: File pattern to match
            camera_id: Camera ID
        """
        self.image_dir = Path(image_dir)
        self.pattern = pattern
        self.camera_id = camera_id
        self.images = sorted(self.image_dir.glob(pattern))
        self.frame_number = 0
        
        logger.info(f"ImageSequenceLoader initialized with {len(self.images)} images")
    
    def __iter__(self) -> Iterator[CameraFrame]:
        """Iterate over images."""
        for image_path in self.images:
            frame = cv2.imread(str(image_path))
            
            if frame is None:
                logger.warning(f"Failed to read image: {image_path}")
                continue
            
            camera_frame = CameraFrame(
                frame=frame,
                camera_id=self.camera_id,
                timestamp=0.0,
                frame_number=self.frame_number
            )
            
            self.frame_number += 1
            yield camera_frame
    
    def __len__(self) -> int:
        """Get number of images."""
        return len(self.images)
