"""
Detector module for MemoryTrack system.
Implements YOLOv11 person detection and ByteTrack multi-object tracking.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics package not found. Install with: pip install ultralytics")

try:
    from bytetracker import BYTETracker
except ImportError:
    # Fallback: implement simple tracking if ByteTrack not available
    BYTETracker = None
    logging.warning("ByteTrack not available, will use simple tracking fallback")

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box representation."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class Track:
    """Track representation with temporal information."""
    track_id: int
    bbox: BoundingBox
    age: int
    time_since_update: int
    state: str  # 'new', 'confirmed', 'lost'
    history: List[Tuple[float, float]]  # Trajectory history
    velocity: Tuple[float, float]  # Current velocity (vx, vy)


class PersonDetector:
    """YOLOv11-based person detection."""
    
    def __init__(self, model_path: str = "yolo11n.pt", 
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.45,
                 device: str = "cuda"):
        """
        Initialize person detector.
        
        Args:
            model_path: Path to YOLOv11 model file
            confidence_threshold: Detection confidence threshold
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = self._check_device(device)
        
        self.model = self._load_model()
        self.person_class_id = 0  # COCO class ID for 'person'
        
        logger.info(f"PersonDetector initialized with {model_path} on {self.device}")
    
    def _check_device(self, device: str) -> str:
        """
        Check if specified device is available, fallback to CPU if not.
        
        Args:
            device: Requested device
            
        Returns:
            Available device
        """
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    return "cpu"
            except ImportError:
                logger.warning("PyTorch not available, falling back to CPU")
                return "cpu"
        return device
    
    def _load_model(self):
        """Load YOLOv11 model."""
        try:
            model = YOLO(self.model_path)
            # Move model to specified device
            if self.device == "cuda":
                model.to('cuda')
            logger.info(f"YOLOv11 model loaded from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLOv11 model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Detect persons in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of bounding boxes for detected persons
        """
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, 
                               iou=self.nms_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    # Filter for person class only
                    if int(box.cls[0]) != self.person_class_id:
                        continue
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    detections.append(BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name
                    ))
            
            logger.debug(f"Detected {len(detections)} persons")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def detect_with_accessories(self, frame: np.ndarray,
                               accessory_classes: List[str]) -> Dict[str, List[BoundingBox]]:
        """
        Detect persons and accessories in a frame.
        
        Args:
            frame: Input frame
            accessory_classes: List of accessory class names to detect
            
        Returns:
            Dictionary with 'persons' and 'accessories' keys
        """
        try:
            results = self.model(frame, conf=self.confidence_threshold,
                               iou=self.nms_threshold, verbose=False)
            
            output = {
                'persons': [],
                'accessories': {}
            }
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    bbox = BoundingBox(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name
                    )
                    
                    if class_name == 'person':
                        output['persons'].append(bbox)
                    elif class_name in accessory_classes:
                        if class_name not in output['accessories']:
                            output['accessories'][class_name] = []
                        output['accessories'][class_name].append(bbox)
            
            return output
            
        except Exception as e:
            logger.error(f"Detection with accessories failed: {e}")
            return {'persons': [], 'accessories': {}}


class MultiObjectTracker:
    """ByteTrack wrapper for multi-object tracking."""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Initialize multi-object tracker.
        
        Args:
            max_age: Maximum number of frames to keep track alive
            min_hits: Minimum number of detections before track is confirmed
            iou_threshold: IOU threshold for track association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        if BYTETracker is not None:
            self.tracker = BYTETracker(
                args=lambda: None  # Simple args object
            )
            # Set tracker parameters
            self.tracker.args.track_thresh = 0.5
            self.tracker.args.track_buffer = self.max_age
            self.tracker.args.match_thresh = self.iou_threshold
            self.tracker.args.min_box_area = 10
            self.tracker.args.mot20 = False
        else:
            self.tracker = None
            logger.warning("Using simple tracking fallback")
        
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}
        self.frame_count = 0
        
        logger.info("MultiObjectTracker initialized")
    
    def update(self, detections: List[BoundingBox]) -> Dict[int, Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of bounding boxes from detector
            
        Returns:
            Dictionary of track_id -> Track objects
        """
        self.frame_count += 1
        
        if self.tracker is not None:
            return self._update_bytetrack(detections)
        else:
            return self._update_simple(detections)
    
    def _update_bytetrack(self, detections: List[BoundingBox]) -> Dict[int, Track]:
        """Update tracks using ByteTrack."""
        try:
            # Convert detections to format expected by ByteTrack
            det_array = np.array([[
                d.x1, d.y1, d.x2, d.y2, d.confidence
            ] for d in detections], dtype=np.float32)
            
            # Update tracker
            online_targets = self.tracker.update(
                det_array, 
                [self.frame_count],  # image info placeholder
                [self.frame_count]
            )
            
            # Convert back to our Track format
            active_tracks = {}
            
            for t in online_targets:
                track_id = int(t.track_id)
                
                # Calculate velocity from history
                history = self.tracks.get(track_id, Track(
                    track_id=track_id,
                    bbox=detections[0] if detections else BoundingBox(0,0,0,0,0,0,""),
                    age=0,
                    time_since_update=0,
                    state='new',
                    history=[],
                    velocity=(0, 0)
                )).history
                
                # Add current position to history
                center_x = (t.tlwh[0] + t.tlwh[2]) / 2
                center_y = (t.tlwh[1] + t.tlwh[3]) / 2
                history.append((center_x, center_y))
                
                # Keep only recent history
                if len(history) > 10:
                    history = history[-10:]
                
                # Calculate velocity
                velocity = (0, 0)
                if len(history) >= 2:
                    velocity = (
                        history[-1][0] - history[-2][0],
                        history[-1][1] - history[-2][1]
                    )
                
                bbox = BoundingBox(
                    x1=float(t.tlwh[0]),
                    y1=float(t.tlwh[1]),
                    x2=float(t.tlwh[0] + t.tlwh[2]),
                    y2=float(t.tlwh[1] + t.tlwh[3]),
                    confidence=float(t.score),
                    class_id=0,
                    class_name="person"
                )
                
                state = 'confirmed' if t.score > 0.5 else 'lost'
                
                active_tracks[track_id] = Track(
                    track_id=track_id,
                    bbox=bbox,
                    age=int(t.start_frame - self.frame_count + self.max_age),
                    time_since_update=0,
                    state=state,
                    history=history,
                    velocity=velocity
                )
            
            self.tracks = active_tracks
            return active_tracks
            
        except Exception as e:
            logger.error(f"ByteTrack update failed: {e}")
            return self._update_simple(detections)
    
    def _update_simple(self, detections: List[BoundingBox]) -> Dict[int, Track]:
        """Simple tracking fallback using IOU matching."""
        active_tracks = {}
        
        # Simple IOU-based matching
        for det in detections:
            best_match_id = None
            best_iou = 0.0
            
            for track_id, track in self.tracks.items():
                iou = self._calculate_iou(det, track.bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                track = self.tracks[best_match_id]
                track.bbox = det
                track.time_since_update = 0
                track.age += 1
                
                # Update history and velocity
                center_x = (det.x1 + det.x2) / 2
                center_y = (det.y1 + det.y2) / 2
                track.history.append((center_x, center_y))
                if len(track.history) > 10:
                    track.history = track.history[-10:]
                
                if len(track.history) >= 2:
                    track.velocity = (
                        track.history[-1][0] - track.history[-2][0],
                        track.history[-1][1] - track.history[-2][1]
                    )
                
                track.state = 'confirmed'
                active_tracks[best_match_id] = track
            else:
                # Create new track
                center_x = (det.x1 + det.x2) / 2
                center_y = (det.y1 + det.y2) / 2
                
                active_tracks[self.next_track_id] = Track(
                    track_id=self.next_track_id,
                    bbox=det,
                    age=0,
                    time_since_update=0,
                    state='new',
                    history=[(center_x, center_y)],
                    velocity=(0, 0)
                )
                self.next_track_id += 1
        
        # Increment time_since_update for lost tracks
        for track_id, track in self.tracks.items():
            if track_id not in active_tracks:
                track.time_since_update += 1
                if track.time_since_update < self.max_age:
                    track.state = 'lost'
                    active_tracks[track_id] = track
        
        self.tracks = active_tracks
        return active_tracks
    
    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union (IOU) between two bounding boxes."""
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
        
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_track_history(self, track_id: int) -> List[Tuple[float, float]]:
        """
        Get trajectory history for a track.
        
        Args:
            track_id: Track ID
            
        Returns:
            List of (x, y) center positions
        """
        if track_id in self.tracks:
            return self.tracks[track_id].history
        return []
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0
        if self.tracker is not None:
            self.tracker = BYTETracker(args=lambda: None)
        logger.info("Tracker reset")
