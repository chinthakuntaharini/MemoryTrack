"""
Visualization module for MemoryTrack system.
Implements bounding box rendering, feature overlays, and match information display.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from core.detector import BoundingBox, Track
from core.pose_extractor import PoseFeatures
from core.memory_bank import MatchResult

logger = logging.getLogger(__name__)


class Visualizer:
    """Video frame visualization with overlays."""
    
    # Color palette for different tracks
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (255, 255, 255), # White
        (128, 128, 128), # Gray
    ]
    
    def __init__(self, show_features: bool = True,
                 show_confidence: bool = True,
                 show_trajectory: bool = True):
        """
        Initialize visualizer.
        
        Args:
            show_features: Show feature overlays
            show_confidence: Show confidence scores
            show_trajectory: Show trajectory paths
        """
        self.show_features = show_features
        self.show_confidence = show_confidence
        self.show_trajectory = show_trajectory
        
        logger.info("Visualizer initialized")
    
    def draw_bbox(self, frame: np.ndarray, bbox: BoundingBox,
                  track_id: Optional[int] = None,
                  color: Optional[Tuple[int, int, int]] = None,
                  label: Optional[str] = None) -> np.ndarray:
        """
        Draw bounding box on frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box to draw
            track_id: Optional track ID
            color: Box color (auto-generated if not provided)
            label: Optional label text
            
        Returns:
            Frame with bounding box drawn
        """
        frame_copy = frame.copy()
        
        # Get color
        if color is None and track_id is not None:
            color = self._get_track_color(track_id)
        elif color is None:
            color = (0, 255, 0)
        
        # Draw box
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if label or track_id is not None:
            text = f"ID: {track_id}" if track_id else ""
            if label:
                text = f"{text} {label}" if text else label
            
            if text:
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw background
                cv2.rectangle(
                    frame_copy,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame_copy,
                    text,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
        
        # Show confidence if enabled
        if self.show_confidence and bbox.confidence > 0:
            conf_text = f"{bbox.confidence:.2f}"
            cv2.putText(
                frame_copy,
                conf_text,
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
        
        return frame_copy
    
    def draw_pose(self, frame: np.ndarray, pose_features: PoseFeatures,
                  bbox: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """
        Draw pose skeleton on frame.
        
        Args:
            frame: Input frame
            pose_features: Pose features to draw
            bbox: Original bounding box if frame was cropped
            
        Returns:
            Frame with pose drawn
        """
        from core.pose_extractor import PoseExtractor
        
        pose_extractor = PoseExtractor()
        return pose_extractor.draw_pose(frame, pose_features, bbox)
    
    def draw_trajectory(self, frame: np.ndarray, track: Track,
                       max_length: int = 30) -> np.ndarray:
        """
        Draw trajectory path for a track.
        
        Args:
            frame: Input frame
            track: Track object
            max_length: Maximum trajectory length to draw
            
        Returns:
            Frame with trajectory drawn
        """
        if not self.show_trajectory or len(track.history) < 2:
            return frame
        
        frame_copy = frame.copy()
        color = self._get_track_color(track.track_id)
        
        # Get recent history
        history = track.history[-max_length:]
        
        # Draw trajectory as connected lines
        for i in range(len(history) - 1):
            pt1 = (int(history[i][0]), int(history[i][1]))
            pt2 = (int(history[i + 1][0]), int(history[i + 1][1]))
            
            # Fade color based on age
            alpha = (i + 1) / len(history)
            line_color = tuple(int(c * alpha) for c in color)
            
            cv2.line(frame_copy, pt1, pt2, line_color, 2)
        
        # Draw velocity arrow
        if len(history) >= 2:
            last_pt = (int(history[-1][0]), int(history[-1][1]))
            velocity = track.velocity
            
            # Scale velocity for visibility
            scale = 10
            end_pt = (
                int(last_pt[0] + velocity[0] * scale),
                int(last_pt[1] + velocity[1] * scale)
            )
            
            cv2.arrowedLine(frame_copy, last_pt, end_pt, color, 2)
        
        return frame_copy
    
    def draw_match_info(self, frame: np.ndarray, match: MatchResult,
                       position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Draw match information on frame.
        
        Args:
            frame: Input frame
            match: Match result to display
            position: Text position (x, y)
            
        Returns:
            Frame with match info drawn
        """
        frame_copy = frame.copy()
        
        # Background for text
        text_lines = [
            f"Match ID: {match.person_id}",
            f"Confidence: {match.confidence:.2%}",
            f"Status: {match.profile.status}",
        ]
        
        # Add explanation if available
        if match.explanation:
            text_lines.append(f"Info: {match.explanation[:50]}...")
        
        # Draw background
        line_height = 25
        bg_height = len(text_lines) * line_height + 10
        bg_width = 400
        
        x, y = position
        cv2.rectangle(
            frame_copy,
            (x, y),
            (x + bg_width, y + bg_height),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        for i, line in enumerate(text_lines):
            text_y = y + (i + 1) * line_height
            cv2.putText(
                frame_copy,
                line,
                (x + 10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        return frame_copy
    
    def draw_features(self, frame: np.ndarray,
                     features: Dict[str, np.ndarray],
                     position: Tuple[int, int] = (10, 150)) -> np.ndarray:
        """
        Draw feature visualization on frame.
        
        Args:
            frame: Input frame
            features: Dictionary of modality -> feature vector
            position: Text position
            
        Returns:
            Frame with features displayed
        """
        if not self.show_features:
            return frame
        
        frame_copy = frame.copy()
        
        # Draw feature summary
        text_lines = [f"Features detected:"]
        
        for modality, feature in features.items():
            if feature is not None and len(feature) > 0:
                text_lines.append(f"  {modality}: {len(feature)}-dim")
        
        # Draw background
        line_height = 20
        bg_height = len(text_lines) * line_height + 10
        bg_width = 250
        
        x, y = position
        cv2.rectangle(
            frame_copy,
            (x, y),
            (x + bg_width, y + bg_height),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        for i, line in enumerate(text_lines):
            text_y = y + (i + 1) * line_height
            cv2.putText(
                frame_copy,
                line,
                (x + 10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
        
        return frame_copy
    
    def draw_camera_info(self, frame: np.ndarray, camera_id: str,
                        frame_number: int, fps: float = 0.0,
                        timestamp: float = 0.0) -> np.ndarray:
        """
        Draw camera information on frame.
        
        Args:
            frame: Input frame
            camera_id: Camera ID
            frame_number: Frame number
            fps: Current FPS
            timestamp: Frame timestamp
            
        Returns:
            Frame with camera info drawn
        """
        frame_copy = frame.copy()
        
        # Position (top-right corner)
        h, w = frame_copy.shape[:2]
        x = w - 200
        y = 30
        
        # Text lines
        text_lines = [
            f"Camera: {camera_id}",
            f"Frame: {frame_number}",
        ]
        
        if fps > 0:
            text_lines.append(f"FPS: {fps:.1f}")
        
        # Draw background
        line_height = 20
        bg_height = len(text_lines) * line_height + 10
        bg_width = 180
        
        cv2.rectangle(
            frame_copy,
            (x, y),
            (x + bg_width, y + bg_height),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        for i, line in enumerate(text_lines):
            text_y = y + (i + 1) * line_height
            cv2.putText(
                frame_copy,
                line,
                (x + 10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
        
        return frame_copy
    
    def draw_accessories(self, frame: np.ndarray,
                        accessories: Dict[str, List[BoundingBox]]) -> np.ndarray:
        """
        Draw detected accessories on frame.
        
        Args:
            frame: Input frame
            accessories: Dictionary of accessory_type -> list of bounding boxes
            
        Returns:
            Frame with accessories drawn
        """
        frame_copy = frame.copy()
        
        # Colors for different accessories
        accessory_colors = {
            'backpack': (255, 128, 0),
            'handbag': (255, 0, 128),
            'cap': (0, 128, 255),
            'umbrella': (128, 255, 0),
            'suitcase': (255, 255, 0),
            'bottle': (0, 255, 255)
        }
        
        for acc_type, bboxes in accessories.items():
            color = accessory_colors.get(acc_type, (255, 255, 255))
            
            for bbox in bboxes:
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                
                # Draw box
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                cv2.putText(
                    frame_copy,
                    acc_type,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )
        
        return frame_copy
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Get color for a track ID.
        
        Args:
            track_id: Track ID
            
        Returns:
            RGB color tuple
        """
        return self.COLORS[track_id % len(self.COLORS)]
    
    def create_comparison_view(self, query_frame: np.ndarray,
                              match_frame: np.ndarray,
                              match_result: MatchResult) -> np.ndarray:
        """
        Create side-by-side comparison view.
        
        Args:
            query_frame: Query image
            match_frame: Matched image
            match_result: Match result information
            
        Returns:
            Combined comparison image
        """
        # Resize frames to same height
        h1, w1 = query_frame.shape[:2]
        h2, w2 = match_frame.shape[:2]
        
        target_height = max(h1, h2)
        
        query_resized = cv2.resize(query_frame, (int(w1 * target_height / h1), target_height))
        match_resized = cv2.resize(match_frame, (int(w2 * target_height / h2), target_height))
        
        # Combine horizontally
        combined = np.hstack([query_resized, match_resized])
        
        # Add labels
        cv2.putText(
            combined,
            "Query",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            combined,
            f"Match (ID: {match_result.person_id})",
            (query_resized.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if match_result.confidence > 0.7 else (0, 165, 255),
            2
        )
        
        # Add confidence
        conf_text = f"Confidence: {match_result.confidence:.2%}"
        cv2.putText(
            combined,
            conf_text,
            (query_resized.shape[1] + 10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return combined
