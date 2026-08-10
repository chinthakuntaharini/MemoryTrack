"""
Pose extractor module for MemoryTrack system.
Implements MediaPipe pose estimation and body ratio calculation.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("mediapipe package not found. Install with: pip install mediapipe")

logger = logging.getLogger(__name__)


@dataclass
class PoseFeatures:
    """Pose features extracted from a person."""
    keypoints: np.ndarray  # 33 keypoints in MediaPipe format
    confidence: float
    body_ratios: Dict[str, float]
    keypoint_confidences: np.ndarray


class PoseExtractor:
    """MediaPipe-based pose estimation."""
    
    # MediaPipe Pose landmark indices
    LANDMARKS = {
        'nose': 0,
        'left_eye_inner': 1,
        'left_eye': 2,
        'left_eye_outer': 3,
        'right_eye_inner': 4,
        'right_eye': 5,
        'right_eye_outer': 6,
        'left_ear': 7,
        'right_ear': 8,
        'mouth_left': 9,
        'mouth_right': 10,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_pinky': 17,
        'right_pinky': 18,
        'left_index': 19,
        'right_index': 20,
        'left_thumb': 21,
        'right_thumb': 22,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
        'left_heel': 29,
        'right_heel': 30,
        'left_foot_index': 31,
        'right_foot_index': 32
    }
    
    def __init__(self, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize pose extractor.
        
        Args:
            model_complexity: Model complexity (0, 1, or 2)
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        logger.info(f"PoseExtractor initialized with complexity {model_complexity}")
    
    def extract(self, frame: np.ndarray, bbox: Optional[Tuple[float, float, float, float]] = None) -> Optional[PoseFeatures]:
        """
        Extract pose features from frame.
        
        Args:
            frame: Input frame (BGR format)
            bbox: Optional bounding box (x1, y1, x2, y2) to crop to person
            
        Returns:
            PoseFeatures object or None if detection failed
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Crop to bounding box if provided
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                frame_rgb = frame_rgb[int(y1):int(y2), int(x1):int(x2)]
            
            # Run pose estimation
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks is None:
                logger.debug("No pose landmarks detected")
                return None
            
            # Extract keypoints
            keypoints = self._extract_keypoints(results.pose_landmarks)
            keypoint_confidences = self._extract_confidences(results.pose_landmarks)
            
            # Calculate overall confidence
            overall_confidence = np.mean(keypoint_confidences)
            
            # Calculate body ratios
            body_ratios = self._calculate_body_ratios(keypoints, keypoint_confidences)
            
            return PoseFeatures(
                keypoints=keypoints,
                confidence=float(overall_confidence),
                body_ratios=body_ratios,
                keypoint_confidences=keypoint_confidences
            )
            
        except Exception as e:
            logger.error(f"Pose extraction failed: {e}")
            return None
    
    def _extract_keypoints(self, landmarks) -> np.ndarray:
        """
        Extract keypoints from MediaPipe landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            numpy array of shape (33, 3) with (x, y, z) coordinates
        """
        keypoints = np.zeros((33, 3), dtype=np.float32)
        
        for i in range(33):
            keypoints[i, 0] = landmarks.landmark[i].x
            keypoints[i, 1] = landmarks.landmark[i].y
            keypoints[i, 2] = landmarks.landmark[i].z
        
        return keypoints
    
    def _extract_confidences(self, landmarks) -> np.ndarray:
        """
        Extract visibility/confidence scores from landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            numpy array of shape (33,) with confidence scores
        """
        confidences = np.zeros(33, dtype=np.float32)
        
        for i in range(33):
            confidences[i] = landmarks.landmark[i].visibility
        
        return confidences
    
    def _calculate_body_ratios(self, keypoints: np.ndarray,
                              confidences: np.ndarray) -> Dict[str, float]:
        """
        Calculate body proportions from keypoints.
        
        Args:
            keypoints: (33, 3) array of keypoints
            confidences: (33,) array of confidence scores
            
        Returns:
            Dictionary of body ratio measurements
        """
        ratios = {}
        
        # Helper function to get keypoint with confidence check
        def get_point(idx):
            if confidences[idx] < 0.5:
                return None
            return keypoints[idx]
        
        # Shoulder width
        left_shoulder = get_point(self.LANDMARKS['left_shoulder'])
        right_shoulder = get_point(self.LANDMARKS['right_shoulder'])
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
            ratios['shoulder_width'] = float(shoulder_width)
        
        # Hip width
        left_hip = get_point(self.LANDMARKS['left_hip'])
        right_hip = get_point(self.LANDMARKS['right_hip'])
        if left_hip is not None and right_hip is not None:
            hip_width = np.linalg.norm(left_hip[:2] - right_hip[:2])
            ratios['hip_width'] = float(hip_width)
        
        # Shoulder-to-hip ratio
        if 'shoulder_width' in ratios and 'hip_width' in ratios:
            ratios['shoulder_to_hip'] = ratios['shoulder_width'] / (ratios['hip_width'] + 1e-6)
        
        # Torso length (shoulder to hip)
        if left_shoulder is not None and left_hip is not None:
            torso_length = np.linalg.norm(left_shoulder[:2] - left_hip[:2])
            ratios['torso_length'] = float(torso_length)
        
        # Leg length (hip to ankle)
        left_ankle = get_point(self.LANDMARKS['left_ankle'])
        if left_hip is not None and left_ankle is not None:
            leg_length = np.linalg.norm(left_hip[:2] - left_ankle[:2])
            ratios['leg_length'] = float(leg_length)
        
        # Leg-to-torso ratio
        if 'leg_length' in ratios and 'torso_length' in ratios:
            ratios['leg_to_torso'] = ratios['leg_length'] / (ratios['torso_length'] + 1e-6)
        
        # Arm length (shoulder to wrist)
        left_wrist = get_point(self.LANDMARKS['left_wrist'])
        if left_shoulder is not None and left_wrist is not None:
            arm_length = np.linalg.norm(left_shoulder[:2] - left_wrist[:2])
            ratios['arm_length'] = float(arm_length)
        
        # Arm-to-torso ratio
        if 'arm_length' in ratios and 'torso_length' in ratios:
            ratios['arm_to_torso'] = ratios['arm_length'] / (ratios['torso_length'] + 1e-6)
        
        # Body height (nose to ankle)
        nose = get_point(self.LANDMARKS['nose'])
        if nose is not None and left_ankle is not None:
            body_height = np.linalg.norm(nose[:2] - left_ankle[:2])
            ratios['body_height'] = float(body_height)
        
        # Head-to-body ratio (nose to shoulder / body height)
        if 'body_height' in ratios and left_shoulder is not None:
            head_height = np.linalg.norm(nose[:2] - left_shoulder[:2])
            ratios['head_to_body'] = head_height / (ratios['body_height'] + 1e-6)
        
        return ratios
    
    def extract_feature_vector(self, pose_features: PoseFeatures) -> np.ndarray:
        """
        Convert pose features to a feature vector for fusion.
        
        Args:
            pose_features: PoseFeatures object
            
        Returns:
            Feature vector (64-dimensional)
        """
        # Start with flattened keypoints (33 * 2 = 64 dimensions, using x,y only)
        feature_vector = pose_features.keypoints[:, :2].flatten()
        
        # If we have fewer than 64 dimensions, pad with body ratios
        if len(feature_vector) < 64:
            ratio_values = list(pose_features.body_ratios.values())
            feature_vector = np.concatenate([
                feature_vector,
                np.array(ratio_values[:64 - len(feature_vector)])
            ])
        
        # If more than 64 dimensions, truncate
        if len(feature_vector) > 64:
            feature_vector = feature_vector[:64]
        
        return feature_vector
    
    def draw_pose(self, frame: np.ndarray, pose_features: PoseFeatures,
                  bbox: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """
        Draw pose skeleton on frame.
        
        Args:
            frame: Input frame
            pose_features: PoseFeatures to draw
            bbox: Original bounding box if frame was cropped
            
        Returns:
            Frame with pose drawn
        """
        frame_copy = frame.copy()
        
        # Create MediaPipe landmarks object for drawing
        mp_landmarks = mp.solutions.pose.PoseLandmark
        
        # Draw connections
        connections = [
            (mp_landmarks.LEFT_SHOULDER, mp_landmarks.RIGHT_SHOULDER),
            (mp_landmarks.LEFT_SHOULDER, mp_landmarks.LEFT_ELBOW),
            (mp_landmarks.LEFT_ELBOW, mp_landmarks.LEFT_WRIST),
            (mp_landmarks.RIGHT_SHOULDER, mp_landmarks.RIGHT_ELBOW),
            (mp_landmarks.RIGHT_ELBOW, mp_landmarks.RIGHT_WRIST),
            (mp_landmarks.LEFT_SHOULDER, mp_landmarks.LEFT_HIP),
            (mp_landmarks.RIGHT_SHOULDER, mp_landmarks.RIGHT_HIP),
            (mp_landmarks.LEFT_HIP, mp_landmarks.RIGHT_HIP),
            (mp_landmarks.LEFT_HIP, mp_landmarks.LEFT_KNEE),
            (mp_landmarks.LEFT_KNEE, mp_landmarks.LEFT_ANKLE),
            (mp_landmarks.RIGHT_HIP, mp_landmarks.RIGHT_KNEE),
            (mp_landmarks.RIGHT_KNEE, mp_landmarks.RIGHT_ANKLE),
        ]
        
        keypoints = pose_features.keypoints
        h, w = frame_copy.shape[:2]
        
        # Offset if frame was cropped
        offset_x, offset_y = 0, 0
        if bbox is not None:
            offset_x, offset_y = int(bbox[0]), int(bbox[1])
        
        # Draw keypoints
        for i in range(33):
            if pose_features.keypoint_confidences[i] > 0.5:
                x = int(keypoints[i, 0] * w) + offset_x
                y = int(keypoints[i, 1] * h) + offset_y
                cv2.circle(frame_copy, (x, y), 3, (0, 255, 0), -1)
        
        # Draw connections
        for start_idx, end_idx in connections:
            if (pose_features.keypoint_confidences[start_idx] > 0.5 and
                pose_features.keypoint_confidences[end_idx] > 0.5):
                
                start_pt = (
                    int(keypoints[start_idx, 0] * w) + offset_x,
                    int(keypoints[start_idx, 1] * h) + offset_y
                )
                end_pt = (
                    int(keypoints[end_idx, 0] * w) + offset_x,
                    int(keypoints[end_idx, 1] * h) + offset_y
                )
                cv2.line(frame_copy, start_pt, end_pt, (0, 255, 0), 2)
        
        return frame_copy
    
    def close(self) -> None:
        """Close pose estimator and release resources."""
        self.pose.close()
        logger.info("PoseExtractor closed")


class BodyRatioCalculator:
    """Calculate body proportions from pose keypoints."""
    
    def __init__(self):
        """Initialize body ratio calculator."""
        logger.info("BodyRatioCalculator initialized")
    
    def calculate_ratios(self, keypoints: np.ndarray,
                         confidences: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive body ratios.
        
        Args:
            keypoints: (33, 3) array of keypoints
            confidences: (33,) array of confidence scores
            
        Returns:
            Dictionary of body ratio measurements
        """
        pose_extractor = PoseExtractor()
        return pose_extractor._calculate_body_ratios(keypoints, confidences)
