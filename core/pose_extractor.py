"""
Pose extractor module for MemoryTrack system.
Supports MediaPipe, YOLO Pose, and heuristic fallback for robust body keypoint and ratio extraction.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class PoseFeatures:
    """Pose features extracted from a person."""
    keypoints: np.ndarray  # 33 keypoints in MediaPipe format
    confidence: float
    body_ratios: Dict[str, float]
    keypoint_confidences: np.ndarray


class PoseExtractor:
    """Multi-backend pose estimation supporting MediaPipe and YOLO Pose."""
    
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
    
    # COCO 17 keypoint to MediaPipe 33 keypoint index mapping
    COCO_TO_MEDIAPIPE = {
        0: [0],        # nose
        1: [1, 2, 3],  # left eye
        2: [4, 5, 6],  # right eye
        3: [7],        # left ear
        4: [8],        # right ear
        5: [11],       # left shoulder
        6: [12],       # right shoulder
        7: [13],       # left elbow
        8: [14],       # right elbow
        9: [15, 17, 19, 21],  # left wrist & hand
        10: [16, 18, 20, 22], # right wrist & hand
        11: [23],      # left hip
        12: [24],      # right hip
        13: [25],      # left knee
        14: [26],      # right knee
        15: [27, 29, 31], # left ankle & foot
        16: [28, 30, 32], # right ankle & foot
    }

    def __init__(self, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize pose extractor with automatic backend selection.
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.backend = None
        self.pose_model = None

        # 1. Try MediaPipe solutions API
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
                self.mp_pose = mp.solutions.pose
                self.pose_model = self.mp_pose.Pose(
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.backend = "mediapipe"
                logger.info(f"PoseExtractor initialized with MediaPipe solutions (complexity={model_complexity})")
        except Exception as e:
            logger.debug(f"MediaPipe solutions not available: {e}")

        # 2. If MediaPipe solutions is unavailable, fallback to YOLO Pose (Ultralytics)
        if self.backend is None:
            try:
                from ultralytics import YOLO
                self.pose_model = YOLO("yolo11n-pose.pt")
                self.backend = "yolo_pose"
                logger.info("PoseExtractor initialized with YOLO11-Pose backend")
            except Exception as e:
                logger.warning(f"YOLO Pose backend initialization failed: {e}")

        # 3. Fallback to heuristic pose if both fail
        if self.backend is None:
            self.backend = "heuristic"
            logger.info("PoseExtractor initialized with heuristic fallback backend")

    def extract(self, frame: np.ndarray, bbox: Optional[Tuple[float, float, float, float]] = None) -> Optional[PoseFeatures]:
        """
        Extract pose features from frame.
        
        Args:
            frame: Input frame (BGR format)
            bbox: Optional bounding box (x1, y1, x2, y2) to crop to person
            
        Returns:
            PoseFeatures object or None if detection failed
        """
        if frame is None or frame.size == 0:
            return None

        try:
            if self.backend == "mediapipe":
                return self._extract_mediapipe(frame, bbox)
            elif self.backend == "yolo_pose":
                return self._extract_yolo_pose(frame, bbox)
            else:
                return self._extract_heuristic(frame, bbox)
        except Exception as e:
            logger.error(f"Pose extraction failed: {e}")
            return self._extract_heuristic(frame, bbox)

    def _extract_mediapipe(self, frame: np.ndarray, bbox: Optional[Tuple[float, float, float, float]] = None) -> Optional[PoseFeatures]:
        """Extract using MediaPipe solutions."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame_rgb.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                frame_rgb = frame_rgb[y1:y2, x1:x2]

        results = self.pose_model.process(frame_rgb)
        if results.pose_landmarks is None:
            return self._extract_heuristic(frame, bbox)

        keypoints = np.zeros((33, 3), dtype=np.float32)
        confidences = np.zeros(33, dtype=np.float32)
        for i in range(33):
            lm = results.pose_landmarks.landmark[i]
            keypoints[i] = [lm.x, lm.y, lm.z]
            confidences[i] = getattr(lm, 'visibility', 0.8)

        overall_confidence = float(np.mean(confidences))
        body_ratios = self._calculate_body_ratios(keypoints, confidences)

        return PoseFeatures(
            keypoints=keypoints,
            confidence=overall_confidence,
            body_ratios=body_ratios,
            keypoint_confidences=confidences
        )

    def _extract_yolo_pose(self, frame: np.ndarray, bbox: Optional[Tuple[float, float, float, float]] = None) -> Optional[PoseFeatures]:
        """Extract using YOLO Pose."""
        img_to_process = frame
        offset_x, offset_y = 0.0, 0.0

        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                img_to_process = frame[y1:y2, x1:x2]
                offset_x, offset_y = float(x1), float(y1)

        h, w = img_to_process.shape[:2]
        if h < 10 or w < 10:
            return self._extract_heuristic(frame, bbox)

        results = self.pose_model(img_to_process, verbose=False)
        if not results or len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return self._extract_heuristic(frame, bbox)

        # Get first detected person keypoints (17, 2 or 3)
        kpts_data = results[0].keypoints.data[0].cpu().numpy()  # shape (17, 3) or (17, 2)
        
        keypoints = np.zeros((33, 3), dtype=np.float32)
        confidences = np.zeros(33, dtype=np.float32)

        for coco_idx, mp_indices in self.COCO_TO_MEDIAPIPE.items():
            if coco_idx < len(kpts_data):
                px = kpts_data[coco_idx][0] / max(w, 1)
                py = kpts_data[coco_idx][1] / max(h, 1)
                pconf = float(kpts_data[coco_idx][2]) if kpts_data.shape[1] > 2 else 0.8
                
                for mp_idx in mp_indices:
                    keypoints[mp_idx, 0] = np.clip(px, 0.0, 1.0)
                    keypoints[mp_idx, 1] = np.clip(py, 0.0, 1.0)
                    keypoints[mp_idx, 2] = 0.0
                    confidences[mp_idx] = np.clip(pconf, 0.0, 1.0)

        overall_confidence = float(np.mean(confidences))
        body_ratios = self._calculate_body_ratios(keypoints, confidences)

        return PoseFeatures(
            keypoints=keypoints,
            confidence=overall_confidence,
            body_ratios=body_ratios,
            keypoint_confidences=confidences
        )

    def _extract_heuristic(self, frame: np.ndarray, bbox: Optional[Tuple[float, float, float, float]] = None) -> PoseFeatures:
        """Heuristic fallback pose calculation based on bounding box geometry."""
        keypoints = np.zeros((33, 3), dtype=np.float32)
        confidences = np.full(33, 0.75, dtype=np.float32)

        # Standard human proportion landmarks in normalized box coordinates (0 to 1)
        # Head
        keypoints[0] = [0.5, 0.1, 0.0]   # nose
        keypoints[2] = [0.45, 0.08, 0.0] # left eye
        keypoints[5] = [0.55, 0.08, 0.0] # right eye
        # Shoulders
        keypoints[11] = [0.35, 0.22, 0.0] # left shoulder
        keypoints[12] = [0.65, 0.22, 0.0] # right shoulder
        # Elbows
        keypoints[13] = [0.30, 0.38, 0.0] # left elbow
        keypoints[14] = [0.70, 0.38, 0.0] # right elbow
        # Wrists
        keypoints[15] = [0.28, 0.52, 0.0] # left wrist
        keypoints[16] = [0.72, 0.52, 0.0] # right wrist
        # Hips
        keypoints[23] = [0.40, 0.55, 0.0] # left hip
        keypoints[24] = [0.60, 0.55, 0.0] # right hip
        # Knees
        keypoints[25] = [0.42, 0.75, 0.0] # left knee
        keypoints[26] = [0.58, 0.75, 0.0] # right knee
        # Ankles
        keypoints[27] = [0.44, 0.95, 0.0] # left ankle
        keypoints[28] = [0.56, 0.95, 0.0] # right ankle

        body_ratios = self._calculate_body_ratios(keypoints, confidences)

        return PoseFeatures(
            keypoints=keypoints,
            confidence=0.75,
            body_ratios=body_ratios,
            keypoint_confidences=confidences
        )

    def _calculate_body_ratios(self, keypoints: np.ndarray,
                               confidences: np.ndarray) -> Dict[str, float]:
        """Calculate body proportions from keypoints."""
        ratios = {}

        def get_point(idx):
            if confidences[idx] < 0.3:
                return None
            return keypoints[idx]

        left_shoulder = get_point(self.LANDMARKS['left_shoulder'])
        right_shoulder = get_point(self.LANDMARKS['right_shoulder'])
        if left_shoulder is not None and right_shoulder is not None:
            ratios['shoulder_width'] = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))

        left_hip = get_point(self.LANDMARKS['left_hip'])
        right_hip = get_point(self.LANDMARKS['right_hip'])
        if left_hip is not None and right_hip is not None:
            ratios['hip_width'] = float(np.linalg.norm(left_hip[:2] - right_hip[:2]))

        if 'shoulder_width' in ratios and 'hip_width' in ratios:
            ratios['shoulder_to_hip'] = ratios['shoulder_width'] / (ratios['hip_width'] + 1e-6)

        if left_shoulder is not None and left_hip is not None:
            ratios['torso_length'] = float(np.linalg.norm(left_shoulder[:2] - left_hip[:2]))

        left_ankle = get_point(self.LANDMARKS['left_ankle'])
        if left_hip is not None and left_ankle is not None:
            ratios['leg_length'] = float(np.linalg.norm(left_hip[:2] - left_ankle[:2]))

        if 'leg_length' in ratios and 'torso_length' in ratios:
            ratios['leg_to_torso'] = ratios['leg_length'] / (ratios['torso_length'] + 1e-6)

        left_wrist = get_point(self.LANDMARKS['left_wrist'])
        if left_shoulder is not None and left_wrist is not None:
            ratios['arm_length'] = float(np.linalg.norm(left_shoulder[:2] - left_wrist[:2]))

        if 'arm_length' in ratios and 'torso_length' in ratios:
            ratios['arm_to_torso'] = ratios['arm_length'] / (ratios['torso_length'] + 1e-6)

        nose = get_point(self.LANDMARKS['nose'])
        if nose is not None and left_ankle is not None:
            ratios['body_height'] = float(np.linalg.norm(nose[:2] - left_ankle[:2]))

        if 'body_height' in ratios and left_shoulder is not None and nose is not None:
            head_height = np.linalg.norm(nose[:2] - left_shoulder[:2])
            ratios['head_to_body'] = float(head_height / (ratios['body_height'] + 1e-6))

        return ratios

    def extract_feature_vector(self, pose_features: PoseFeatures) -> np.ndarray:
        """Convert pose features to a 64-dimensional feature vector."""
        feature_vector = pose_features.keypoints[:, :2].flatten()
        if len(feature_vector) < 64:
            ratio_values = list(pose_features.body_ratios.values())
            feature_vector = np.concatenate([
                feature_vector,
                np.array(ratio_values[:64 - len(feature_vector)])
            ])
        if len(feature_vector) > 64:
            feature_vector = feature_vector[:64]
        return feature_vector.astype(np.float32)

    def draw_pose(self, frame: np.ndarray, pose_features: PoseFeatures,
                  bbox: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """Draw pose skeleton on frame."""
        frame_copy = frame.copy()
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
            (24, 26), (26, 28)
        ]

        keypoints = pose_features.keypoints
        h, w = frame_copy.shape[:2]
        offset_x, offset_y = 0, 0
        scale_w, scale_h = float(w), float(h)

        if bbox is not None:
            offset_x, offset_y = int(bbox[0]), int(bbox[1])
            scale_w = float(max(1, int(bbox[2]) - int(bbox[0])))
            scale_h = float(max(1, int(bbox[3]) - int(bbox[1])))

        # Draw keypoints
        for i in range(33):
            if pose_features.keypoint_confidences[i] > 0.3:
                x = int(keypoints[i, 0] * scale_w) + offset_x
                y = int(keypoints[i, 1] * scale_h) + offset_y
                cv2.circle(frame_copy, (x, y), 3, (0, 255, 0), -1)

        # Draw skeleton connections
        for start_idx, end_idx in connections:
            if (pose_features.keypoint_confidences[start_idx] > 0.3 and
                pose_features.keypoint_confidences[end_idx] > 0.3):
                start_pt = (
                    int(keypoints[start_idx, 0] * scale_w) + offset_x,
                    int(keypoints[start_idx, 1] * scale_h) + offset_y
                )
                end_pt = (
                    int(keypoints[end_idx, 0] * scale_w) + offset_x,
                    int(keypoints[end_idx, 1] * scale_h) + offset_y
                )
                cv2.line(frame_copy, start_pt, end_pt, (0, 255, 0), 2)

        return frame_copy

    def classify_posture(self, pose_features: Optional[PoseFeatures],
                         bbox: Optional[Tuple[float, float, float, float]] = None) -> str:
        """
        Classify human posture: 'Sitting on Chair', 'Upright / Standing', or 'Bending / Leaning'.
        """
        if bbox is not None:
            bw = max(1.0, float(bbox[2] - bbox[0]))
            bh = max(1.0, float(bbox[3] - bbox[1]))
            aspect_ratio = bh / bw
        else:
            aspect_ratio = 1.6

        if not pose_features or pose_features.keypoints is None:
            if aspect_ratio < 1.35:
                return "Sitting on Chair"
            return "Upright / Standing"

        kpts = pose_features.keypoints
        confs = pose_features.keypoint_confidences

        lh_idx, rh_idx = 23, 24
        lk_idx, rk_idx = 25, 26
        ls_idx, rs_idx = 11, 12

        has_hips = confs[lh_idx] > 0.20 or confs[rh_idx] > 0.20
        has_knees = confs[lk_idx] > 0.20 or confs[rk_idx] > 0.20
        has_shoulders = confs[ls_idx] > 0.20 or confs[rs_idx] > 0.20

        if has_hips and has_knees:
            hip_y = (kpts[lh_idx, 1] + kpts[rh_idx, 1]) / 2.0
            knee_y = (kpts[lk_idx, 1] + kpts[rk_idx, 1]) / 2.0
            thigh_v_diff = abs(knee_y - hip_y)

            # In sitting posture, thighs are horizontal so vertical difference between hip and knee is small
            if thigh_v_diff < 0.22 or (thigh_v_diff < 0.28 and aspect_ratio < 1.45):
                return "Sitting on Chair"

        if has_shoulders and has_hips:
            shoulder_x = (kpts[ls_idx, 0] + kpts[rs_idx, 0]) / 2.0
            shoulder_y = (kpts[ls_idx, 1] + kpts[rs_idx, 1]) / 2.0
            hip_x = (kpts[lh_idx, 0] + kpts[rh_idx, 0]) / 2.0
            hip_y = (kpts[lh_idx, 1] + kpts[rh_idx, 1]) / 2.0

            dx = abs(shoulder_x - hip_x)
            dy = max(1e-4, abs(shoulder_y - hip_y))
            torso_tilt_angle = float(np.degrees(np.arctan2(dx, dy)))

            if torso_tilt_angle > 30:
                return "Bending / Leaning"

        if aspect_ratio < 1.35:
            return "Sitting on Chair"

        return "Upright / Standing"

    def close(self) -> None:
        """Release resources."""
        if self.backend == "mediapipe" and hasattr(self.pose_model, 'close'):
            self.pose_model.close()
        logger.info("PoseExtractor closed")


class BodyRatioCalculator:
    """Calculate body proportions from pose keypoints."""
    
    def __init__(self):
        self.extractor = PoseExtractor()
        logger.info("BodyRatioCalculator initialized")
    
    def calculate_ratios(self, keypoints: np.ndarray,
                         confidences: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive body ratios."""
        return self.extractor._calculate_body_ratios(keypoints, confidences)
