"""
Skeletal Pose Biometrics & Cadence Estimator
Computes 17-keypoint scale-invariant anthropometrics, torso-to-leg ratios, and step cadence.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)


class PoseBiometrics:
    """17-Keypoint Anthropometric & Gait Biometrics."""

    def __init__(self,
                 pose_model_path: str = "yolo11n-pose.pt",
                 device: str = "cpu",
                 cadence_window_len: int = 30):
        self.pose_model_path = pose_model_path
        self.device = device
        self.cadence_window_len = cadence_window_len

        # Ankle distance history per track_id for cadence calculation: track_id -> deque
        self.ankle_histories: Dict[int, deque] = {}

        if YOLO is not None:
            self.model = YOLO(self.pose_model_path)
            logger.info(f"PoseBiometrics initialized with {pose_model_path}")
        else:
            self.model = None

    def estimate_pose(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], track_id: int = 0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, float], str]:
        """
        Run pose estimation on person crop, compute biometric ratios, and classify posture.
        Returns: (keypoints_frame_17x3, keypoints_crop_17x3, biometrics_dict, posture_str)
        """
        default_bio = {"torso_leg_ratio": 0.0, "shoulder_hip_ratio": 0.0, "cadence_hz": 0.0}
        if self.model is None or frame is None or frame.size == 0:
            return None, None, default_bio, "Standing"

        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None, None, default_bio, "Standing"

        # Run pose model on person crop
        try:
            results = self.model(crop, conf=0.25, verbose=False, device=self.device)
        except Exception as e:
            logger.debug(f"Pose inference error: {e}")
            return None, None, default_bio, "Standing"

        keypoints_frame = None
        keypoints_crop = None
        for r in results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                kpts_data = r.keypoints.data[0].cpu().numpy()  # Expecting shape (17, 3) or (17, 2)
                # Strict shape verification: must be 2D with at least 17 rows
                if kpts_data is not None and kpts_data.ndim == 2 and kpts_data.shape[0] >= 17:
                    keypoints_crop = kpts_data.copy()
                    kpts_f = kpts_data.copy()
                    kpts_f[:, 0] += x1
                    kpts_f[:, 1] += y1
                    keypoints_frame = kpts_f
                    break

        # If no keypoints or invalid shape, return safe defaults
        if keypoints_frame is None or len(keypoints_frame) < 17:
            return None, None, default_bio, "Standing"

        # Compute Invariant Ratios as per SKILL.md Section 3.C
        biometrics = self._calculate_ratios(keypoints_frame, track_id)
        posture = self._classify_posture(keypoints_frame, crop.shape[:2], biometrics.get("cadence_hz", 0.0))

        return keypoints_frame, keypoints_crop, biometrics, posture

    def _calculate_ratios(self, keypoints: np.ndarray, track_id: int) -> Dict[str, float]:
        """
        Compute scale-invariant anthropometrics:
          torso_leg_ratio = ||Shoulder - Hip|| / ||Hip - Ankle||
          shoulder_hip_ratio = ||Shoulder_L - Shoulder_R|| / ||Hip_L - Hip_R||
          cadence_hz over 30-frame window
        """
        if keypoints is None or len(keypoints) < 17:
            return {"torso_leg_ratio": 0.0, "shoulder_hip_ratio": 0.0, "cadence_hz": 0.0}

        # COCO 17 Keypoints:
        # 5: L_shoulder, 6: R_shoulder
        # 11: L_hip, 12: R_hip
        # 13: L_knee, 14: R_knee
        # 15: L_ankle, 16: R_ankle
        ls, rs = keypoints[5][:2], keypoints[6][:2]
        lh, rh = keypoints[11][:2], keypoints[12][:2]
        la, ra = keypoints[15][:2], keypoints[16][:2]

        shoulder_center = (ls + rs) / 2.0
        hip_center = (lh + rh) / 2.0
        ankle_center = (la + ra) / 2.0

        torso_len = np.linalg.norm(shoulder_center - hip_center)
        leg_len = np.linalg.norm(hip_center - ankle_center)

        torso_leg_ratio = float(torso_len / max(leg_len, 1.0))
        torso_leg_ratio = round(min(2.5, max(0.1, torso_leg_ratio)), 2)

        shoulder_width = np.linalg.norm(ls - rs)
        hip_width = np.linalg.norm(lh - rh)
        shoulder_hip_ratio = float(shoulder_width / max(hip_width, 1.0))
        shoulder_hip_ratio = round(min(3.0, max(0.2, shoulder_hip_ratio)), 2)

        # Cadence (Hz) tracking via ankle distance oscillation
        ankle_dist = float(np.linalg.norm(la - ra))
        if track_id not in self.ankle_histories:
            self.ankle_histories[track_id] = deque(maxlen=self.cadence_window_len)
        self.ankle_histories[track_id].append(ankle_dist)

        cadence_hz = self._estimate_cadence(self.ankle_histories[track_id])

        return {
            "torso_leg_ratio": torso_leg_ratio,
            "shoulder_hip_ratio": shoulder_hip_ratio,
            "cadence_hz": round(cadence_hz, 2)
        }

    @staticmethod
    def _estimate_cadence(history: deque, fps: float = 25.0) -> float:
        """Estimate stepping cadence (Hz) from ankle oscillation frequency."""
        if len(history) < 15:
            return 0.0

        arr = np.array(history)
        arr = arr - np.mean(arr)  # Remove DC offset
        # Zero crossings count
        zero_crossings = np.where(np.diff(np.sign(arr)))[0]
        num_cycles = len(zero_crossings) / 2.0
        duration_sec = len(history) / fps
        if duration_sec > 0:
            cadence = num_cycles / duration_sec
            return float(min(3.5, max(0.0, cadence)))
        return 0.0

    @staticmethod
    def _classify_posture(keypoints: np.ndarray, crop_shape: Tuple[int, int], cadence_hz: float) -> str:
        """Classify biological posture based on joint angles and motion cadence."""
        if keypoints is None or len(keypoints) < 17:
            return "Standing"

        h, w = crop_shape
        # Torso vertical tilt (Shoulder to Hip)
        sc = (keypoints[5][:2] + keypoints[6][:2]) / 2.0
        hc = (keypoints[11][:2] + keypoints[12][:2]) / 2.0

        dy = hc[1] - sc[1]
        dx = abs(hc[0] - sc[0])

        aspect = float(h) / max(w, 1)

        # Sitting detection
        if aspect < 1.35 and dy < h * 0.40:
            return "Sitting"

        # Bending / Leaning detection
        if dx > (dy * 0.65) and dy > 10:
            return "Bending / Leaning"

        # Walking / Running detection via step cadence
        if cadence_hz > 2.0:
            return "Running"
        elif cadence_hz > 0.8:
            return "Walking"

        return "Standing"
