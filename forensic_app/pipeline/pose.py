"""
Skeletal Pose Biometrics & Cadence Estimator
Computes 17-keypoint scale-invariant anthropometrics, torso-to-leg ratios, and step cadence, with batched
per-crop pose inference that picks the skeleton belonging to each tracked person.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PoseResult = Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, float], str]

_KP_CONF = 0.30


def _resolve_weights(path: str) -> Optional[str]:
    """Locate a local weights file (as given, project root, or models/); None when absent (no download)."""
    p = Path(path)
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, PROJECT_ROOT / p, PROJECT_ROOT / "models" / p.name]
    for c in candidates:
        try:
            if c.is_file():
                return str(c)
        except OSError:
            continue
    return None


def _box_iou(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


class PoseBiometrics:
    """17-Keypoint Anthropometric & Gait Biometrics."""

    def __init__(self,
                 pose_model_path: str = "yolo11n-pose.pt",
                 device: str = "cpu",
                 cadence_window_len: int = 30,
                 fps: float = 25.0,
                 batch_imgsz: int = 320,
                 crop_padding: float = 0.08,
                 max_batch: int = 16):
        self.pose_model_path = pose_model_path
        self.device = device
        self.cadence_window_len = cadence_window_len
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.batch_imgsz = batch_imgsz
        self.crop_padding = crop_padding
        self.max_batch = max_batch

        # Ankle distance history per track_id for cadence calculation: track_id -> deque
        self.ankle_histories: Dict[int, deque] = {}

        self.model = None
        if YOLO is not None:
            weights = _resolve_weights(self.pose_model_path)
            if weights is not None:
                try:
                    self.model = YOLO(weights)
                    logger.info(f"PoseBiometrics initialized with {weights}")
                except Exception as e:
                    logger.error(f"Failed to load pose weights {weights}: {e}")
            else:
                logger.warning(f"Pose weights '{pose_model_path}' not found locally; pose disabled "
                               f"(automatic downloads are not allowed).")

    @staticmethod
    def _default_bio() -> Dict[str, float]:
        return {"torso_leg_ratio": 0.0, "shoulder_hip_ratio": 0.0, "cadence_hz": 0.0}

    def set_fps(self, fps: float) -> None:
        """Update the stream frame rate used for cadence estimation."""
        if fps and fps > 0:
            self.fps = float(fps)

    def forget_track(self, track_id: int) -> None:
        """Drop cadence history of a finished track."""
        self.ankle_histories.pop(track_id, None)

    # ------------------------------------------------------------------ inference
    def estimate_pose(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], track_id: int = 0) -> PoseResult:
        """
        Run pose estimation on person crop, compute biometric ratios, and classify posture.
        Returns: (keypoints_frame_17x3, keypoints_crop_17x3, biometrics_dict, posture_str)
        """
        return self.estimate_pose_batch(frame, [bbox], [track_id], imgsz=None)[0]

    def estimate_pose_batch(self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]],
                            track_ids: Optional[List[int]] = None,
                            imgsz: Optional[int] = -1) -> List[PoseResult]:
        """
        Batched pose estimation: one model call over all person crops (slightly padded), each letterboxed to
        `imgsz` (default `self.batch_imgsz`; None = model default). For every crop the skeleton whose box
        best matches the tracked person's box is selected (not blindly the first instance).
        Returns one (kpts_frame, kpts_crop, biometrics, posture) per bbox, in order; kpts_crop are relative
        to the ORIGINAL bbox origin (x1, y1), matching `frame[y1:y2, x1:x2]`.
        """
        n = len(bboxes)
        if track_ids is None:
            track_ids = list(range(n))
        results: List[PoseResult] = [(None, None, self._default_bio(), "Standing") for _ in range(n)]
        if self.model is None or frame is None or frame.size == 0 or n == 0:
            return results
        if imgsz == -1:
            imgsz = self.batch_imgsz

        fh, fw = frame.shape[:2]
        jobs = []  # (index, crop, pad_x1, pad_y1, target_box_in_crop, bbox_clipped)
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = (int(v) for v in bbox)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < 20 or bh < 20:
                continue
            px, py = int(self.crop_padding * bw), int(self.crop_padding * bh)
            cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
            cx2, cy2 = min(fw, x2 + px), min(fh, y2 + py)
            crop = np.ascontiguousarray(frame[cy1:cy2, cx1:cx2])
            target = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
            jobs.append((i, crop, cx1, cy1, target, (x1, y1, x2, y2)))

        for start in range(0, len(jobs), self.max_batch):
            chunk = jobs[start:start + self.max_batch]
            kwargs: Dict[str, Any] = {"conf": 0.25, "verbose": False, "device": self.device}
            if imgsz:
                kwargs["imgsz"] = int(imgsz)
            try:
                outs = self.model([j[1] for j in chunk], **kwargs)
            except Exception as e:
                logger.debug(f"Pose inference error: {e}")
                continue
            for job, r in zip(chunk, outs):
                i, crop, cx1, cy1, target, (x1, y1, x2, y2) = job
                kpts_pad = self._select_instance(r, target)
                if kpts_pad is None:
                    continue
                kpts_f = kpts_pad.copy()
                kpts_f[:, 0] += cx1
                kpts_f[:, 1] += cy1
                kpts_c = kpts_f.copy()
                kpts_c[:, 0] -= x1
                kpts_c[:, 1] -= y1
                biometrics = self._calculate_ratios(kpts_f, track_ids[i])
                posture = self._classify_posture(kpts_f, (y2 - y1, x2 - x1), biometrics.get("cadence_hz", 0.0))
                results[i] = (kpts_f, kpts_c, biometrics, posture)
        return results

    @staticmethod
    def _select_instance(result, target_box) -> Optional[np.ndarray]:
        """Pick the skeleton (17x3, crop coords) whose detection box best matches the expected person box."""
        kp = getattr(result, "keypoints", None)
        if kp is None or kp.data is None or len(kp.data) == 0:
            return None
        data = kp.data.cpu().numpy()
        if data.ndim != 3 or data.shape[1] < 17:
            return None
        if data.shape[2] == 2:
            data = np.concatenate([data, np.ones(data.shape[:2] + (1,), dtype=data.dtype)], axis=2)
        if len(data) == 1:
            return data[0][:17].astype(np.float32)

        boxes = None
        if getattr(result, "boxes", None) is not None and len(result.boxes) == len(data):
            boxes = result.boxes.xyxy.cpu().numpy()
            det_conf = result.boxes.conf.cpu().numpy()
        tx1, ty1, tx2, ty2 = (float(v) for v in target_box)
        tcx, tcy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
        diag = max(1.0, float(np.hypot(tx2 - tx1, ty2 - ty1)))
        best_i, best_s = 0, -1e9
        for j in range(len(data)):
            if boxes is not None:
                b = boxes[j]
                conf = float(det_conf[j])
            else:
                pts = data[j][data[j][:, 2] >= _KP_CONF][:, :2]
                if len(pts) == 0:
                    continue
                b = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
                conf = float(np.mean(data[j][:, 2]))
            bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            s = _box_iou(b, (tx1, ty1, tx2, ty2)) - 0.5 * (np.hypot(bcx - tcx, bcy - tcy) / diag) + 0.1 * conf
            if s > best_s:
                best_i, best_s = j, s
        return data[best_i][:17].astype(np.float32)

    # ------------------------------------------------------------------ biometrics
    def _calculate_ratios(self, keypoints: np.ndarray, track_id: int) -> Dict[str, float]:
        """
        Compute scale-invariant anthropometrics:
          torso_leg_ratio = ||Shoulder - Hip|| / ||Hip - Ankle||
          shoulder_hip_ratio = ||Shoulder_L - Shoulder_R|| / ||Hip_L - Hip_R||
          cadence_hz over a sliding window (uses self.fps)
        Ratios are 0.0 when the required joints are not confidently visible (e.g. upper-body views).
        """
        if keypoints is None or len(keypoints) < 17:
            return self._default_bio()

        kp = np.asarray(keypoints, dtype=np.float32)
        confs = kp[:, 2] if kp.shape[1] >= 3 else np.ones(len(kp), dtype=np.float32)

        def ok(*idx) -> bool:
            return all(float(confs[i]) >= _KP_CONF for i in idx)

        # COCO 17 Keypoints:
        # 5: L_shoulder, 6: R_shoulder
        # 11: L_hip, 12: R_hip
        # 13: L_knee, 14: R_knee
        # 15: L_ankle, 16: R_ankle
        ls, rs = kp[5][:2], kp[6][:2]
        lh, rh = kp[11][:2], kp[12][:2]
        la, ra = kp[15][:2], kp[16][:2]

        shoulder_center = (ls + rs) / 2.0
        hip_center = (lh + rh) / 2.0
        ankle_center = (la + ra) / 2.0

        torso_leg_ratio = 0.0
        if ok(5, 6, 11, 12, 15, 16):
            torso_len = np.linalg.norm(shoulder_center - hip_center)
            leg_len = np.linalg.norm(hip_center - ankle_center)
            torso_leg_ratio = float(torso_len / max(leg_len, 1.0))
            torso_leg_ratio = round(min(2.5, max(0.1, torso_leg_ratio)), 2)

        shoulder_hip_ratio = 0.0
        if ok(5, 6, 11, 12):
            shoulder_width = np.linalg.norm(ls - rs)
            hip_width = np.linalg.norm(lh - rh)
            shoulder_hip_ratio = float(shoulder_width / max(hip_width, 1.0))
            shoulder_hip_ratio = round(min(3.0, max(0.2, shoulder_hip_ratio)), 2)

        # Cadence (Hz) tracking via ankle distance oscillation (only from confidently visible ankles)
        if track_id not in self.ankle_histories:
            self.ankle_histories[track_id] = deque(maxlen=self.cadence_window_len)
        if ok(15, 16):
            self.ankle_histories[track_id].append(float(np.linalg.norm(la - ra)))

        cadence_hz = self._estimate_cadence(self.ankle_histories[track_id], fps=self.fps)

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
        duration_sec = len(history) / max(fps, 1e-3)
        if duration_sec > 0:
            cadence = num_cycles / duration_sec
            return float(min(3.5, max(0.0, cadence)))
        return 0.0

    @staticmethod
    def _classify_posture(keypoints: np.ndarray, crop_shape: Tuple[int, int], cadence_hz: float) -> str:
        """Classify biological posture based on joint angles and motion cadence."""
        if keypoints is None or len(keypoints) < 17:
            return "Standing"

        kp = np.asarray(keypoints, dtype=np.float32)
        if kp.shape[1] >= 3 and min(float(kp[i][2]) for i in (5, 6, 11, 12)) < _KP_CONF:
            # Torso joints not visible (upper-body / occluded view): posture from cadence only
            if cadence_hz > 2.0:
                return "Running"
            if cadence_hz > 0.8:
                return "Walking"
            return "Standing"

        h, w = crop_shape
        # Torso vertical tilt (Shoulder to Hip)
        sc = (kp[5][:2] + kp[6][:2]) / 2.0
        hc = (kp[11][:2] + kp[12][:2]) / 2.0

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
