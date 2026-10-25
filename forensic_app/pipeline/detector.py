"""
Pedestrian Detection & Tracklet Manager
Implements YOLOv11 person detection + ByteTrack with 60-frame tracklet archiving.
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)


@dataclass
class Tracklet:
    """Active temporal tracklet for a pedestrian."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    age: int = 1
    hits: int = 1
    lost_frames: int = 0
    history: List[Tuple[int, int]] = field(default_factory=list)
    archived: bool = False


class PedestrianDetector:
    """YOLOv11-based person detection with tracklet persistence."""

    def __init__(self,
                 model_path: str = "yolo11n.pt",
                 conf_thresh: float = 0.45,
                 iou_thresh: float = 0.40,
                 device: str = "cpu",
                 archive_lost_threshold: int = 60):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.archive_lost_threshold = archive_lost_threshold
        self.device = self._verify_device(device)

        if YOLO is not None:
            self.model = YOLO(self.model_path)
        else:
            self.model = None
            logger.warning("Ultralytics YOLO not installed.")

        self.tracklets: Dict[int, Tracklet] = {}
        self.next_track_id = 1
        logger.info(f"PedestrianDetector initialized ({model_path} on {self.device})")

    def _verify_device(self, req_device: str) -> str:
        if req_device == "cuda":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return "cpu"

    def detect_and_track(self, frame: np.ndarray) -> Tuple[List[Tracklet], List[Tracklet], List[Tuple[str, Tuple[int, int, int, int], float]]]:
        """
        Detect pedestrians, update active tracklets, and extract carried accessories from full scene.
        Returns: (active_tracklets, newly_archived_tracklets, frame_accessories)
        """
        if self.model is None or frame is None or frame.size == 0:
            return [], [], []

        h, w = frame.shape[:2]
        # COCO accessory class map
        COCO_ACCESSORIES = {
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            39: "bottle",
            67: "cell phone",
            73: "laptop"
        }

        # Run multi-class detection on full scene
        try:
            results = self.model(frame, conf=0.25, iou=self.iou_thresh, verbose=False, device=self.device)
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return [], [], []

        detections = []
        frame_accessories = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                bx1 = int(max(0, min(w - 1, x1)))
                by1 = int(max(0, min(h - 1, y1)))
                bx2 = int(max(bx1 + 10, min(w, x2)))
                by2 = int(max(by1 + 10, min(h, y2)))

                if cls_id == 0 and conf >= self.conf_thresh:  # COCO Person
                    detections.append(((bx1, by1, bx2, by2), conf))
                elif cls_id in COCO_ACCESSORIES and conf >= 0.22:
                    acc_name = COCO_ACCESSORIES[cls_id]
                    frame_accessories.append((acc_name, (bx1, by1, bx2, by2), conf))

        # Spatial tracking association
        active_list, archived_list = self._update_tracks(detections)
        return active_list, archived_list, frame_accessories

    @staticmethod
    def associate_accessories(person_bbox: Tuple[int, int, int, int],
                              frame_accessories: List[Tuple[str, Tuple[int, int, int, int], float]]) -> List[str]:
        """
        Associate frame-level detected accessories with a person bounding box.
        """
        matched = []
        px1, py1, px2, py2 = person_bbox
        pw = px2 - px1
        ph = py2 - py1

        # Generous envelope for carried/worn items
        margin_x = int(pw * 0.25)
        margin_y = int(ph * 0.15)
        env_x1 = px1 - margin_x
        env_y1 = py1 - margin_y
        env_x2 = px2 + margin_x
        env_y2 = py2 + margin_y

        for acc_name, (ax1, ay1, ax2, ay2), aconf in frame_accessories:
            acx = (ax1 + ax2) / 2.0
            acy = (ay1 + ay2) / 2.0

            # 1. Center containment
            if env_x1 <= acx <= env_x2 and env_y1 <= acy <= env_y2:
                if acc_name not in matched:
                    matched.append(acc_name)
                continue

            # 2. Intersection Over Accessory area
            ix1 = max(env_x1, ax1)
            iy1 = max(env_y1, ay1)
            ix2 = min(env_x2, ax2)
            iy2 = min(env_y2, ay2)

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                acc_area = (ax2 - ax1) * (ay2 - ay1)
                if acc_area > 0 and (inter_area / acc_area) >= 0.30:
                    if acc_name not in matched:
                        matched.append(acc_name)

        return matched

    def _update_tracks(self, detections: List[Tuple[Tuple[int, int, int, int], float]]) -> Tuple[List[Tracklet], List[Tracklet]]:
        """
        Associate detections with existing tracks using a two-stage association strategy:
        Stage 1: Global Maximum IoU Matching
        Stage 2: Spatial Center Proximity & Scale Consistency for fast-moving pedestrians
        """
        assigned_det = set()
        assigned_trk = set()
        matched_pairs = []

        track_ids = list(self.tracklets.keys())

        # Stage 1: Global Maximum IoU Matching (Sort highest IoU first)
        iou_pairs = []
        for tid in track_ids:
            t_bbox = self.tracklets[tid].bbox
            for d_idx, (d_bbox, _) in enumerate(detections):
                iou = self._iou(t_bbox, d_bbox)
                if iou >= 0.20:
                    iou_pairs.append((iou, tid, d_idx))

        iou_pairs.sort(key=lambda x: x[0], reverse=True)
        for iou, tid, d_idx in iou_pairs:
            if tid not in assigned_trk and d_idx not in assigned_det:
                matched_pairs.append((tid, d_idx))
                assigned_trk.add(tid)
                assigned_det.add(d_idx)

        # Stage 2: Spatial Center Proximity Matching for fast movement or momentary detection jitter
        unassigned_trks = [tid for tid in track_ids if tid not in assigned_trk]
        unassigned_dets = [d_idx for d_idx in range(len(detections)) if d_idx not in assigned_det]

        dist_pairs = []
        for tid in unassigned_trks:
            trk = self.tracklets[tid]
            tb = trk.bbox
            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0
            tw, th = max(1, tb[2] - tb[0]), max(1, tb[3] - tb[1])
            max_dist = max(tw, th) * (1.1 + 0.25 * min(trk.lost_frames, 5))

            for d_idx in unassigned_dets:
                db = detections[d_idx][0]
                dcx, dcy = (db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0
                dist = np.hypot(tcx - dcx, tcy - dcy)
                if dist <= max_dist:
                    dw, dh = max(1, db[2] - db[0]), max(1, db[3] - db[1])
                    area_ratio = min(tw * th, dw * dh) / max(tw * th, dw * dh, 1)
                    if area_ratio >= 0.35:
                        dist_pairs.append((dist, tid, d_idx))

        dist_pairs.sort(key=lambda x: x[0])  # Shortest distance first
        for dist, tid, d_idx in dist_pairs:
            if tid not in assigned_trk and d_idx not in assigned_det:
                matched_pairs.append((tid, d_idx))
                assigned_trk.add(tid)
                assigned_det.add(d_idx)

        # Update matched tracks
        for tid, d_idx in matched_pairs:
            bbox, conf = detections[d_idx]
            trk = self.tracklets[tid]
            trk.bbox = bbox
            trk.confidence = conf
            trk.age += 1
            trk.hits += 1
            trk.lost_frames = 0
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            trk.history.append((cx, cy))
            if len(trk.history) > 60:
                trk.history.pop(0)

        # Increment lost frames for unmatched tracks
        archived_list = []
        to_delete = []

        for tid in track_ids:
            if tid not in assigned_trk:
                trk = self.tracklets[tid]
                trk.lost_frames += 1
                trk.age += 1
                # If lost > 60 frames, archive to Re-ID pool
                if trk.lost_frames >= self.archive_lost_threshold and not trk.archived:
                    trk.archived = True
                    archived_list.append(trk)
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracklets[tid]

        # Initialize new tracklets for unassigned detections
        for d_idx, (bbox, conf) in enumerate(detections):
            if d_idx not in assigned_det:
                new_id = self.next_track_id
                self.next_track_id += 1
                cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                new_trk = Tracklet(
                    track_id=new_id,
                    bbox=bbox,
                    confidence=conf,
                    history=[(cx, cy)]
                )
                self.tracklets[new_id] = new_trk

        # Active tracks: include current matches plus momentarily coasting tracks (lost <= 2 frames)
        active_list = [
            t for t in self.tracklets.values()
            if (t.lost_frames == 0 or (t.lost_frames <= 2 and t.hits >= 4))
        ]
        return active_list, archived_list

    @staticmethod
    def _iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return float(inter / union) if union > 0 else 0.0
