"""
Pedestrian Detection & Tracklet Manager
Implements YOLOv11 person + accessory detection, constant-velocity tracklet association with 60-frame
archiving, exclusive accessory-to-person assignment, and upscaled ROI accessory re-detection.
"""

import os
import glob
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from forensic_app.core.catalog import (
    COCO_ACCESSORY_CLASSES, ACCESSORY_LABELS, OPEN_VOCAB_PROMPTS, normalize_accessory, accessory_threshold,
    ACCESSORY_CONF_THRESHOLDS,
)
from forensic_app.pipeline.attributes import accessory_person_affinity

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]
AccessoryDet = Tuple[str, Box, float]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Stronger local COCO detectors used for the ROI accessory pass when present (never downloaded)
_STRONGER_ACCESSORY_MODELS = ("yolo11m.pt", "yolo11s.pt", "yolov8m.pt", "yolov8s.pt", "yolo11l.pt", "yolov8l.pt")
_MIN_ACCESSORY_CONF = min(ACCESSORY_CONF_THRESHOLDS[n] for n in COCO_ACCESSORY_CLASSES.values())


def resolve_weights(path: str) -> Optional[str]:
    """Locate a local weights file (as given, project root, or models/); None when absent (no download)."""
    if not path:
        return None
    p = Path(path)
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, PROJECT_ROOT / p, PROJECT_ROOT / "models" / p.name]
    for c in candidates:
        try:
            if c.is_file():
                return str(c)
        except OSError:
            continue
    return None


def model_accessory_classes(model) -> Dict[int, str]:
    """Class id -> canonical accessory name for any YOLO model (COCO ids, or names for open-vocab models)."""
    names = getattr(model, "names", None)
    if not names:
        return dict(COCO_ACCESSORY_CLASSES)
    items = names.items() if isinstance(names, dict) else enumerate(names)
    out: Dict[int, str] = {}
    for cid, cname in items:
        key = str(cname).strip().lower()
        canonical = normalize_accessory(OPEN_VOCAB_PROMPTS.get(key, key))
        if canonical in ACCESSORY_LABELS:
            out[int(cid)] = canonical
    return out


def detect_accessories_with_model(model, frame: np.ndarray, roi_bbox: Box, imgsz: int = 480,
                                  device: str = "cpu", class_map: Optional[Dict[int, str]] = None,
                                  iou: float = 0.45) -> List[AccessoryDet]:
    """
    Run `model` on an ROI of `frame` (letterboxed/upscaled to `imgsz`) and return accessory detections
    above their per-class thresholds as (name, bbox in FRAME coords, conf).
    """
    if model is None or frame is None or getattr(frame, "ndim", 0) != 3:
        return []
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in roi_bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return []
    roi = np.ascontiguousarray(frame[y1:y2, x1:x2])
    cmap = class_map if class_map is not None else model_accessory_classes(model)
    if not cmap:
        return []
    min_conf = min(accessory_threshold(n) for n in cmap.values())
    try:
        results = model(roi, imgsz=int(imgsz), conf=min_conf, iou=iou, classes=sorted(cmap.keys()),
                        verbose=False, device=device)
    except Exception as e:
        logger.debug(f"ROI accessory inference error: {e}")
        return []
    out: List[AccessoryDet] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        cls = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        for c, cf, b in zip(cls, confs, boxes):
            name = cmap.get(int(c))
            if name is None or float(cf) < accessory_threshold(name):
                continue
            bx1 = int(max(0, min(fw - 1, b[0] + x1)))
            by1 = int(max(0, min(fh - 1, b[1] + y1)))
            bx2 = int(max(bx1 + 1, min(fw, b[2] + x1)))
            by2 = int(max(by1 + 1, min(fh, b[3] + y1)))
            out.append((name, (bx1, by1, bx2, by2), float(cf)))
    return out


@dataclass
class Tracklet:
    """Active temporal tracklet for a pedestrian."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (last measured)
    confidence: float
    age: int = 1
    hits: int = 1
    lost_frames: int = 0
    history: List[Tuple[int, int]] = field(default_factory=list)
    archived: bool = False
    velocity: Tuple[float, float] = (0.0, 0.0)  # centre velocity (px / frame, EMA)

    def predicted_bbox(self, steps: Optional[int] = None) -> Tuple[int, int, int, int]:
        """Constant-velocity prediction of the box `steps` frames after the last measurement."""
        n = (self.lost_frames + 1) if steps is None else steps
        n = min(max(n, 0), 8)  # never extrapolate far
        dx, dy = self.velocity[0] * n, self.velocity[1] * n
        x1, y1, x2, y2 = self.bbox
        return (int(round(x1 + dx)), int(round(y1 + dy)), int(round(x2 + dx)), int(round(y2 + dy)))


class PedestrianDetector:
    """YOLOv11-based person detection with tracklet persistence."""

    def __init__(self,
                 model_path: str = "yolo11n.pt",
                 conf_thresh: float = 0.45,
                 iou_thresh: float = 0.40,
                 device: str = "cpu",
                 archive_lost_threshold: int = 60,
                 accessory_model_path: Optional[str] = "auto",
                 roi_imgsz: int = 480):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.archive_lost_threshold = archive_lost_threshold
        self.roi_imgsz = roi_imgsz
        self.device = self._verify_device(device)

        self.model = None
        self.accessory_model = None
        self.accessory_model_name = "none"
        if YOLO is not None:
            weights = resolve_weights(self.model_path)
            if weights is not None:
                try:
                    self.model = YOLO(weights)
                except Exception as e:
                    logger.error(f"Failed to load detector weights {weights}: {e}")
            else:
                logger.warning(f"Detector weights '{model_path}' not found locally; detection disabled "
                               f"(automatic downloads are not allowed).")
            self._init_accessory_model(accessory_model_path)
        else:
            logger.warning("Ultralytics YOLO not installed.")

        self.class_map: Dict[int, str] = model_accessory_classes(self.model) if self.model is not None else {}
        self.accessory_class_map: Dict[int, str] = (
            model_accessory_classes(self.accessory_model) if self.accessory_model is not None else {}
        )

        self.tracklets: Dict[int, Tracklet] = {}
        self.next_track_id = 1
        logger.info(f"PedestrianDetector initialized ({model_path} on {self.device}; "
                    f"accessory model: {self.accessory_model_name})")

    def _verify_device(self, req_device: str) -> str:
        if req_device == "cuda":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return "cpu"

    # ------------------------------------------------------------------ models
    def _init_accessory_model(self, accessory_model_path: Optional[str]) -> None:
        """Pick the ROI accessory model: explicit path > usable YOLO-World > stronger local COCO > main model."""
        if accessory_model_path not in (None, "", "auto"):
            weights = resolve_weights(accessory_model_path)
            if weights is not None and self._try_load_accessory(weights):
                return
        if accessory_model_path is not None:
            # Open-vocabulary YOLO-World: only when present AND its text encoder is available offline
            world = sorted(set(glob.glob(str(PROJECT_ROOT / "*world*.pt"))
                               + glob.glob(str(PROJECT_ROOT / "models" / "*world*.pt"))))
            if world and self._clip_available():
                for wpath in world:
                    if self._try_load_accessory(wpath, open_vocab=True):
                        return
            for name in _STRONGER_ACCESSORY_MODELS:
                if os.path.basename(self.model_path) == name:
                    continue
                weights = resolve_weights(name)
                if weights is not None and self._try_load_accessory(weights):
                    return
        self.accessory_model = self.model
        self.accessory_model_name = os.path.basename(self.model_path) if self.model is not None else "none"

    def _try_load_accessory(self, weights: str, open_vocab: bool = False) -> bool:
        try:
            model = YOLO(weights)
            if open_vocab:
                model.set_classes(list(OPEN_VOCAB_PROMPTS.keys()))
            self.accessory_model = model
            self.accessory_model_name = os.path.basename(weights)
            return True
        except Exception as e:
            logger.info(f"Accessory model {weights} unusable ({e}); trying next option")
            return False

    @staticmethod
    def _clip_available() -> bool:
        """
        YOLO-World set_classes needs the CLIP package AND its cached ViT-B/32 weights; otherwise ultralytics
        would try to pip-install / download them, which is not allowed here.
        """
        try:
            import clip  # noqa: F401
        except Exception:
            return False
        cache = Path(os.path.expanduser("~")) / ".cache" / "clip" / "ViT-B-32.pt"
        return cache.is_file()

    # ------------------------------------------------------------------ detection
    def detect_and_track(self, frame: np.ndarray) -> Tuple[List[Tracklet], List[Tracklet], List[Tuple[str, Tuple[int, int, int, int], float]]]:
        """
        Detect pedestrians, update active tracklets, and extract carried accessories from full scene.
        Returns: (active_tracklets, newly_archived_tracklets, frame_accessories)
        """
        if self.model is None or frame is None or frame.size == 0:
            return [], [], []

        h, w = frame.shape[:2]
        acc_map = self.class_map
        classes = sorted({0} | set(acc_map.keys()))
        model_conf = min(self.conf_thresh, _MIN_ACCESSORY_CONF, 0.25)

        # Run multi-class detection on full scene
        try:
            results = self.model(frame, conf=model_conf, iou=self.iou_thresh, classes=classes,
                                 verbose=False, device=self.device)
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return [], [], []

        detections = []
        frame_accessories = []

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            for cls_id, conf, (x1, y1, x2, y2) in zip(cls_ids, confs, boxes):
                conf = float(conf)
                bx1 = int(max(0, min(w - 1, x1)))
                by1 = int(max(0, min(h - 1, y1)))
                bx2 = int(max(bx1 + 10, min(w, x2)))
                by2 = int(max(by1 + 10, min(h, y2)))

                if cls_id == 0 and conf >= self.conf_thresh:  # COCO Person
                    detections.append(((bx1, by1, bx2, by2), conf))
                elif cls_id in acc_map:
                    acc_name = acc_map[cls_id]
                    if conf >= accessory_threshold(acc_name):
                        frame_accessories.append((acc_name, (bx1, by1, bx2, by2), conf))

        # Spatial tracking association
        active_list, archived_list = self._update_tracks(detections)
        return active_list, archived_list, frame_accessories

    def detect_accessories_in_roi(self, frame: np.ndarray, roi_bbox: Box,
                                  imgsz: Optional[int] = None) -> List[AccessoryDet]:
        """
        Upscaled accessory re-detection inside a person ROI with the (stronger, if available) accessory model.
        Returns [(canonical_name, bbox in frame coords, conf)] above per-class thresholds.
        """
        model = self.accessory_model if self.accessory_model is not None else self.model
        if model is None:
            return []
        cmap = self.accessory_class_map if model is self.accessory_model else self.class_map
        return detect_accessories_with_model(model, frame, roi_bbox, imgsz=imgsz or self.roi_imgsz,
                                             device=self.device, class_map=cmap or None)

    # ------------------------------------------------------------------ accessory association
    @staticmethod
    def associate_accessories(person_bbox: Tuple[int, int, int, int],
                              frame_accessories: List[Tuple[str, Tuple[int, int, int, int], float]]) -> List[str]:
        """
        Associate frame-level detected accessories with a person bounding box (legacy, non-exclusive).
        Prefer `assign_accessories`, which gives every accessory to at most one person.
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

    @staticmethod
    def assign_accessories(person_boxes: Dict[int, Box],
                           frame_accessories: List[AccessoryDet],
                           keypoints_by_track: Optional[Dict[int, np.ndarray]] = None,
                           min_affinity: float = 0.25) -> Dict[int, Dict[str, float]]:
        """
        Globally exclusive accessory -> person assignment.
        Every accessory box goes to at most ONE person: the one with the best geometric / keypoint affinity
        (wrists, hips, shoulders, head). Returns {track_id: {accessory_name: confidence}} for every track.
        """
        out: Dict[int, Dict[str, float]] = {tid: {} for tid in person_boxes}
        if not person_boxes or not frame_accessories:
            return out
        kpts = keypoints_by_track or {}
        pairs = []
        for a_idx, (name, abox, aconf) in enumerate(frame_accessories):
            key = normalize_accessory(name)
            for tid, pbox in person_boxes.items():
                aff = accessory_person_affinity(key, abox, pbox, kpts.get(tid))
                if aff >= min_affinity:
                    pairs.append((aff, float(aconf), a_idx, tid, key))
        # Highest affinity first (ties -> higher detector confidence)
        pairs.sort(key=lambda p: (p[0], p[1]), reverse=True)
        used = set()
        for aff, aconf, a_idx, tid, key in pairs:
            if a_idx in used:
                continue
            used.add(a_idx)
            conf = aconf * (0.5 + 0.5 * aff)
            out[tid][key] = round(max(out[tid].get(key, 0.0), conf), 3)
        return out

    # ------------------------------------------------------------------ tracking
    def _update_tracks(self, detections: List[Tuple[Tuple[int, int, int, int], float]]) -> Tuple[List[Tracklet], List[Tracklet]]:
        """
        Associate detections with existing tracks using a two-stage association strategy on
        constant-velocity predicted boxes:
        Stage 1: Global Maximum IoU Matching
        Stage 2: Spatial Center Proximity & Scale Consistency for fast-moving pedestrians
        """
        assigned_det = set()
        assigned_trk = set()
        matched_pairs = []

        track_ids = list(self.tracklets.keys())
        predicted = {tid: self.tracklets[tid].predicted_bbox() for tid in track_ids}

        # Stage 1: Global Maximum IoU Matching (Sort highest IoU first)
        iou_pairs = []
        for tid in track_ids:
            t_last = self.tracklets[tid].bbox
            t_pred = predicted[tid]
            for d_idx, (d_bbox, _) in enumerate(detections):
                iou = max(self._iou(t_pred, d_bbox), self._iou(t_last, d_bbox))
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
            tb = predicted[tid]
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

        # Update matched tracks (position + constant-velocity estimate)
        for tid, d_idx in matched_pairs:
            bbox, conf = detections[d_idx]
            trk = self.tracklets[tid]
            ob = trk.bbox
            steps = trk.lost_frames + 1
            vx = (((bbox[0] + bbox[2]) - (ob[0] + ob[2])) / 2.0) / steps
            vy = (((bbox[1] + bbox[3]) - (ob[1] + ob[3])) / 2.0) / steps
            vmax = 0.6 * max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1)
            vx, vy = float(np.clip(vx, -vmax, vmax)), float(np.clip(vy, -vmax, vmax))
            if trk.hits >= 2:
                vx = 0.6 * vx + 0.4 * trk.velocity[0]
                vy = 0.6 * vy + 0.4 * trk.velocity[1]
            trk.velocity = (vx, vy)
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
                # damp velocity while coasting so long occlusions do not drift far
                trk.velocity = (trk.velocity[0] * 0.9, trk.velocity[1] * 0.9)
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

        # Active tracks: current matches plus momentarily coasting tracks
        # (well-established tracks coast a little longer through short occlusions / missed detections)
        active_list = [
            t for t in self.tracklets.values()
            if (t.lost_frames == 0
                or (t.lost_frames <= 2 and t.hits >= 4)
                or (t.lost_frames <= 4 and t.hits >= 20))
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
