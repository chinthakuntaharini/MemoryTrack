"""
Forensic Attribute Extraction Engine
Extracts illumination-robust garment colors (CIELAB k-means on skin-free masks) and scored accessories
(frame detections, ROI re-detection with geometric validation, headwear / face-worn heuristics).
"""

import logging
import math
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterable, Union

from forensic_app.utils.color_naming import (
    bgr_to_cielab, classify_cielab_color, lab_to_lch
)
from forensic_app.pipeline.parser import (
    ParsedGarments, SkinModel, build_skin_model, face_geometry, normalize_keypoints,
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI,
    L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK,
)
from forensic_app.core.catalog import (
    normalize_accessory, accessory_threshold, sort_accessories, COCO_ACCESSORY_CLASSES
)

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]

UNKNOWN_HEX = "#808080"
BARE_LEGS_HEX = "#D4B996"
OCCLUDED_HEX = "#404040"

# ---------------------------------------------------------------------------------------------------
# Accessory <-> person geometry (shared with detector.assign_accessories)
# ---------------------------------------------------------------------------------------------------

# Keypoint anchors per accessory class: (keypoint indices, sigma as a fraction of person height)
_ACCESSORY_ANCHORS: Dict[str, List[Tuple[Tuple[int, ...], float]]] = {
    "handbag": [((L_WRI, R_WRI), 0.12), ((L_HIP, R_HIP), 0.12), ((L_ELB, R_ELB), 0.10), ((L_SHO, R_SHO), 0.10)],
    "suitcase": [((L_WRI, R_WRI), 0.15), ((L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK), 0.18)],
    "umbrella": [((L_WRI, R_WRI), 0.15), ((NOSE, L_EYE, R_EYE, L_EAR, R_EAR), 0.30)],
    "bottle": [((L_WRI, R_WRI), 0.08), ((NOSE,), 0.06)],
    "cell phone": [((L_WRI, R_WRI), 0.08), ((NOSE, L_EAR, R_EAR), 0.06)],
    "laptop": [((L_WRI, R_WRI), 0.12), ((L_HIP, R_HIP), 0.15)],
}

# Box-only location priors: (ry_min, ry_max, |rx|_max) in person-normalised coordinates
# (rx: 0 = person centre, +-0.5 = box edges; ry: 0 = top of box, 1 = bottom)
_LOCATION_PRIORS: Dict[str, Tuple[float, float, float]] = {
    "backpack": (0.08, 0.70, 0.60),
    "handbag": (0.22, 0.92, 0.85),
    "suitcase": (0.40, 1.10, 1.10),
    "umbrella": (-0.70, 0.75, 0.90),
    "tie": (0.10, 0.60, 0.22),
    "bottle": (0.05, 0.90, 0.85),
    "cell phone": (0.03, 0.90, 0.85),
    "laptop": (0.25, 0.90, 0.85),
}

_MAX_REL_AREA: Dict[str, float] = {"suitcase": 1.2, "umbrella": 2.5, "laptop": 0.8, "backpack": 0.7}


def _area(b) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def _inter(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def _point_box_distance(px: float, py: float, box) -> float:
    dx = max(float(box[0]) - px, 0.0, px - float(box[2]))
    dy = max(float(box[1]) - py, 0.0, py - float(box[3]))
    return math.hypot(dx, dy)


def _range_prior(v: float, lo: float, hi: float, fall: float = 0.15) -> float:
    if lo <= v <= hi:
        return 1.0
    d = (lo - v) if v < lo else (v - hi)
    return math.exp(-((d / fall) ** 2))


def _kp_arrays(keypoints) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if keypoints is None:
        return None, None
    kp = np.asarray(keypoints, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[0] < 17 or kp.shape[1] < 2:
        return None, None
    confs = kp[:17, 2] if kp.shape[1] >= 3 else np.ones(17, dtype=np.float32)
    return kp[:17, :2], confs


def accessory_person_affinity(name: str, acc_box: Box, person_box: Box,
                              keypoints: Optional[np.ndarray] = None, kp_conf: float = 0.30) -> float:
    """
    Geometric plausibility (0..1) that accessory `name` at `acc_box` belongs to the person at `person_box`.
    `keypoints` (optional) are FRAME coordinates (17x3). Combines envelope containment, a class-specific
    location prior and (when available) the distance to the joints that hold/wear that item.
    """
    name = normalize_accessory(name)
    px1, py1, px2, py2 = (float(v) for v in person_box)
    ax1, ay1, ax2, ay2 = (float(v) for v in acc_box)
    pw, ph = max(1.0, px2 - px1), max(1.0, py2 - py1)
    a_area = _area(acc_box)
    if a_area <= 0:
        return 0.0
    rel_area = a_area / (pw * ph)
    if rel_area > _MAX_REL_AREA.get(name, 0.6) or rel_area < 0.0015:
        return 0.0

    # 1. Envelope containment (carried items hang beside the body; umbrellas are above it)
    mx = (0.60 if name == "suitcase" else 0.30) * pw
    my_top = (0.70 if name == "umbrella" else 0.10) * ph
    env = (px1 - mx, py1 - my_top, px2 + mx, py2 + 0.10 * ph)
    ioa_env = _inter(acc_box, env) / a_area
    if ioa_env < 0.20:
        return 0.0
    ioa_person = _inter(acc_box, person_box) / a_area

    # 2. Location prior relative to the person box
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    rx = (acx - (px1 + px2) / 2.0) / pw
    ry = (acy - py1) / ph
    lo, hi, rx_max = _LOCATION_PRIORS.get(name, (0.0, 1.0, 0.7))
    prior = _range_prior(ry, lo, hi) * _range_prior(abs(rx), 0.0, rx_max)
    if name == "backpack":
        prior *= min(1.0, ioa_person / 0.5)
    elif name == "tie":
        prior *= min(1.0, ioa_person / 0.8)

    # 3. Keypoint proximity
    xy, confs = _kp_arrays(keypoints)
    prox = None
    if xy is not None:
        if name in ("backpack", "tie"):
            torso_idx = [i for i in (L_SHO, R_SHO, L_HIP, R_HIP) if confs[i] >= kp_conf]
            if len(torso_idx) >= 2:
                pts = xy[torso_idx]
                tx1, tx2 = float(pts[:, 0].min()), float(pts[:, 0].max())
                ty1, ty2 = float(pts[:, 1].min()), float(pts[:, 1].max())
                if ty2 - ty1 < 0.2 * ph:
                    ty2 = ty1 + 0.35 * ph
                tw = max(tx2 - tx1, 0.3 * pw)
                if name == "backpack":
                    torso = (tx1 - 0.35 * tw, ty1 - 0.10 * ph, tx2 + 0.35 * tw, ty2 + 0.05 * ph)
                else:
                    cx = (tx1 + tx2) / 2.0
                    torso = (cx - 0.25 * tw, ty1 - 0.05 * ph, cx + 0.25 * tw, ty1 + 0.6 * (ty2 - ty1))
                prox = _inter(acc_box, torso) / a_area
        else:
            best = 0.0
            for idxs, sigma in _ACCESSORY_ANCHORS.get(name, [((L_WRI, R_WRI), 0.12)]):
                for i in idxs:
                    if confs[i] >= kp_conf:
                        d = _point_box_distance(float(xy[i][0]), float(xy[i][1]), acc_box) / ph
                        best = max(best, math.exp(-((d / sigma) ** 2)))
            if any(confs[i] >= kp_conf for i in range(17)):
                prox = best

    if prox is None:
        score = 0.5 * ioa_env + 0.5 * prior
    else:
        score = 0.25 * ioa_env + 0.25 * prior + 0.50 * prox
    return float(np.clip(score, 0.0, 1.0))


def _noisy_or(a: float, b: float) -> float:
    return float(min(0.99, 1.0 - (1.0 - a) * (1.0 - b)))


class _RawModelAccessoryDetector:
    """Adapter giving a bare YOLO model the `detect_accessories_in_roi` interface (legacy call path)."""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    def detect_accessories_in_roi(self, frame: np.ndarray, roi_bbox: Box):
        from forensic_app.pipeline.detector import detect_accessories_with_model  # lazy: heavy import
        return detect_accessories_with_model(self.model, frame, roi_bbox, device=self.device)


class AttributeExtractor:
    """Extracts dominant CIELAB garment colors and carried / worn accessories."""

    def __init__(self,
                 k_clusters: int = 3,
                 min_luminance: float = 20.0,
                 max_luminance: float = 95.0,
                 max_color_samples: int = 2000,
                 roi_min_affinity: float = 0.35):
        self.k_clusters = k_clusters
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance
        self.max_color_samples = max_color_samples
        self.roi_min_affinity = roi_min_affinity

        # COCO accessory class IDs and names (kept for backward compatibility)
        self.coco_accessory_map = dict(COCO_ACCESSORY_CLASSES)

    # ------------------------------------------------------------------ public API
    def extract_attributes(self,
                           frame: np.ndarray,
                           person_bbox: Tuple[int, int, int, int],
                           parsed_garments: ParsedGarments,
                           yolo_model=None,
                           keypoints: Optional[np.ndarray] = None,
                           keypoints_crop: Optional[np.ndarray] = None,
                           initial_accessories: Optional[Union[List[str], Dict[str, float]]] = None
                           ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        """
        Extract upper garment color, lower garment color, and carried accessories.
        Returns: (upper_garment_dict, lower_garment_dict, accessories_list) — names in catalog order.
        """
        upper, lower, scored = self.extract_attributes_scored(
            frame, person_bbox, parsed_garments,
            accessory_detector=None, keypoints=keypoints, keypoints_crop=keypoints_crop,
            initial_accessories=initial_accessories, yolo_model=yolo_model,
        )
        return upper, lower, sort_accessories(scored.keys())

    def extract_attributes_scored(self,
                                  frame: np.ndarray,
                                  person_bbox: Tuple[int, int, int, int],
                                  parsed_garments: ParsedGarments,
                                  accessory_detector=None,
                                  keypoints: Optional[np.ndarray] = None,
                                  keypoints_crop: Optional[np.ndarray] = None,
                                  initial_accessories: Optional[Union[List[str], Dict[str, float]]] = None,
                                  yolo_model=None,
                                  exclude_boxes: Optional[Iterable[Box]] = None
                                  ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
        """
        Extract garment dicts and accessory confidences for THIS frame.
        - upper dict: {"color", "hex", "lab" ([L,a,b] or None), "confidence"}
        - lower dict: same + "type" (one of catalog.LOWER_GARMENT_TYPES); when no measurable pants,
          "color" equals the type string and "lab" is None.
        - accessories: canonical name -> confidence (0..1).
        `accessory_detector` (object with detect_accessories_in_roi) enables the ROI re-detection pass;
        `yolo_model` (legacy) is wrapped into one when no detector is given. `exclude_boxes` are other
        people's boxes: ROI detections that fit a neighbour better than this person are rejected.
        """
        upper_dict, lower_dict, skin_model = self._garment_colors(parsed_garments)

        accessories = self._score_accessories(
            frame, person_bbox, parsed_garments, skin_model, accessory_detector, yolo_model,
            keypoints, keypoints_crop, initial_accessories, exclude_boxes,
        )
        return upper_dict, lower_dict, accessories

    # ------------------------------------------------------------------ garment colors
    def _garment_colors(self, parsed: ParsedGarments) -> Tuple[Dict[str, Any], Dict[str, Any], SkinModel]:
        skin_lab = getattr(parsed, "skin_lab", None)
        skin_model = SkinModel(skin_lab, 2.0) if skin_lab is not None else SkinModel()

        # 1. Upper garment
        upper_dict: Dict[str, Any] = {"color": "Unknown", "hex": UNKNOWN_HEX, "lab": None, "confidence": 0.0}
        if parsed.has_upper_garment and parsed.upper_mask is not None:
            q = float(getattr(parsed, "upper_quality", 0.0) or 0.5)
            res = self._dominant_color(parsed.upper_crop, parsed.upper_mask, skin_model, q)
            if res is not None:
                upper_dict = res

        # 2. Lower garment (strict presence enforcement: never hallucinate pants on bare legs / occlusion)
        ltype = parsed.lower_garment_type
        lower_dict: Dict[str, Any] = {
            "color": ltype, "hex": BARE_LEGS_HEX if "Bare" in ltype else OCCLUDED_HEX,
            "lab": None, "confidence": round(float(getattr(parsed, "lower_quality", 0.0)), 3), "type": ltype,
        }
        if (parsed.has_lower_garment and parsed.lower_mask is not None
                and np.count_nonzero(parsed.lower_mask) > 20):
            q = float(getattr(parsed, "lower_quality", 0.0) or 0.5)
            res = self._dominant_color(parsed.lower_crop, parsed.lower_mask, skin_model, q)
            if res is not None:
                res["type"] = ltype
                lower_dict = res
        return upper_dict, lower_dict, skin_model

    def _extract_dominant_cielab_color(self, crop: np.ndarray, mask: np.ndarray) -> Tuple[str, str]:
        """Legacy helper: dominant color name and hex of the masked pixels."""
        res = self._dominant_color(crop, mask, SkinModel(), 0.5)
        if res is None:
            return "Unknown", UNKNOWN_HEX
        return res["color"], res["hex"]

    def _dominant_color(self, crop: np.ndarray, mask: np.ndarray, skin_model: SkinModel,
                        quality: float) -> Optional[Dict[str, Any]]:
        """
        Dominant garment color: subsample masked pixels, k-means (k=3, k-means++) in Lab, drop skin-like and
        extreme shadow/highlight clusters, merge shading variants of the same hue, keep the largest group.
        """
        if crop is None or crop.size == 0 or mask is None or crop.shape[:2] != mask.shape[:2]:
            return None
        pixels = crop[mask > 0]
        n = len(pixels)
        if n < 15 or pixels.ndim != 2 or pixels.shape[1] != 3:
            return None
        if n > self.max_color_samples:
            idx = np.linspace(0, n - 1, self.max_color_samples).astype(np.int64)
            pixels = pixels[idx]
        lab = bgr_to_cielab(np.ascontiguousarray(pixels))
        if len(lab) == 0:
            return None

        k = int(min(self.k_clusters, len(lab)))
        try:
            cv2.setRNGSeed(12345)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
            _, labels, centers = cv2.kmeans(lab.astype(np.float32), k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
            labels = labels.reshape(-1)
        except cv2.error:
            labels = np.zeros(len(lab), dtype=np.int32)
            centers = np.median(lab, axis=0, keepdims=True).astype(np.float32)
            k = 1
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        shares = counts / max(counts.sum(), 1.0)

        clusters = []
        for j in range(k):
            if shares[j] <= 0:
                continue
            c = centers[j]
            L, C, _ = lab_to_lch(c)
            clusters.append({
                "lab": c.astype(np.float64), "share": float(shares[j]), "members": [j],
                "skin": skin_model.is_skin_color(c),
                # only blown-out specular highlights are discarded; very dark clusters are real black fabric
                "extreme": L > 96.0 and C < 4.0,
            })

        # Merge shading variants of one fabric (same chroma/hue, different lightness)
        groups: List[Dict[str, Any]] = []
        for cl in sorted(clusters, key=lambda c: -c["share"]):
            for g in groups:
                if g["skin"] == cl["skin"] and self._same_fabric(g["lab"], cl["lab"]):
                    tot = g["share"] + cl["share"]
                    g["lab"] = (g["lab"] * g["share"] + cl["lab"] * cl["share"]) / tot
                    g["share"] = tot
                    g["members"] = g["members"] + cl["members"]
                    g["extreme"] = g["extreme"] and cl["extreme"]
                    break
            else:
                groups.append(dict(cl))

        keep = [g for g in groups if not g["skin"] and not g["extreme"]]
        penalty = 1.0
        if not keep:
            keep = [g for g in groups if not g["skin"]] or groups   # all highlight / all skin-like
            penalty = 0.6 if all(g["skin"] for g in keep) else 0.85
        best = max(keep, key=lambda g: g["share"])

        # Representative color: median chroma of the group, lightness from its lit side (65th percentile)
        # so folds and self-shadow do not darken the reported garment color.
        member = np.isin(labels, best["members"])
        lab_c = best["lab"].copy()
        if np.count_nonzero(member) >= 10:
            mp = lab[member]
            lab_c = np.array([np.percentile(mp[:, 0], 65), np.median(mp[:, 1]), np.median(mp[:, 2])])
        name, hex_code = classify_cielab_color(lab_c)
        conf = float(np.clip(quality, 0.0, 1.0)) * float(np.clip(0.4 + best["share"], 0.0, 1.0)) * penalty
        return {
            "color": name,
            "hex": hex_code,
            "lab": [round(float(v), 2) for v in lab_c],
            "confidence": round(float(np.clip(conf, 0.0, 1.0)), 3),
        }

    @staticmethod
    def _same_fabric(lab1: np.ndarray, lab2: np.ndarray) -> bool:
        """True when two cluster centers look like lit / shaded parts of the same fabric."""
        d = lab1 - lab2
        if math.sqrt((0.3 * d[0]) ** 2 + d[1] ** 2 + d[2] ** 2) < 9.0:
            return True
        L1, C1, h1 = lab_to_lch(lab1)
        L2, C2, h2 = lab_to_lch(lab2)
        if min(C1, C2) < 12.0 or abs(L1 - L2) > 50.0:
            return False
        dh = abs(h1 - h2) % 360.0
        dh = min(dh, 360.0 - dh)
        return dh < 18.0 and 0.5 <= C1 / max(C2, 1e-6) <= 2.0

    # ------------------------------------------------------------------ accessories
    def _score_accessories(self, frame, person_bbox, parsed, skin_model, accessory_detector, yolo_model,
                           keypoints, keypoints_crop, initial_accessories, exclude_boxes) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        # (a) Frame-level assignments from the detector
        if initial_accessories:
            if isinstance(initial_accessories, dict):
                items = initial_accessories.items()
            else:
                items = ((a, 0.5) for a in initial_accessories)
            for acc, conf in items:
                key = normalize_accessory(str(acc))
                if key:
                    scores[key] = max(scores.get(key, 0.0), float(np.clip(conf, 0.0, 1.0)))

        if frame is None or getattr(frame, "ndim", 0) != 3:
            return scores
        px1, py1, px2, py2 = (int(v) for v in person_bbox)
        pw, ph = px2 - px1, py2 - py1
        if pw < 10 or ph < 10:
            return scores
        fh, fw = frame.shape[:2]

        # Frame-coordinate keypoints for geometric validation
        kpts_frame = None
        if keypoints is not None:
            kpts_frame = np.asarray(keypoints, dtype=np.float32)
        elif keypoints_crop is not None:
            kc = np.asarray(keypoints_crop, dtype=np.float32)
            if kc.ndim == 2 and kc.shape[0] >= 17:
                kpts_frame = kc.copy()
                kpts_frame[:, 0] += px1
                kpts_frame[:, 1] += py1

        # (b) ROI re-detection pass (throttled by the caller: only when a detector is given)
        detector = accessory_detector
        if detector is None and yolo_model is not None:
            detector = _RawModelAccessoryDetector(yolo_model)
        if detector is not None:
            roi = (max(0, px1 - int(0.25 * pw)), max(0, py1 - int(0.15 * ph)),
                   min(fw, px2 + int(0.25 * pw)), min(fh, py2 + int(0.08 * ph)))
            try:
                dets = detector.detect_accessories_in_roi(frame, roi) or []
            except Exception as e:
                logger.debug(f"ROI accessory detection failed: {e}")
                dets = []
            others = [tuple(b) for b in (exclude_boxes or [])]
            roi_scores: Dict[str, float] = {}
            for acc_name, abox, aconf in dets:
                key = normalize_accessory(acc_name)
                if float(aconf) < accessory_threshold(key):
                    continue
                aff = accessory_person_affinity(key, abox, (px1, py1, px2, py2), kpts_frame)
                if aff < self.roi_min_affinity:
                    continue
                if others:
                    own_box_aff = accessory_person_affinity(key, abox, (px1, py1, px2, py2), None)
                    other_aff = max(accessory_person_affinity(key, abox, ob, None) for ob in others)
                    if other_aff > own_box_aff + 0.05:
                        continue
                s = float(aconf) * (0.55 + 0.45 * aff)
                roi_scores[key] = max(roi_scores.get(key, 0.0), s)
            for key, s in roi_scores.items():
                scores[key] = _noisy_or(scores[key], s) if key in scores else s

        # (c) Face-worn items: cap / hat, face mask (keypoint-gated heuristics)
        kc = keypoints_crop
        if kc is None and kpts_frame is not None:
            kc = kpts_frame.copy()
            kc[:, 0] -= px1
            kc[:, 1] -= py1
        if kc is not None:
            crop = frame[max(0, py1):min(fh, py2), max(0, px1):min(fw, px2)]
            try:
                face_items = self._face_accessories(crop, kc, getattr(parsed, "skin_lab", None))
            except Exception as e:
                logger.debug(f"Face accessory heuristic failed: {e}")
                face_items = {}
            for key, s in face_items.items():
                if s >= accessory_threshold(key):
                    scores[key] = max(scores.get(key, 0.0), s)

        return {k: round(float(v), 3) for k, v in scores.items()}

    def _detect_accessories(self, frame, person_bbox, yolo_model=None, keypoints=None, keypoints_crop=None,
                            initial_accessories=None) -> List[str]:
        """Legacy helper: accessory names (catalog order)."""
        scored = self._score_accessories(frame, person_bbox, None, SkinModel(), None, yolo_model,
                                         keypoints, keypoints_crop, initial_accessories, None)
        return sort_accessories(scored.keys())

    # ------------------------------------------------------------------ face-worn heuristics
    @staticmethod
    def _region(img: np.ndarray, x1: float, y1: float, x2: float, y2: float, min_inside: float = 0.6):
        """Clip a float rectangle to the image; None when less than `min_inside` of it is inside."""
        h, w = img.shape[:2]
        area = max(1e-6, (x2 - x1) * (y2 - y1))
        cx1, cy1 = int(max(0, math.floor(x1))), int(max(0, math.floor(y1)))
        cx2, cy2 = int(min(w, math.ceil(x2))), int(min(h, math.ceil(y2)))
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            return None
        if (cx2 - cx1) * (cy2 - cy1) < min_inside * area:
            return None
        return img[cy1:cy2, cx1:cx2]

    def _face_accessories(self, crop: np.ndarray, keypoints_crop: np.ndarray,
                          skin_lab: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Cap / hat and face mask from face-keypoint geometry.
        Requires a confident frontal face (nose + both eyes). Low-chroma crowns are treated as hair unless
        there is clear brim evidence, so bare dark (or grey) hair never produces a "cap". Sunglasses are not
        inferred: an eye-band darkness test proved unreliable on real faces (shadowed eye sockets).
        """
        out: Dict[str, float] = {}
        if crop is None or crop.size == 0 or crop.ndim != 3:
            return out
        h, w = crop.shape[:2]
        xy, confs = normalize_keypoints(keypoints_crop, None, h, w)
        geo = face_geometry(xy, confs)
        if geo is None or not geo["frontal"] or geo["roll"] > 25.0:
            return out
        d, cx, ey, ny = geo["d"], geo["cx"], geo["ey"], geo["nose_y"]
        if d < 6.0 or not (0.2 * d <= ny - ey <= 1.6 * d):
            return out

        lab_img = bgr_to_cielab(crop)
        ycc_img = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        skin = SkinModel(skin_lab, 2.0) if skin_lab is not None else build_skin_model(lab_img, ycc_img, xy, confs)
        generic = SkinModel()

        def stats(x1, y1, x2, y2, model: SkinModel = skin, min_inside: float = 0.6):
            lab = self._region(lab_img, x1, y1, x2, y2, min_inside)
            if lab is None:
                return None
            ycc = self._region(ycc_img, x1, y1, x2, y2, min_inside)
            sk = model.classify(lab, ycc)
            flat = lab.reshape(-1, 3)
            non = flat[~sk.reshape(-1)]
            med = np.median(non if len(non) >= 4 else flat, axis=0)
            return {"skin": float(np.mean(sk)), "lab": med, "std_L": float(np.std(flat[:, 0])),
                    "lab_px": lab}

        forehead = stats(cx - 0.5 * d, ey - 0.85 * d, cx + 0.5 * d, ey - 0.35 * d)

        # ---- cap / hat
        crown = stats(cx - 0.6 * d, ey - 1.6 * d, cx + 0.6 * d, ey - 1.0 * d)
        if crown is not None and crown["skin"] < 0.45:
            L, C, hue = lab_to_lch(crown["lab"])
            fh_skin = forehead["skin"] if forehead is not None else 0.5
            uniform = crown["std_L"] < 12.0
            # Hair: any low-chroma crown (black / brown / grey / white hair) or warm desaturated tones
            # (brown / blonde hair). Such crowns are never called a cap without strong brim evidence.
            neutral = C < 16.0 and L < 82.0
            warm_hair = 30.0 <= hue <= 95.0 and C < 36.0 and L < 80.0
            score = 0.0
            if not neutral and not warm_hair and C >= 20.0 and crown["skin"] < 0.25:
                score = 0.55 + 0.25 * (1.0 - fh_skin) + (0.1 if uniform else 0.0)        # coloured cap
            elif L >= 82.0 and C < 15.0 and crown["skin"] < 0.25 and uniform:
                score = 0.52 + 0.25 * (1.0 - fh_skin)                                  # white cap
            elif neutral and fh_skin < 0.15 and crown["std_L"] < 8.0 and self._brim_evidence(crop, cx, ey, d):
                score = 0.52                                                           # dark cap with brim
            logger.debug(f"headwear: crown L={L:.1f} C={C:.1f} h={hue:.0f} skin={crown['skin']:.2f} "
                         f"stdL={crown['std_L']:.1f} forehead_skin={fh_skin:.2f} d={d:.1f} -> {score:.2f}")
            if score > 0.0:
                out["cap"] = float(min(0.95, score))

        # ---- face mask (nose/mouth covered while the forehead shows skin)
        if d >= 10.0:
            fg = stats(cx - 0.5 * d, ey - 0.85 * d, cx + 0.5 * d, ey - 0.35 * d, model=generic)
            nm = stats(cx - 0.55 * d, ny - 0.15 * d, cx + 0.55 * d, ny + 0.9 * d, model=generic)
            if fg is not None and nm is not None and fg["skin"] >= 0.4 and nm["skin"] < 0.10 and nm["std_L"] < 12.0:
                L, C, hue = lab_to_lch(nm["lab"])
                mask_color = ((L >= 70.0 and C < 18.0) or (180.0 <= hue <= 300.0 and C >= 10.0)
                              or (L < 30.0 and C < 10.0))
                if mask_color:
                    out["face mask"] = 0.6

        return out

    @staticmethod
    def _brim_evidence(crop: np.ndarray, cx: float, ey: float, d: float) -> bool:
        """Strong, wide horizontal edge across the head at brim height (between eyebrows and mid forehead)."""
        h, w = crop.shape[:2]
        x1, x2 = int(max(0, cx - 0.9 * d)), int(min(w, cx + 0.9 * d))
        y1, y2 = int(max(0, ey - 1.3 * d)), int(min(h, ey - 0.1 * d))
        if x2 - x1 < 6 or y2 - y1 < 6:
            return False
        gray = cv2.cvtColor(crop[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        prof = gy.mean(axis=1)
        if len(prof) < 3:
            return False
        r = int(np.argmax(prof))
        peak = float(prof[r])
        base = float(np.median(prof))
        pos = (y1 + r) - ey  # negative: above the eyes
        if not (-0.7 * d <= pos <= -0.2 * d):
            return False
        row = gy[max(0, r - 1):r + 2].max(axis=0)
        coverage = float(np.mean(row > 0.5 * peak))
        return peak >= 40.0 and peak >= 3.0 * (base + 1.0) and coverage >= 0.7
