"""
Human Parsing & Garment Segmentation Engine
Builds pose-guided garment masks (torso polygon, leg bands), removes skin with an adaptive face-sampled skin
model, and prevents pants color hallucination when legs are bare, occluded, or out of frame.
"""

import logging
import math
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from forensic_app.utils.color_naming import bgr_to_cielab, lab_to_bgr

logger = logging.getLogger(__name__)

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

KP_CONF = 0.30       # minimum confidence for a joint used in mask geometry
FACE_CONF = 0.50     # minimum confidence for face joints used for skin sampling / face heuristics


@dataclass
class ParsedGarments:
    """Masked crops and presence indicators for upper and lower garments."""
    upper_mask: np.ndarray
    lower_mask: np.ndarray
    upper_crop: np.ndarray
    lower_crop: np.ndarray
    has_upper_garment: bool
    has_lower_garment: bool
    lower_garment_type: str  # 'Pants/Jeans', 'Bare Legs / Shorts', 'None / Occluded'
    upper_quality: float = 0.0           # 0..1 reliability of the upper mask (geometry, coverage, skin purity)
    lower_quality: float = 0.0           # 0..1 reliability of the lower decision / mask
    skin_lab: Optional[np.ndarray] = None  # (3,) Lab of the person's face skin when it could be sampled


class SkinModel:
    """
    Pixel skin classifier.
    Adaptive mode compares pixels to the person's own face skin (sampled around nose/eyes); the generic
    fallback uses strict YCrCb + Lab ranges that reject khaki, tan, orange and dark brown fabrics.
    """

    def __init__(self, ref_lab: Optional[np.ndarray] = None, spread: float = 0.0):
        self.ref_lab = None if ref_lab is None else np.asarray(ref_lab, dtype=np.float32).reshape(3)
        self.spread = float(spread)
        # Lab distance threshold (lightness down-weighted: legs and face are lit differently)
        self.threshold = float(np.clip(5.0 + 2.0 * self.spread, 7.0, 11.0))

    @property
    def adaptive(self) -> bool:
        """True when the model is anchored on the person's sampled face skin."""
        return self.ref_lab is not None

    def classify(self, lab: np.ndarray, ycrcb: np.ndarray) -> np.ndarray:
        """Boolean skin mask for Lab (float, L 0..100) and YCrCb (uint8) arrays of identical leading shape."""
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        cr = ycrcb[..., 1].astype(np.int16)
        cb = ycrcb[..., 2].astype(np.int16)
        if self.adaptive:
            Lr, ar, br = self.ref_lab
            # Lightness: tolerant to moderate shading (+-12), strict beyond (dark brown fabric vs. skin)
            dl = np.abs(L - Lr)
            dl_eff = 0.35 * np.minimum(dl, 12.0) + 0.8 * np.maximum(dl - 12.0, 0.0)
            d = np.sqrt(dl_eff ** 2 + (a - ar) ** 2 + (b - br) ** 2)
            loose = (cr >= 128) & (cr <= 185) & (cb >= 65) & (cb <= 140)
            light_ok = (L >= max(12.0, Lr - 40.0)) & (L <= min(97.0, Lr + 25.0))
            return (d < self.threshold) & loose & light_ok
        # Strict generic rule (no face reference available)
        ycc = (cr >= 138) & (cr <= 170) & (cb >= 85) & (cb <= 122)
        lab_ok = ((L >= 32.0) & (L <= 88.0) & (a >= 9.0) & (a <= 30.0) & (b >= 8.0) & (b <= 35.0)
                  & (a >= 0.33 * b) & (a <= 1.4 * b + 5.0))
        return ycc & lab_ok

    def is_skin_color(self, lab_val) -> bool:
        """Test a single Lab color (e.g. a k-means cluster center)."""
        bgr = np.array(lab_to_bgr(lab_val), dtype=np.uint8).reshape(1, 1, 3)
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        lab = np.asarray(lab_val, dtype=np.float32).reshape(1, 1, 3)
        return bool(self.classify(lab, ycc)[0, 0])


def _conf_ok(confs: np.ndarray, idx: int, thr: float = KP_CONF) -> bool:
    return confs is not None and idx < len(confs) and float(confs[idx]) >= thr


def _normalize_keypoints(keypoints: Optional[np.ndarray], keypoint_confs: Optional[np.ndarray],
                         h: int, w: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (17x2 pixel xy, 17 confs) or (None, None). Accepts pixel or normalised (0..1) coordinates."""
    if keypoints is None:
        return None, None
    kp = np.asarray(keypoints, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[0] < 17 or kp.shape[1] < 2:
        return None, None
    xy = kp[:17, :2].copy()
    if keypoint_confs is not None:
        confs = np.asarray(keypoint_confs, dtype=np.float32).reshape(-1)[:17]
        if len(confs) < 17:
            return None, None
    elif kp.shape[1] >= 3:
        confs = kp[:17, 2].copy()
    else:
        confs = np.ones(17, dtype=np.float32)
    confs = np.nan_to_num(confs.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.isfinite(xy).all(axis=1)
    confs[~finite] = 0.0
    xy[~finite] = 0.0
    valid = confs >= KP_CONF
    if np.any(valid) and float(np.max(np.abs(xy[valid]))) <= 1.5 and max(h, w) > 20:
        xy[:, 0] *= w
        xy[:, 1] *= h
    # Joints far outside the crop are pose errors (or belong to someone else): ignore them
    far = ((xy[:, 0] < -0.25 * w) | (xy[:, 0] > 1.25 * w) | (xy[:, 1] < -0.25 * h) | (xy[:, 1] > 1.25 * h))
    confs[far] = 0.0
    return xy, confs


def _face_geometry(xy: np.ndarray, confs: np.ndarray) -> Optional[Dict[str, float]]:
    """Eye-line geometry of a (roughly frontal) face: centre x, eye y, inter-eye distance d, nose y."""
    if xy is None or not _conf_ok(confs, NOSE, FACE_CONF):
        return None
    le_ok, re_ok = _conf_ok(confs, L_EYE, FACE_CONF), _conf_ok(confs, R_EYE, FACE_CONF)
    nose = xy[NOSE]
    if le_ok and re_ok:
        le, re = xy[L_EYE], xy[R_EYE]
        d = float(np.hypot(*(le - re)))
        cx, ey = float((le[0] + re[0]) / 2.0), float((le[1] + re[1]) / 2.0)
        roll = math.degrees(math.atan2(abs(le[1] - re[1]), abs(le[0] - re[0]) + 1e-6))
        frontal = True
    elif le_ok or re_ok:
        eye = xy[L_EYE] if le_ok else xy[R_EYE]
        d = float(2.0 * abs(eye[0] - nose[0]) + 0.5 * abs(eye[1] - nose[1]))
        cx, ey = float((eye[0] + nose[0]) / 2.0), float(eye[1])
        roll = 0.0
        frontal = False
    else:
        return None
    if d < 3.0:
        return None
    return {"cx": cx, "ey": ey, "d": d, "nose_x": float(nose[0]), "nose_y": float(nose[1]),
            "roll": roll, "frontal": frontal}


def sample_face_skin(lab_img: np.ndarray, ycrcb_img: np.ndarray, xy: Optional[np.ndarray],
                     confs: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, float]]:
    """
    Sample the person's skin color from the cheeks/nose patch below the eyes.
    Returns (median Lab, chroma spread) or None when the face is not visible or the sample is not skin-like.
    """
    geo = _face_geometry(xy, confs)
    if geo is None:
        return None
    h, w = lab_img.shape[:2]
    d = geo["d"]
    y1 = geo["ey"] + 0.20 * d
    y2 = max(geo["nose_y"] + 0.25 * d, y1 + 3.0)
    x1, x2 = geo["cx"] - 0.60 * d, geo["cx"] + 0.60 * d
    x1, x2 = int(max(0, math.floor(x1))), int(min(w, math.ceil(x2)))
    y1, y2 = int(max(0, math.floor(y1))), int(min(h, math.ceil(y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    lab = lab_img[y1:y2, x1:x2].reshape(-1, 3)
    ycc = ycrcb_img[y1:y2, x1:x2].reshape(-1, 3).astype(np.int16)
    loose = ((ycc[:, 1] >= 128) & (ycc[:, 1] <= 185) & (ycc[:, 2] >= 65) & (ycc[:, 2] <= 140)
             & (lab[:, 0] >= 15.0) & (lab[:, 0] <= 95.0))
    pix = lab[loose]
    if len(pix) < 10:
        return None
    ref = np.median(pix, axis=0).astype(np.float32)
    L, a, b = float(ref[0]), float(ref[1]), float(ref[2])
    chroma = math.hypot(a, b)
    hue = math.degrees(math.atan2(b, a)) % 360.0
    # Sanity: must look like human skin (rejects masks, glasses, hair, backgrounds)
    if not (15.0 <= L <= 92.0 and 2.0 <= a <= 35.0 and 2.0 <= b <= 45.0 and 6.0 <= chroma <= 50.0
            and 15.0 <= hue <= 90.0):
        return None
    dev = np.sqrt((pix[:, 1] - a) ** 2 + (pix[:, 2] - b) ** 2)
    spread = float(np.median(dev))
    return ref, spread


def build_skin_model(lab_img: np.ndarray, ycrcb_img: np.ndarray, xy: Optional[np.ndarray],
                     confs: Optional[np.ndarray], skin_lab: Optional[np.ndarray] = None) -> SkinModel:
    """Adaptive skin model from the face when possible (or a supplied reference), else the strict generic one."""
    if skin_lab is not None:
        return SkinModel(skin_lab, 2.0)
    sample = sample_face_skin(lab_img, ycrcb_img, xy, confs)
    if sample is None:
        return SkinModel()
    return SkinModel(sample[0], sample[1])


# Public names for helpers shared with the attribute extractor
normalize_keypoints = _normalize_keypoints
face_geometry = _face_geometry


def _scale_polygon(poly: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Shrink a polygon about its centroid (sx, sy < 1 move vertices inward)."""
    c = poly.mean(axis=0)
    out = poly.copy()
    out[:, 0] = c[0] + (poly[:, 0] - c[0]) * sx
    out[:, 1] = c[1] + (poly[:, 1] - c[1]) * sy
    return out


class HumanParser:
    """Pose-guided cloth and body segmenter."""

    def __init__(self, min_mask_pixels: int = 40):
        self.min_mask_pixels = min_mask_pixels

    def parse_pedestrian(self,
                         crop: np.ndarray,
                         keypoints: Optional[np.ndarray] = None,
                         keypoint_confs: Optional[np.ndarray] = None) -> ParsedGarments:
        """
        Segment upper and lower garments from a pedestrian crop.
        `keypoints` are crop-relative COCO-17 (17x3 with confidences, or 17x2 + `keypoint_confs`).
        Guarantees NO false pants color if legs are bare, occluded, or out of frame.
        """
        if crop is None or crop.ndim != 3 or crop.shape[2] != 3:
            crop = np.zeros((0, 0, 3), dtype=np.uint8)
        h, w = crop.shape[:2]
        empty_mask = np.zeros((h, w), dtype=np.uint8)

        if h < 20 or w < 20:
            return ParsedGarments(
                upper_mask=empty_mask, lower_mask=empty_mask,
                upper_crop=np.empty((0, 0, 3), dtype=np.uint8),
                lower_crop=np.empty((0, 0, 3), dtype=np.uint8),
                has_upper_garment=False, has_lower_garment=False,
                lower_garment_type="None / Occluded"
            )

        crop = np.ascontiguousarray(crop, dtype=np.uint8)
        xy, confs = _normalize_keypoints(keypoints, keypoint_confs, h, w)

        # 1. Color spaces + skin model (adaptive from the face when visible)
        lab_img = bgr_to_cielab(crop)
        ycrcb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        skin_model = build_skin_model(lab_img, ycrcb_img, xy, confs)
        skin_map = skin_model.classify(lab_img, ycrcb_img)

        # 2. Upper garment mask (torso polygon minus arms/head minus skin)
        upper_mask, upper_quality = self._build_upper(xy, confs, h, w, skin_map)

        # 3. Lower garment (leg bands, bare-leg decision)
        lower_mask, lower_type, has_lower_garment, lower_quality = self._build_lower(
            crop, xy, confs, h, w, skin_map, skin_model
        )

        # Extract masked crops
        upper_crop = cv2.bitwise_and(crop, crop, mask=upper_mask)
        lower_crop = (cv2.bitwise_and(crop, crop, mask=lower_mask) if has_lower_garment
                      else np.empty((0, 0, 3), dtype=np.uint8))

        has_upper_garment = int(np.count_nonzero(upper_mask)) >= self.min_mask_pixels
        if not has_upper_garment:
            upper_quality = 0.0

        return ParsedGarments(
            upper_mask=upper_mask,
            lower_mask=lower_mask,
            upper_crop=upper_crop,
            lower_crop=lower_crop,
            has_upper_garment=has_upper_garment,
            has_lower_garment=has_lower_garment,
            lower_garment_type=lower_type,
            upper_quality=round(float(upper_quality), 3),
            lower_quality=round(float(lower_quality), 3),
            skin_lab=skin_model.ref_lab.copy() if skin_model.adaptive else None,
        )

    # ------------------------------------------------------------------ upper body
    def _build_upper(self, xy: Optional[np.ndarray], confs: Optional[np.ndarray], h: int, w: int,
                     skin_map: np.ndarray) -> Tuple[np.ndarray, float]:
        """Torso polygon (or geometric fallback) with arm/head exclusion and skin removal."""
        mask = np.zeros((h, w), dtype=np.uint8)
        poly, geom_q, pose_based, shoulder_w = self._torso_polygon(xy, confs, h, w)

        if poly is not None:
            cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
            excl = np.zeros_like(mask)
            # Arms (sleeves / bare arms crossing the torso)
            thick = int(max(3, round(0.22 * shoulder_w)))
            for a_idx, b_idx in ((L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB), (R_ELB, R_WRI)):
                if _conf_ok(confs, a_idx) and _conf_ok(confs, b_idx):
                    p1 = tuple(int(round(v)) for v in xy[a_idx])
                    p2 = tuple(int(round(v)) for v in xy[b_idx])
                    cv2.line(excl, p1, p2, 255, thickness=thick)
            # Head (when bending forward the face can drop below the shoulder line)
            if _conf_ok(confs, NOSE):
                r = int(max(4, round(0.30 * shoulder_w)))
                cv2.circle(excl, tuple(int(round(v)) for v in xy[NOSE]), r, 255, thickness=-1)
            mask[excl > 0] = 0
            if int(np.count_nonzero(mask)) < self.min_mask_pixels:
                poly, pose_based = None, False   # degenerate pose geometry -> geometric fallback

        if poly is None:
            # No usable keypoints: geometric rectangle chosen by crop shape
            mask[:] = 0
            aspect = float(h) / max(w, 1)
            if aspect >= 1.7:      # full body
                y1, y2, x1, x2 = 0.18, 0.48, 0.25, 0.75
            elif aspect < 1.25:    # head + shoulders (webcam / close-up): face sits in the top half
                y1, y2, x1, x2 = 0.58, 0.97, 0.15, 0.85
            else:                  # upper body / partial
                y1, y2, x1, x2 = 0.38, 0.72, 0.22, 0.78
            mask[int(h * y1):int(h * y2), int(w * x1):int(w * x2)] = 255
            geom_q = 0.55

        n_total = int(np.count_nonzero(mask))
        if n_total == 0:
            return mask, 0.0

        # Skin removal (necklines, bare arms, faces in fallback rectangles)
        skin_in = skin_map & (mask > 0)
        n_skin = int(np.count_nonzero(skin_in))
        skin_frac = n_skin / float(n_total)
        if pose_based and skin_frac > 0.75:
            # A reliable torso that is almost entirely "skin": skin-coloured garment (or shirtless) -> keep
            geom_q *= 0.5
        else:
            mask[skin_in] = 0
        n_kept = int(np.count_nonzero(mask))

        coverage_q = math.sqrt(min(1.0, n_kept / 300.0))
        quality = geom_q * coverage_q * (1.0 - 0.5 * min(skin_frac, 1.0))
        return mask, float(np.clip(quality, 0.0, 1.0))

    def _torso_polygon(self, xy: Optional[np.ndarray], confs: Optional[np.ndarray], h: int, w: int
                       ) -> Tuple[Optional[np.ndarray], float, bool, float]:
        """Return (polygon Nx2, geometric quality, pose_based, shoulder width) or (None, 0, False, est.)."""
        if xy is None:
            return None, 0.0, False, 0.3 * w
        ls_ok, rs_ok = _conf_ok(confs, L_SHO), _conf_ok(confs, R_SHO)
        lh_ok, rh_ok = _conf_ok(confs, L_HIP), _conf_ok(confs, R_HIP)

        if ls_ok and rs_ok:
            ls, rs = xy[L_SHO], xy[R_SHO]
            sho_c = (ls + rs) / 2.0
            sw = float(abs(ls[0] - rs[0]))
            half_top = max(sw / 2.0, 0.14 * w)          # side views: shoulders overlap -> keep a minimum width
            top = [np.array([sho_c[0] - half_top, sho_c[1]]), np.array([sho_c[0] + half_top, sho_c[1]])]
            sw_eff = 2.0 * half_top
            conf_q = float(np.clip(np.mean([confs[L_SHO], confs[R_SHO]]), 0.5, 1.0))

            hip_pts = [xy[i] for i, ok in ((L_HIP, lh_ok), (R_HIP, rh_ok)) if ok]
            if hip_pts:
                hip_y = float(np.mean([p[1] for p in hip_pts]))
            else:
                hip_y = None

            if hip_y is not None and hip_y > sho_c[1] + max(8.0, 0.3 * sw_eff):
                # Full torso quad: shoulders -> hips
                if lh_ok and rh_ok:
                    hip_c = (xy[L_HIP] + xy[R_HIP]) / 2.0
                    half_bot = max(abs(xy[L_HIP][0] - xy[R_HIP][0]) / 2.0, 0.35 * sw_eff)
                else:
                    hip_c = np.array([sho_c[0], hip_y])
                    half_bot = 0.40 * sw_eff
                bottom = [np.array([hip_c[0] - half_bot, hip_y]), np.array([hip_c[0] + half_bot, hip_y])]
                poly = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
                poly = _scale_polygon(poly, 0.80, 0.86)
                q = 1.0 * conf_q
                if not (lh_ok and rh_ok):
                    q *= 0.9
                return poly, q, True, sw_eff

            # Upper-body-only view (webcam / desk): hips not visible -> torso = 1.6 x shoulder width
            torso_len = 1.6 * sw_eff
            bot_y = min(float(h - 1), sho_c[1] + torso_len)
            if bot_y - sho_c[1] < 6.0:
                return None, 0.0, False, sw_eff
            half_bot = 0.9 * half_top
            bottom = [np.array([sho_c[0] - half_bot, bot_y]), np.array([sho_c[0] + half_bot, bot_y])]
            poly = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
            poly = _scale_polygon(poly, 0.80, 0.88)
            return poly, 0.85 * conf_q, True, sw_eff

        if ls_ok or rs_ok:
            # Profile view: one shoulder; the torso is centred around it
            s = xy[L_SHO] if ls_ok else xy[R_SHO]
            hip = xy[L_HIP] if lh_ok else (xy[R_HIP] if rh_ok else None)
            half = 0.16 * w
            y1 = float(s[1]) + 0.04 * h
            if hip is not None and hip[1] > s[1] + 10:
                y2 = float(hip[1]) - 0.02 * h
                cx = float(s[0] + hip[0]) / 2.0
            else:
                y2 = min(float(h - 1), float(s[1]) + 0.40 * h)
                cx = float(s[0])
            if y2 - y1 < 6.0:
                return None, 0.0, False, 2.0 * half
            poly = np.array([[cx - half, y1], [cx + half, y1], [cx + half, y2], [cx - half, y2]],
                            dtype=np.float32)
            return poly, 0.70, True, 2.0 * half

        return None, 0.0, False, 0.3 * w

    # ------------------------------------------------------------------ lower body
    def _build_lower(self, crop: np.ndarray, xy: Optional[np.ndarray], confs: Optional[np.ndarray],
                     h: int, w: int, skin_map: np.ndarray, skin_model: SkinModel
                     ) -> Tuple[np.ndarray, str, bool, float]:
        """Return (lower_mask, lower_garment_type, has_lower_garment, lower_quality)."""
        lower_mask = np.zeros((h, w), dtype=np.uint8)
        thigh_mask = shin_mask = None
        geom_q = 0.0
        rect_mode = False

        if xy is not None:
            conf_hips = max(float(confs[L_HIP]), float(confs[R_HIP]))
            conf_knees = max(float(confs[L_KNE]), float(confs[R_KNE]))
            conf_ankles = max(float(confs[L_ANK]), float(confs[R_ANK]))

            # Knees AND ankles missing -> occluded / upper-body only
            if conf_knees < 0.20 and conf_ankles < 0.20:
                return lower_mask, "None / Occluded", False, 0.0
            if conf_hips < KP_CONF:
                return lower_mask, "None / Occluded", False, 0.0

            thigh_mask, shin_mask, geom_q = self._leg_bands(xy, confs, h, w)
            if thigh_mask is None:
                return lower_mask, "None / Occluded", False, 0.0
            lower_mask = cv2.bitwise_or(thigh_mask, shin_mask)
        else:
            # Without keypoints, only inspect the lower half when the crop looks like a full body
            aspect = float(h) / max(w, 1)
            if aspect < 1.70:
                return lower_mask, "None / Occluded", False, 0.0
            lower_mask[int(h * 0.54):int(h * 0.94), int(w * 0.20):int(w * 0.80)] = 255
            rect_mode = True
            geom_q = 0.5

        n_lower = int(np.count_nonzero(lower_mask))
        if n_lower < self.min_mask_pixels:
            return np.zeros((h, w), dtype=np.uint8), "None / Occluded", False, 0.0

        # Skin ratios (whole lower region, thighs, shins)
        skin_lower = skin_map & (lower_mask > 0)
        skin_ratio = np.count_nonzero(skin_lower) / float(n_lower)
        thigh_ratio = shin_ratio = 0.0
        n_shin = 0
        if thigh_mask is not None:
            n_thigh = int(np.count_nonzero(thigh_mask))
            n_shin = int(np.count_nonzero(shin_mask))
            if n_thigh > 0:
                thigh_ratio = np.count_nonzero(skin_map & (thigh_mask > 0)) / float(n_thigh)
            if n_shin >= self.min_mask_pixels // 2:
                shin_ratio = np.count_nonzero(skin_map & (shin_mask > 0)) / float(n_shin)

        if rect_mode:
            # Rectangle includes background between the legs; bare legs cover ~30-50% of it
            bare = skin_ratio >= (0.22 if skin_model.adaptive else 0.30)
            ref_thr = 0.30
        elif skin_model.adaptive:
            bare = shin_ratio >= 0.40 or thigh_ratio >= 0.45 or skin_ratio >= 0.32
            ref_thr = 0.32
        else:
            # Generic skin rule is less specific -> require much stronger evidence
            bare = shin_ratio >= 0.55 or skin_ratio >= 0.45
            ref_thr = 0.45

        if bare:
            base = 0.85 if skin_model.adaptive else 0.6
            q = base * min(1.0, max(skin_ratio, shin_ratio) / max(ref_thr, 1e-6)) * max(geom_q, 0.5)
            return np.zeros((h, w), dtype=np.uint8), "Bare Legs / Shorts", False, float(np.clip(q, 0.0, 1.0))

        # Pants: remove residual skin pixels (hands, ankles) from the garment mask
        lower_mask[skin_lower] = 0
        n_kept = int(np.count_nonzero(lower_mask))
        if n_kept < self.min_mask_pixels:
            return np.zeros((h, w), dtype=np.uint8), "None / Occluded", False, 0.0
        coverage_q = math.sqrt(min(1.0, n_kept / 300.0))
        q = geom_q * coverage_q * (1.0 - skin_ratio)
        return lower_mask, "Pants/Jeans", True, float(np.clip(q, 0.0, 1.0))

    def _leg_bands(self, xy: np.ndarray, confs: np.ndarray, h: int, w: int
                   ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Thigh and shin band masks drawn along hip->knee->ankle; wrists/hands excluded."""
        lh_ok, rh_ok = _conf_ok(confs, L_HIP), _conf_ok(confs, R_HIP)
        if lh_ok and rh_ok:
            hip_w = float(np.hypot(*(xy[L_HIP] - xy[R_HIP])))
        elif _conf_ok(confs, L_SHO) and _conf_ok(confs, R_SHO):
            hip_w = 0.6 * float(abs(xy[L_SHO][0] - xy[R_SHO][0]))
        else:
            hip_w = 0.25 * w
        t_thigh = int(np.clip(0.55 * hip_w, max(4.0, 0.08 * w), max(5.0, 0.30 * w)))
        t_shin = max(3, int(round(0.8 * t_thigh)))

        thigh = np.zeros((h, w), dtype=np.uint8)
        shin = np.zeros((h, w), dtype=np.uint8)
        used = []
        hip_y_vals = []
        lowest_y = -1e9
        for hip_i, knee_i, ank_i, hip_ok in ((L_HIP, L_KNE, L_ANK, lh_ok), (R_HIP, R_KNE, R_ANK, rh_ok)):
            if hip_ok:
                hip = xy[hip_i]
            elif lh_ok or rh_ok:
                other = xy[R_HIP] if lh_ok else xy[L_HIP]
                hip = other  # mirrored estimate: start the band at the visible hip
            else:
                continue
            hip_y_vals.append(float(hip[1]))
            if not _conf_ok(confs, knee_i):
                continue
            knee = xy[knee_i]
            if knee[1] <= hip[1] + 4:
                continue
            # start below the hip joint: jacket / coat / shirt hems usually cover the top of the thighs
            p_hip = tuple(int(round(v)) for v in hip + 0.25 * (knee - hip))
            p_knee = tuple(int(round(v)) for v in knee)
            cv2.line(thigh, p_hip, p_knee, 255, thickness=t_thigh)
            used.extend([float(confs[hip_i]) if hip_ok else 0.5, float(confs[knee_i])])
            lowest_y = max(lowest_y, float(knee[1]))
            if _conf_ok(confs, ank_i):
                ank = xy[ank_i]
                if ank[1] > knee[1] + 4:
                    # stop short of the ankle so shoes are not counted as trousers
                    end = knee + 0.85 * (ank - knee)
                    cv2.line(shin, p_knee, tuple(int(round(v)) for v in end), 255, thickness=t_shin)
                    used.append(float(confs[ank_i]))
                    lowest_y = max(lowest_y, float(end[1]))

        if not used:
            return None, None, 0.0

        # Hands hang next to the thighs: exclude wrist discs
        excl = np.zeros((h, w), dtype=np.uint8)
        for wi in (L_WRI, R_WRI):
            if _conf_ok(confs, wi):
                cv2.circle(excl, tuple(int(round(v)) for v in xy[wi]), int(max(3, 0.7 * t_thigh)), 255, -1)
        thigh[excl > 0] = 0
        shin[excl > 0] = 0
        shin[thigh > 0] = 0

        # Enough lower body must be inside the crop
        hip_y = float(np.mean(hip_y_vals)) if hip_y_vals else 0.0
        if min(lowest_y, h - 1) - hip_y < 0.12 * h:
            return None, None, 0.0
        geom_q = float(np.clip(np.mean(used), 0.5, 1.0))
        if np.count_nonzero(shin) == 0:
            geom_q *= 0.85
        return thigh, shin, geom_q
