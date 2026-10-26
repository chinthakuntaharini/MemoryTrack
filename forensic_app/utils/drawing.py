"""
Drawing & Visual Overlay Utilities for Forensic Re-ID
Renders glowing identity-coloured HUD boxes, translucent label chips, attribute swatches and skeletons.
All blending is done per region of interest (never full-frame copies or full-frame addWeighted).
"""

import logging
import math
import os
import re
import sys
import zlib
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:  # Anti-aliased TrueType HUD text (Pillow ships with ultralytics); Hershey fallback otherwise
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover - Pillow missing
    _PIL_AVAILABLE = False

# COCO 17 Keypoint Skeleton Connections
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),             # Face (Nose to eyes, eyes to ears)
    (5, 6),                                     # Left shoulder to right shoulder
    (5, 7), (7, 9),                             # Left arm (Shoulder -> Elbow -> Wrist)
    (6, 8), (8, 10),                            # Right arm (Shoulder -> Elbow -> Wrist)
    (5, 11), (6, 12),                           # Torso sides
    (11, 12),                                   # Hips
    (11, 13), (13, 15),                         # Left leg (Hip -> Knee -> Ankle)
    (12, 14), (14, 16)                          # Right leg (Hip -> Knee -> Ankle)
]

# Color Palette for Overlays (BGR format)
PALETTE = {
    "cyan": (248, 189, 56),      # Primary highlight #38bdf8
    "emerald": (153, 211, 52),   # New identity #34d399
    "amber": (36, 191, 251),     # Carried accessories #fbbf24
    "purple": (250, 139, 167),   # Re-identified #a78bfa
    "navy_dark": (32, 16, 11),   # Chip background #0b1020
    "white": (255, 255, 255),
    "text": (247, 237, 230),     # #e6edf7
    "muted": (184, 163, 148),    # Tentative / muted #94a3b8
    "red": (94, 63, 244),        # #f43f5e
    "skeleton_joint": (255, 255, 255),
    "skeleton_limb": (248, 189, 56),
    "skeleton_left": (252, 211, 125),   # #7dd3fc
    "skeleton_right": (250, 139, 167),  # #a78bfa
    "skeleton_center": (240, 232, 226),
}

# Short HUD labels for canonical accessory names
ACCESSORY_SHORT_LABELS: Dict[str, str] = {
    "backpack": "Backpack",
    "handbag": "Bag",
    "suitcase": "Suitcase",
    "umbrella": "Umbrella",
    "cap": "Cap",
    "helmet": "Helmet",
    "glasses": "Glasses",
    "face mask": "Mask",
    "tie": "Tie",
    "bottle": "Bottle",
    "cell phone": "Phone",
    "laptop": "Laptop",
}

# Short HUD labels for lower-garment presence classes
_LOWER_TYPE_SHORT = {
    "bare legs / shorts": "Shorts",
    "none / occluded": "Occluded",
    "not detected": "n/a",
    "pants/jeans": "Pants",
}

_GOLDEN_RATIO_CONJ = 0.6180339887498949
_UID_HUE_OFFSET_DEG = 337.5     # makes UID-0001 a sky cyan, UID-0002 amber, UID-0003 violet, ...

ColorLike = Union[str, Sequence[int], None]


# ---------------------------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------------------------
def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    """Convert HEX string (#RRGGBB) to BGR tuple."""
    hex_str = (hex_str or "").strip().lstrip('#')
    if len(hex_str) != 6:
        return (128, 128, 128)
    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
    except ValueError:
        return (128, 128, 128)
    return (b, g, r)


def _oklch_to_srgb(lightness: float, chroma: float, hue_deg: float) -> Tuple[float, float, float]:
    """Convert OKLCH to gamma-encoded sRGB floats (may be out of [0, 1] when out of gamut)."""
    h = math.radians(hue_deg)
    a, b = chroma * math.cos(h), chroma * math.sin(h)
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    l3, m3, s3 = l_ ** 3, m_ ** 3, s_ ** 3
    rl = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    gl = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    bl = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3

    def _gamma(c: float) -> float:
        if c <= 0.0031308:
            return 12.92 * c
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055

    return _gamma(rl), _gamma(gl), _gamma(bl)


def _vivid_rgb(hue_deg: float, lightness: float = 0.79, max_chroma: float = 0.165) -> Tuple[int, int, int]:
    """Vivid, perceptually even colour at a hue (chroma reduced until it fits the sRGB gamut)."""
    chroma = max_chroma
    while chroma > 0.0:
        r, g, b = _oklch_to_srgb(lightness, chroma, hue_deg)
        if -1e-4 <= r <= 1.0001 and -1e-4 <= g <= 1.0001 and -1e-4 <= b <= 1.0001:
            break
        chroma -= 0.005
    r, g, b = _oklch_to_srgb(lightness, max(chroma, 0.0), hue_deg)
    return tuple(int(round(min(1.0, max(0.0, c)) * 255)) for c in (r, g, b))  # type: ignore[return-value]


def _uid_index(uid: str) -> int:
    """Numeric identity index: trailing digits of the uid, else a stable hash."""
    m = re.search(r"(\d+)\D*$", uid)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return zlib.crc32(uid.encode("utf-8")) & 0xFFFFF


@lru_cache(maxsize=4096)
def identity_color_rgb(uid: Optional[str]) -> Tuple[int, int, int]:
    """Deterministic vivid RGB identity colour (golden-ratio hue stepping on the uid's number)."""
    if not uid or not str(uid).strip():
        return (148, 163, 184)
    n = _uid_index(str(uid).strip())
    hue = (_UID_HUE_OFFSET_DEG + n * _GOLDEN_RATIO_CONJ * 360.0) % 360.0
    return _vivid_rgb(hue)


def identity_color_bgr(uid: Optional[str]) -> Tuple[int, int, int]:
    """Deterministic identity colour as a BGR tuple (same uid -> same colour everywhere)."""
    r, g, b = identity_color_rgb(uid)
    return (b, g, r)


def identity_color_hex(uid: Optional[str]) -> str:
    """Deterministic identity colour as '#RRGGBB'."""
    r, g, b = identity_color_rgb(uid)
    return f"#{r:02X}{g:02X}{b:02X}"


def accessory_short_label(name: str) -> str:
    """Compact display label for an accessory (canonical or legacy spelling)."""
    key = (name or "").strip().lower()
    try:
        from forensic_app.core.catalog import accessory_label, normalize_accessory
        key = normalize_accessory(key)
        return ACCESSORY_SHORT_LABELS.get(key) or accessory_label(key).split(" / ")[0]
    except Exception:
        return ACCESSORY_SHORT_LABELS.get(key, key.title())


def _resolve_bgr(color: ColorLike, fallback_uid: Optional[str]) -> Tuple[int, int, int]:
    """Accept '#RRGGBB', a BGR tuple, or None (derive from uid)."""
    if isinstance(color, str) and color.strip():
        return hex_to_bgr(color)
    if isinstance(color, (tuple, list, np.ndarray)) and len(color) >= 3:
        return (int(color[0]), int(color[1]), int(color[2]))
    return identity_color_bgr(fallback_uid)


def _mix(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))  # type: ignore[return-value]


# ---------------------------------------------------------------------------------------------
# Text rendering (TrueType via Pillow per ROI, Hershey fallback)
# ---------------------------------------------------------------------------------------------
_FONT_FILES = {
    False: ("segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "LiberationSans-Regular.ttf"),
    True: ("seguisb.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf",
           "LiberationSans-Bold.ttf"),
}


def _font_dirs() -> List[str]:
    dirs = []
    if sys.platform == "win32":
        dirs.append(os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(os.path.join(local, "Microsoft", "Windows", "Fonts"))
    dirs += ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF", "/usr/share/fonts/dejavu",
             "/usr/share/fonts/truetype/liberation", "/Library/Fonts", "/System/Library/Fonts/Supplemental"]
    return dirs


@lru_cache(maxsize=2)
def _font_path(bold: bool) -> Optional[str]:
    for d in _font_dirs():
        for name in _FONT_FILES[bold]:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


@lru_cache(maxsize=64)
def _pil_font(size: int, bold: bool):
    """Cached Pillow TrueType font (None when unavailable)."""
    if not _PIL_AVAILABLE:
        return None
    path = _font_path(bold) or _font_path(not bold)
    try:
        if path:
            return ImageFont.truetype(path, size)
        return ImageFont.load_default(size)
    except Exception:
        return None


def _hershey_scale(size: int) -> float:
    return max(0.3, size / 30.0)


@lru_cache(maxsize=2048)
def _measure_text(text: str, size: int, bold: bool) -> Tuple[int, int, int]:
    """(width, height, baseline offset) of a single text line at a pixel size."""
    font = _pil_font(size, bold)
    if font is not None:
        try:
            ascent, descent = font.getmetrics()
            width = int(math.ceil(font.getlength(text)))
            return width, ascent + descent, ascent
        except Exception:
            pass
    scale = _hershey_scale(size)
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2 if bold else 1)
    return w, h + base, h


TextItem = Tuple[int, int, str, int, Tuple[int, int, int], bool]   # (x, y_top, text, size, bgr, bold)


@lru_cache(maxsize=384)
def _text_bitmap(text: str, size: int, bold: bool) -> Optional[np.ndarray]:
    """Cached anti-aliased coverage mask (float32 0..1, h x w) of a text line rendered by Pillow."""
    font = _pil_font(size, bold)
    if font is None or not text:
        return None
    try:
        width, height, _ = _measure_text(text, size, bold)
        img = Image.new("L", (max(1, width + 2), max(1, height)), 0)
        ImageDraw.Draw(img).text((0, 0), text, font=font, fill=255)
        mask = np.asarray(img, dtype=np.float32) / 255.0
        mask.setflags(write=False)
        return mask
    except Exception:
        return None


def _blend_mask(frame: np.ndarray, x: int, y: int, mask: np.ndarray, color: Tuple[int, int, int],
                alpha: float = 1.0) -> None:
    """Blend a solid colour into the frame through a coverage mask placed at (x, y) (clipped)."""
    h_f, w_f = frame.shape[:2]
    mh, mw = mask.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, x), max(0, y), min(w_f, x + mw), min(h_f, y + mh)
    if cx2 <= cx1 or cy2 <= cy1:
        return
    w2 = mask[cy1 - y:cy2 - y, cx1 - x:cx2 - x]
    if alpha != 1.0:
        w2 = w2 * float(alpha)
    w2 = np.ascontiguousarray(w2, dtype=np.float32)
    roi = frame[cy1:cy2, cx1:cx2]
    solid = np.empty_like(roi)
    solid[:] = color
    roi[:] = cv2.blendLinear(roi, solid, 1.0 - w2, w2)


def _draw_text_items(frame: np.ndarray, rect: Tuple[int, int, int, int], items: List[TextItem]) -> None:
    """Draw text items (TrueType coverage masks blended per glyph box; Hershey fallback)."""
    for x, y, text, size, color, bold in items:
        mask = _text_bitmap(text, size, bold) if _PIL_AVAILABLE else None
        if mask is not None:
            _blend_mask(frame, int(x), int(y), mask, color)
            continue
        _, _, base = _measure_text(text, size, bold)
        cv2.putText(frame, text, (int(x), int(y) + base), cv2.FONT_HERSHEY_SIMPLEX, _hershey_scale(size), color,
                    2 if bold else 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------------------------
# ROI blending primitives
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=256)
def _rounded_mask(w: int, h: int, radius: int) -> np.ndarray:
    """Anti-aliased rounded-rectangle coverage mask (float32 0..1, shape h x w)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    r = int(max(0, min(radius, w // 2, h // 2)))
    if r <= 0:
        mask[:] = 255
    else:
        cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
        cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
        for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
            cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    out = mask.astype(np.float32) / 255.0
    out.setflags(write=False)
    return out


def _blend_rounded(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int],
                   alpha: float, radius: int) -> None:
    """Alpha-blend a rounded rectangle of colour into the frame (ROI only)."""
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return
    _blend_mask(frame, x1, y1, _rounded_mask(w, h, radius), color, alpha)


def _rounded_border(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, radius: int,
                    color: Tuple[int, int, int], thickness: int = 1) -> None:
    """Anti-aliased rounded-rectangle outline."""
    r = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    lt = cv2.LINE_AA
    cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness, lt)
    cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness, lt)
    cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness, lt)
    cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness, lt)
    if r > 0:
        cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, lt)
        cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, lt)
        cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, lt)
        cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, lt)


def _glow_band(frame: np.ndarray, box: Tuple[int, int, int, int], band: Tuple[int, int, int, int],
               color: Tuple[int, int, int], strength: float, sigma: float, thickness: int) -> None:
    """Screen-blend a blurred outline of `box` into one band ROI of the frame."""
    h_f, w_f = frame.shape[:2]
    bx1, by1, bx2, by2 = max(0, band[0]), max(0, band[1]), min(w_f, band[2]), min(h_f, band[3])
    if bx2 <= bx1 or by2 <= by1:
        return
    x1, y1, x2, y2 = box
    mask = np.zeros((by2 - by1, bx2 - bx1), dtype=np.uint8)
    cv2.rectangle(mask, (x1 - bx1, y1 - by1), (x2 - bx1, y2 - by1), 255, thickness)
    mask = _soft_blur(mask, sigma)
    k = strength / _glow_peak(thickness, round(sigma, 2))
    # Additive light: brightens toward the identity colour, never darkens the scene (saturating add)
    glow = cv2.multiply(cv2.merge([mask, mask, mask]), tuple(k * c / 255.0 for c in color) + (0.0,))
    roi = frame[by1:by2, bx1:bx2]
    cv2.add(roi, glow, dst=roi)


def _soft_blur(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Fast Gaussian-like blur (stack blur when available, ~10x faster than GaussianBlur on uint8)."""
    k = max(3, int(round(sigma * 6.0)) | 1)
    if hasattr(cv2, "stackBlur"):
        return cv2.stackBlur(mask, (k, k))
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)


@lru_cache(maxsize=64)
def _glow_peak(thickness: int, sigma: float) -> float:
    """Peak coverage (0..1) of a blurred line of given thickness (normalises glow strength)."""
    size = int(sigma * 8 + thickness + 6)
    line = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    cv2.line(line, (0, c), (size - 1, c), 255, thickness)
    blurred = _soft_blur(line, sigma)
    return float(max(blurred.max() / 255.0, 1e-3))


def _draw_glow_rect(frame: np.ndarray, box: Tuple[int, int, int, int], color: Tuple[int, int, int],
                    strength: float, sigma: float, thickness: int) -> None:
    """Soft glowing halo around a box, computed only in four thin bands along its edges."""
    x1, y1, x2, y2 = box
    pad = int(math.ceil(sigma * 3.0)) + thickness
    inner_w, inner_h = x2 - x1, y2 - y1
    if inner_h > 4 * pad and inner_w > 4 * pad:
        bands = [
            (x1 - pad, y1 - pad, x2 + pad + 1, y1 + pad + 1),          # top (incl. corners)
            (x1 - pad, y2 - pad, x2 + pad + 1, y2 + pad + 1),          # bottom (incl. corners)
            (x1 - pad, y1 + pad + 1, x1 + pad + 1, y2 - pad),          # left
            (x2 - pad, y1 + pad + 1, x2 + pad + 1, y2 - pad),          # right
        ]
    else:
        bands = [(x1 - pad, y1 - pad, x2 + pad + 1, y2 + pad + 1)]
    for band in bands:
        _glow_band(frame, box, band, color, strength, sigma, thickness)


def _draw_dashed_rect(frame: np.ndarray, box: Tuple[int, int, int, int], color: Tuple[int, int, int],
                      thickness: int, dash: int, gap: int) -> None:
    """Dashed rectangle outline (tentative tracks)."""
    x1, y1, x2, y2 = box
    step = max(2, dash + gap)

    def _segments(a: int, b: int):
        pos = a
        while pos < b:
            yield pos, min(pos + dash, b)
            pos += step

    for s, e in _segments(x1, x2):
        cv2.line(frame, (s, y1), (e, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (s, y2), (e, y2), color, thickness, cv2.LINE_AA)
    for s, e in _segments(y1, y2):
        cv2.line(frame, (x1, s), (x1, e), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, s), (x2, e), color, thickness, cv2.LINE_AA)


def _draw_corners(frame: np.ndarray, box: Tuple[int, int, int, int], color: Tuple[int, int, int],
                  length: int, thickness: int) -> None:
    """HUD corner brackets."""
    x1, y1, x2, y2 = box
    lt = cv2.LINE_AA
    for (cx, cy), (dx, dy) in (((x1, y1), (1, 1)), ((x2, y1), (-1, 1)), ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))):
        cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, thickness, lt)
        cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, thickness, lt)


def _hud_scale(frame_h: int) -> float:
    """HUD scale relative to a 720p frame (keeps text readable after the canvas rescales)."""
    return float(min(2.2, max(0.72, frame_h / 720.0)))


def _short_lower(lower_color: str) -> str:
    return _LOWER_TYPE_SHORT.get((lower_color or "").strip().lower(), lower_color or "n/a")


# ---------------------------------------------------------------------------------------------
# Public drawing API
# ---------------------------------------------------------------------------------------------
def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, conf_thresh: float = 0.3,
                  color: ColorLike = None) -> None:
    """
    Render 17-keypoint skeleton on image frame.
    keypoints shape: (17, 2) or (17, 3) where columns are (x, y, [conf]).
    `color` (hex or BGR) tints the limbs, e.g. with the identity colour; default is a cyan/violet scheme.
    """
    if frame is None or keypoints is None or len(keypoints) < 17:
        return

    h, w = frame.shape[:2]
    pts = []
    confs = []

    for kp in keypoints:
        x, y = float(kp[0]), float(kp[1])
        c = float(kp[2]) if len(kp) > 2 else 1.0
        # Check normalized vs absolute coordinates
        if x <= 1.0 and y <= 1.0 and w > 1:
            px, py = int(x * w), int(y * h)
        else:
            px, py = int(x), int(y)
        pts.append((px, py))
        confs.append(c)

    tint = _resolve_bgr(color, None) if color is not None else None
    thickness = max(1, int(round(2 * _hud_scale(h))))

    # Draw limb connections (left side sky, right side violet, centre light)
    for idx1, idx2 in SKELETON_EDGES:
        if idx1 < len(pts) and idx2 < len(pts):
            if confs[idx1] >= conf_thresh and confs[idx2] >= conf_thresh:
                pt1 = pts[idx1]
                pt2 = pts[idx2]
                if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                    if tint is not None:
                        limb = tint
                    elif idx1 % 2 == 1 and idx2 % 2 == 1:
                        limb = PALETTE["skeleton_left"]
                    elif idx1 % 2 == 0 and idx2 % 2 == 0 and idx1 > 0:
                        limb = PALETTE["skeleton_right"]
                    else:
                        limb = PALETTE["skeleton_center"]
                    cv2.line(frame, pt1, pt2, limb, thickness, cv2.LINE_AA)

    # Draw joint points (light core with dark outline)
    r = max(2, thickness + 1)
    for idx, (px, py) in enumerate(pts):
        if confs[idx] >= conf_thresh:
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(frame, (px, py), r + 1, (40, 24, 12), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), r, PALETTE["skeleton_joint"], -1, cv2.LINE_AA)


def draw_forensic_bbox(frame: np.ndarray,
                       bbox: Tuple[int, int, int, int],
                       person_uid: str,
                       track_id: int,
                       upper_color: str = "Unknown",
                       upper_hex: str = "#808080",
                       lower_color: str = "Unknown",
                       lower_hex: str = "#808080",
                       accessories: Optional[List[str]] = None,
                       posture: str = "Standing",
                       confidence: float = 1.0,
                       is_selected: bool = False,
                       *,
                       identity_color: ColorLike = None,
                       status: str = "confirmed",
                       match_score: Optional[float] = None,
                       display_label: Optional[str] = None) -> None:
    """
    Render a glowing identity-coloured HUD box with a rounded uid chip (uid, track id, match %) and a
    bottom chip with upper/lower colour swatches and accessories. Tentative tracks are drawn as a muted
    dashed box labelled "Identifying...". All blending is per-ROI.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return
    h_frame, w_frame = frame.shape[:2]
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
    except Exception:
        return
    x1, y1 = max(0, min(w_frame - 1, x1)), max(0, min(h_frame - 1, y1))
    x2, y2 = max(0, min(w_frame - 1, x2)), max(0, min(h_frame - 1, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    box = (x1, y1, x2, y2)
    w_box, h_box = x2 - x1, y2 - y1

    s = _hud_scale(h_frame)
    status = (status or "confirmed").lower()
    tentative = status == "tentative" or not person_uid
    accessories = [a for a in (accessories or []) if a]

    # ---- Box -------------------------------------------------------------------------------
    if tentative:
        color = PALETTE["muted"]
        _draw_dashed_rect(frame, box, color, max(1, int(round(1.5 * s))), int(10 * s), int(7 * s))
        c_len = int(max(6, min(16 * s, w_box / 4, h_box / 4)))
        _draw_corners(frame, box, PALETTE["text"], c_len, max(1, int(round(2 * s))))
    else:
        color = _resolve_bgr(identity_color, person_uid)
        _draw_glow_rect(frame, box, color, strength=0.85 if is_selected else 0.62, sigma=5.0 * s,
                        thickness=max(2, int(round(3 * s))))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, int(round(1.2 * s))), cv2.LINE_AA)
        c_len = int(max(6, min(22 * s, w_box / 4, h_box / 4)))
        _draw_corners(frame, box, _mix(color, PALETTE["white"], 0.45), c_len, max(2, int(round(3 * s))))

    # ---- Top identity chip -----------------------------------------------------------------
    f_main = max(12, int(round(17 * s)))
    f_small = max(10, int(round(13.5 * s)))
    pad_x, pad_y, gap = int(9 * s), int(5 * s), int(7 * s)
    dot_r = max(3, int(round(4 * s)))

    label = display_label or ("Identifying..." if tentative else str(person_uid))
    track_txt = f"#{track_id}"
    badge_txt, badge_col = "", None
    if not tentative:
        if status == "new":
            badge_txt, badge_col = "NEW", PALETTE["emerald"]
        elif status in ("reacquired", "merged"):
            badge_txt, badge_col = "RE-ID", PALETTE["purple"]
    score_txt = ""
    if match_score is not None and not tentative:
        try:
            score_txt = f"{float(match_score) * 100:.0f}%"
        except (TypeError, ValueError):
            score_txt = ""

    lw, lh, _ = _measure_text(label, f_main, True)
    tw, th, _ = _measure_text(track_txt, f_small, False)
    chip_h = lh + 2 * pad_y
    parts_w = pad_x + dot_r * 2 + gap + lw + gap + tw
    bw = 0
    if badge_txt:
        bw, bh, _ = _measure_text(badge_txt, f_small, True)
        parts_w += gap + bw + 2 * int(6 * s)
    sw = 0
    if score_txt:
        sw, _, _ = _measure_text(score_txt, f_small, True)
        parts_w += gap + sw
    chip_w = parts_w + pad_x

    cy2 = y1 - int(4 * s)
    cy1 = cy2 - chip_h
    if cy1 < 0:                                   # no room above: tuck the chip inside the box
        cy1 = y1 + int(4 * s)
        cy2 = cy1 + chip_h
    cx1 = min(x1, max(0, w_frame - chip_w - 1))
    cx2 = cx1 + chip_w
    radius = chip_h // 2

    _blend_rounded(frame, cx1, cy1, cx2, cy2, PALETTE["navy_dark"], 0.78, radius)
    border = _mix(color, PALETTE["navy_dark"], 0.35)
    _rounded_border(frame, cx1, cy1, cx2, cy2, radius, border, 1)

    dot_c = (cx1 + pad_x + dot_r, (cy1 + cy2) // 2)
    cv2.circle(frame, dot_c, dot_r + max(1, int(2 * s)), _mix(color, PALETTE["navy_dark"], 0.55), -1, cv2.LINE_AA)
    cv2.circle(frame, dot_c, dot_r, color, -1, cv2.LINE_AA)

    items: List[TextItem] = []
    tx = cx1 + pad_x + dot_r * 2 + gap
    label_col = PALETTE["text"] if tentative else _mix(color, PALETTE["white"], 0.55)
    items.append((tx, cy1 + pad_y, label, f_main, label_col, True))
    tx += lw + gap
    items.append((tx, cy1 + pad_y + max(0, (lh - th) // 2), track_txt, f_small, PALETTE["muted"], False))
    tx += tw
    if badge_txt:
        tx += gap
        bx1, bx2 = tx, tx + bw + 2 * int(6 * s)
        by1, by2 = cy1 + int(4 * s), cy2 - int(4 * s)
        _blend_rounded(frame, bx1, by1, bx2, by2, badge_col, 0.30, (by2 - by1) // 2)
        items.append((bx1 + int(6 * s), cy1 + pad_y + max(0, (lh - th) // 2), badge_txt, f_small,
                      _mix(badge_col, PALETTE["white"], 0.55), True))
        tx = bx2
    if score_txt:
        tx += gap
        items.append((tx, cy1 + pad_y + max(0, (lh - th) // 2), score_txt, f_small, PALETTE["text"], True))
    _draw_text_items(frame, (cx1, cy1, cx2, cy2), items)

    # ---- Bottom attribute chip -------------------------------------------------------------
    f_attr = max(10, int(round(14 * s)))
    sq = max(9, int(round(12 * s)))
    attr_txt = f"{upper_color or 'Unknown'} / {_short_lower(lower_color)}"
    acc_txt = "  ".join(accessory_short_label(a) for a in accessories[:3])
    if len(accessories) > 3:
        acc_txt += f"  +{len(accessories) - 3}"
    aw, ah, _ = _measure_text(attr_txt, f_attr, False)
    line_h = max(ah, sq)
    acc_w = 0
    if acc_txt:
        acc_w, _, _ = _measure_text(acc_txt, f_attr, True)
    inner_w = max(sq * 2 + int(5 * s) + gap + aw, acc_w)
    b_w = inner_w + 2 * pad_x
    b_h = line_h + 2 * pad_y + ((line_h + int(3 * s)) if acc_txt else 0)

    by1 = y2 + int(4 * s)
    if by1 + b_h > h_frame:                       # no room below: tuck the chip inside the box
        by1 = max(0, y2 - b_h - int(4 * s))
    by2 = by1 + b_h
    bx1 = min(x1, max(0, w_frame - b_w - 1))
    bx2 = bx1 + b_w
    b_radius = int(8 * s)
    _blend_rounded(frame, bx1, by1, bx2, by2, PALETTE["navy_dark"], 0.74, b_radius)
    _rounded_border(frame, bx1, by1, bx2, by2, b_radius, _mix(color, PALETTE["navy_dark"], 0.55), 1)

    row_y = by1 + pad_y
    sq_y = row_y + (line_h - sq) // 2
    sx = bx1 + pad_x
    for hex_c in (upper_hex, lower_hex):
        sc = hex_to_bgr(hex_c or "#808080")
        cv2.rectangle(frame, (sx, sq_y), (sx + sq, sq_y + sq), sc, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (sx, sq_y), (sx + sq, sq_y + sq), (235, 235, 235), 1, cv2.LINE_AA)
        sx += sq + int(5 * s)
    items = [(sx + gap - int(5 * s), row_y + (line_h - ah) // 2, attr_txt, f_attr, PALETTE["text"], False)]
    if acc_txt:
        items.append((bx1 + pad_x, row_y + line_h + int(3 * s), acc_txt, f_attr, PALETTE["amber"], True))
    _draw_text_items(frame, (bx1, by1, bx2, by2), items)


__all__ = [
    "SKELETON_EDGES", "PALETTE", "ACCESSORY_SHORT_LABELS", "hex_to_bgr", "identity_color_rgb",
    "identity_color_bgr", "identity_color_hex", "accessory_short_label", "draw_skeleton", "draw_forensic_bbox",
]
