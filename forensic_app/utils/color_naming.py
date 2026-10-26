"""
Forensic Color Naming Utility (ISCC-NBS / CIELAB)
Maps segmented garment pixels to standardized forensic color descriptors using CIELAB / LCh space.
"""

import math
import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional


def _rgb_to_lab_exact(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert an sRGB triplet (0-255) to CIELAB (D65) with float precision."""
    bgr = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.float32) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[0, 0]
    return (round(float(lab[0]), 1), round(float(lab[1]), 1), round(float(lab[2]), 1))


# Standard forensic garment color palette.
# The original 12 surveillance names are kept (same display hex); common garment colors are added so
# that jeans, grey hoodies, green/purple/pink/orange shirts etc. get a meaningful name.
# "lab" is derived from "rgb" (D65) so palette lookups and measured colors share one space.
_PALETTE_RGB: Dict[str, Tuple[int, int, int]] = {
    # --- original 12 ---
    "Black": (20, 20, 20),
    "Charcoal": (54, 57, 63),
    "White": (240, 240, 240),
    "Navy Blue": (15, 23, 62),
    "Royal Blue": (29, 78, 216),
    "Beige": (224, 212, 187),
    "Khaki": (189, 172, 129),
    "Brown": (101, 55, 30),
    "Maroon": (115, 16, 29),
    "Red": (220, 38, 38),
    "Olive Green": (74, 93, 35),
    "Yellow": (234, 179, 8),
    # --- extended garment colors ---
    "Grey": (128, 128, 128),
    "Light Grey": (190, 190, 190),
    "Denim Blue": (74, 100, 145),
    "Sky Blue": (135, 206, 235),
    "Teal": (0, 128, 128),
    "Green": (34, 139, 34),
    "Dark Green": (20, 70, 40),
    "Purple": (112, 48, 160),
    "Lavender": (190, 170, 220),
    "Pink": (240, 128, 170),
    "Orange": (245, 130, 32),
    "Cream": (245, 240, 215),
    "Tan": (210, 180, 140),
}

FORENSIC_COLOR_PALETTE: Dict[str, Dict] = {
    name: {
        "rgb": rgb,
        "hex": "#{:02X}{:02X}{:02X}".format(*rgb),
        "lab": _rgb_to_lab_exact(rgb),
    }
    for name, rgb in _PALETTE_RGB.items()
}

# Color family of every palette name (coarse grouping used for tolerant matching and queries)
COLOR_FAMILIES: Dict[str, str] = {
    "Black": "neutral-dark",
    "Charcoal": "neutral-dark",
    "Grey": "neutral-mid",
    "Light Grey": "neutral-light",
    "White": "neutral-light",
    "Cream": "neutral-light",
    "Navy Blue": "blue",
    "Royal Blue": "blue",
    "Denim Blue": "blue",
    "Sky Blue": "blue",
    "Teal": "teal",
    "Green": "green",
    "Dark Green": "green",
    "Olive Green": "green",
    "Yellow": "yellow",
    "Orange": "orange",
    "Brown": "brown",
    "Tan": "brown",
    "Khaki": "brown",
    "Beige": "brown",
    "Red": "red",
    "Maroon": "red",
    "Pink": "pink",
    "Purple": "purple",
    "Lavender": "purple",
}

# Loose keyword -> family mapping for names outside the palette (legacy records, free-text queries)
_FAMILY_KEYWORDS: List[Tuple[str, str]] = [
    ("bare", "skin"), ("skin", "skin"), ("occluded", "none"), ("none", "none"), ("unknown", "unknown"),
    ("navy", "blue"), ("denim", "blue"), ("blue", "blue"), ("teal", "teal"), ("turquoise", "teal"),
    ("cyan", "teal"), ("olive", "green"), ("green", "green"), ("black", "neutral-dark"),
    ("charcoal", "neutral-dark"), ("light gr", "neutral-light"), ("silver", "neutral-light"),
    ("grey", "neutral-mid"), ("gray", "neutral-mid"), ("white", "neutral-light"), ("cream", "neutral-light"),
    ("ivory", "neutral-light"), ("maroon", "red"), ("burgundy", "red"), ("red", "red"), ("pink", "pink"),
    ("magenta", "pink"), ("purple", "purple"), ("violet", "purple"), ("lavender", "purple"),
    ("orange", "orange"), ("yellow", "yellow"), ("gold", "yellow"), ("brown", "brown"), ("tan", "brown"),
    ("khaki", "brown"), ("beige", "brown"), ("camel", "brown"),
]


def bgr_to_cielab(bgr_pixel: np.ndarray) -> np.ndarray:
    """
    Convert a BGR pixel array or image to CIELAB space.
    Safely handles 1D pixel (3,), 2D array of pixels (N, 3), and 3D images (H, W, 3).
    """
    if bgr_pixel is None or bgr_pixel.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # 1D single pixel: shape (3,)
    if bgr_pixel.ndim == 1 and len(bgr_pixel) == 3:
        img = bgr_pixel.reshape(1, 1, 3).astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        L = lab[0, 0, 0] * (100.0 / 255.0)
        a = float(lab[0, 0, 1]) - 128.0
        b = float(lab[0, 0, 2]) - 128.0
        return np.array([L, a, b], dtype=np.float32)

    # 2D array of N pixels: shape (N, 3) -> reshape to (N, 1, 3) so OpenCV sees 3 channels!
    elif bgr_pixel.ndim == 2 and bgr_pixel.shape[1] == 3:
        img = bgr_pixel.reshape(-1, 1, 3).astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
        lab[:, 0] = lab[:, 0] * (100.0 / 255.0)
        lab[:, 1] = lab[:, 1] - 128.0
        lab[:, 2] = lab[:, 2] - 128.0
        return lab

    # 3D image: shape (H, W, 3)
    elif bgr_pixel.ndim == 3 and bgr_pixel.shape[2] == 3:
        lab = cv2.cvtColor(bgr_pixel.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)
        lab[:, :, 0] = lab[:, :, 0] * (100.0 / 255.0)
        lab[:, :, 1] = lab[:, :, 1] - 128.0
        lab[:, :, 2] = lab[:, :, 2] - 128.0
        return lab

    # Fallback
    return np.zeros((0, 3), dtype=np.float32)


def lab_to_bgr(lab_val) -> Tuple[int, int, int]:
    """Convert a single (L, a, b) vector back to an 8-bit BGR triplet (gamut clipped)."""
    lab = np.array([[[float(lab_val[0]), float(lab_val[1]), float(lab_val[2])]]], dtype=np.float32)
    bgr = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)[0, 0], 0.0, 1.0)
    return int(round(bgr[0] * 255)), int(round(bgr[1] * 255)), int(round(bgr[2] * 255))


def lab_to_hex(lab_val) -> str:
    """Return the '#RRGGBB' hex string of a measured (L, a, b) color."""
    b, g, r = lab_to_bgr(lab_val)
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def delta_e_cielab(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Compute Euclidean Delta-E (CIE76) color difference in CIELAB space."""
    return float(np.sqrt(np.sum((np.asarray(lab1, dtype=np.float32) - np.asarray(lab2, dtype=np.float32)) ** 2)))


def delta_e_ciede2000(lab1, lab2) -> float:
    """Compute the CIEDE2000 perceptual color difference between two (L, a, b) colors."""
    L1, a1, b1 = (float(v) for v in lab1[:3])
    L2, a2, b2 = (float(v) for v in lab2[:3])

    c_bar = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2.0
    g = 0.5 * (1.0 - math.sqrt(c_bar ** 7 / (c_bar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0

    d_lp = L2 - L1
    d_cp = c2p - c1p
    if c1p * c2p == 0.0:
        d_hp = 0.0
    else:
        d_hp = h2p - h1p
        if d_hp > 180.0:
            d_hp -= 360.0
        elif d_hp < -180.0:
            d_hp += 360.0
    d_big_hp = 2.0 * math.sqrt(c1p * c2p) * math.sin(math.radians(d_hp) / 2.0)

    lp_bar = (L1 + L2) / 2.0
    cp_bar = (c1p + c2p) / 2.0
    if c1p * c2p == 0.0:
        hp_bar = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        hp_bar = (h1p + h2p) / 2.0
    elif h1p + h2p < 360.0:
        hp_bar = (h1p + h2p + 360.0) / 2.0
    else:
        hp_bar = (h1p + h2p - 360.0) / 2.0

    t = (1.0
         - 0.17 * math.cos(math.radians(hp_bar - 30.0))
         + 0.24 * math.cos(math.radians(2.0 * hp_bar))
         + 0.32 * math.cos(math.radians(3.0 * hp_bar + 6.0))
         - 0.20 * math.cos(math.radians(4.0 * hp_bar - 63.0)))
    d_theta = 30.0 * math.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    r_c = 2.0 * math.sqrt(cp_bar ** 7 / (cp_bar ** 7 + 25.0 ** 7))
    s_l = 1.0 + (0.015 * (lp_bar - 50.0) ** 2) / math.sqrt(20.0 + (lp_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * cp_bar
    s_h = 1.0 + 0.015 * cp_bar * t
    r_t = -math.sin(math.radians(2.0 * d_theta)) * r_c

    return float(math.sqrt(
        (d_lp / s_l) ** 2 + (d_cp / s_c) ** 2 + (d_big_hp / s_h) ** 2
        + r_t * (d_cp / s_c) * (d_big_hp / s_h)
    ))


def color_similarity(lab1, lab2, scale: float = 20.0) -> float:
    """Map CIEDE2000 distance to a 0..1 similarity (1 = identical, ~0.37 at `scale`)."""
    if lab1 is None or lab2 is None or len(lab1) < 3 or len(lab2) < 3:
        return 0.0
    d = delta_e_ciede2000(lab1, lab2)
    return float(math.exp(-((d / max(scale, 1e-6)) ** 2)))


def lab_to_lch(lab_val) -> Tuple[float, float, float]:
    """Convert (L, a, b) to (L, C, h) with hue in degrees [0, 360)."""
    L, a, b = float(lab_val[0]), float(lab_val[1]), float(lab_val[2])
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360.0


def achromatic_chroma_gate(L: float, hue: float) -> float:
    """
    Chroma below which a color is treated as neutral (black/grey/white).
    Slack grows in deep shadows and highlights (chroma is unreliable there) and for cool hues
    (bluish camera white-balance casts on grey fabric).
    """
    gate = 7.0
    if L < 25.0:
        gate += (25.0 - L) * 0.12
    if L > 75.0:
        gate += (L - 75.0) * 0.15
    if 220.0 <= hue <= 300.0:
        gate += 2.5
    return gate


def _hsv_hue_deg(lab_val) -> float:
    """HSV hue (degrees) of the sRGB rendering of a Lab color — separates blue from purple robustly."""
    b, g, r = (v / 255.0 for v in lab_to_bgr(lab_val))
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d <= 1e-6:
        return 0.0
    if mx == r:
        h = ((g - b) / d) % 6.0
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return (h * 60.0) % 360.0


def _neutral_name(L: float) -> str:
    """Name an achromatic color by lightness."""
    if L < 22.0:
        return "Black"
    if L < 38.0:
        return "Charcoal"
    if L < 60.0:
        return "Grey"
    if L < 83.0:
        return "Light Grey"
    return "White"


def _blue_name(L: float, C: float) -> str:
    """Name a blue by lightness and chroma (navy / denim / royal / sky)."""
    if L < 29.0 and not (C >= 50.0 and L >= 22.0):
        return "Navy Blue"
    if L >= 62.0:
        return "Sky Blue"
    if C >= 42.0:
        return "Royal Blue"
    return "Denim Blue"


def classify_cielab_name(lab_val) -> str:
    """Classify an (L, a, b) vector into a forensic palette color name using LCh rules."""
    if lab_val is None or len(lab_val) < 3:
        return "Unknown"
    L, C, h = lab_to_lch(lab_val)
    if not all(math.isfinite(v) for v in (L, C, h)):
        return "Unknown"

    # 1. Achromatic gate (very dark pixels need a stronger chroma to count as colored)
    gate = achromatic_chroma_gate(L, h)
    # Deep shadows: chroma is dominated by sensor / compression noise unless it is strong
    if L < 6.0:
        gate = max(gate, 18.0)
    elif L < 12.0:
        gate = max(gate, 15.0)
    elif L < 20.0:
        gate = max(gate, 12.0)
    if C < gate:
        return _neutral_name(L)
    if L >= 92.0 and C < 12.0:
        return "White"

    # 2. Chromatic: hue sector + lightness / chroma modifiers
    if h >= 345.0 or h < 15.0:                      # pink / crimson / wine
        if L < 32.0:
            return "Maroon"
        if h < 15.0 and C >= 50.0 and L < 50.0:
            return "Red"                            # crimson / raspberry
        return "Pink"

    if h < 45.0:                                    # red / maroon / salmon
        if L < 32.0:
            return "Maroon"
        if C < 28.0:
            if L >= 55.0:
                return "Pink"
            return "Brown" if h >= 30.0 else "Maroon"
        if L >= 65.0:
            return "Pink"
        return "Red"

    if h < 75.0:                                    # orange / brown / tan
        if C >= 48.0 and L >= 50.0:
            return "Orange"
        if L >= 80.0:
            return "Orange" if C >= 30.0 else "Beige"
        if L >= 60.0:
            return "Tan"
        if L >= 48.0 and C < 24.0:
            return "Tan"                            # light, desaturated warm tone (camel / shaded beige)
        return "Brown"

    if h < 100.0:                                   # yellow / khaki / beige / cream
        if C >= 45.0:
            if L >= 55.0:
                return "Yellow"
            return "Olive Green" if h >= 85.0 else "Brown"
        if L >= 90.0:
            return "Cream"
        if L >= 78.0:
            return "Beige"
        if L >= 58.0:
            return "Khaki" if h >= 84.0 else "Tan"
        if L >= 45.0 or C < 22.0:
            if h >= 84.0:
                return "Khaki"                      # incl. shaded khaki
            return "Tan" if (L >= 48.0 and C < 24.0) else "Brown"
        return "Olive Green" if h >= 88.0 else "Brown"

    if h < 128.0:                                   # yellow-green / olive
        if L >= 68.0 and C >= 45.0:
            return "Yellow" if h < 110.0 else "Green"
        if L >= 85.0:
            return "Cream"
        if L >= 70.0:
            return "Khaki" if h < 110.0 else "Green"
        if C < 15.0 and L >= 45.0:
            return "Khaki"                          # greyish olive / shaded cream
        return "Olive Green"

    if h < 170.0:                                   # green
        return "Dark Green" if L < 38.0 else "Green"

    if h < 225.0:                                   # teal / turquoise / light blue
        if L >= 65.0 and h >= 195.0:
            return "Sky Blue"
        return "Teal"

    if h < 270.0:                                   # blue (cyan side)
        return _blue_name(L, C)

    # 270..345: saturated blues and purples overlap in Lab hue -> resolve with sRGB HSV hue
    hsv_h = _hsv_hue_deg(lab_val)
    if hsv_h < 258.0 and hsv_h > 150.0:
        return _blue_name(L, C)
    if 258.0 <= hsv_h < 290.0:
        return "Lavender" if L >= 70.0 else "Purple"
    # magenta side
    return "Purple" if L < 55.0 else "Pink"


def classify_cielab_color(lab_val: np.ndarray) -> Tuple[str, str]:
    """
    Classify a single (L, a, b) vector into standard forensic color name and hex.
    Returns (color_name, hex_code).
    """
    name = classify_cielab_name(lab_val)
    if name not in FORENSIC_COLOR_PALETTE:
        return "Unknown", "#808080"
    return name, FORENSIC_COLOR_PALETTE[name]["hex"]


def classify_bgr_color(bgr) -> Tuple[str, str]:
    """Convenience: classify an 8-bit BGR triplet."""
    return classify_cielab_color(bgr_to_cielab(np.asarray(bgr, dtype=np.uint8).reshape(3)))


def color_family(name: str) -> str:
    """
    Coarse color family of a color name: "neutral-dark", "neutral-mid", "neutral-light", "blue", "teal",
    "green", "yellow", "orange", "brown", "red", "pink", "purple", plus "skin" (bare legs), "none"
    (occluded) and "unknown". Tolerant of free-text names such as "Blue" or "Dark Gray".
    """
    if not name:
        return "unknown"
    if name in COLOR_FAMILIES:
        return COLOR_FAMILIES[name]
    key = name.strip().lower()
    for canonical, fam in COLOR_FAMILIES.items():
        if canonical.lower() == key:
            return fam
    for kw, fam in _FAMILY_KEYWORDS:
        if kw in key:
            return fam
    return "unknown"


def palette_lab(name: str) -> Optional[np.ndarray]:
    """Reference Lab of a palette color name (case-insensitive), or None when the name is not a color."""
    if not name:
        return None
    entry = FORENSIC_COLOR_PALETTE.get(name)
    if entry is None:
        key = name.strip().lower()
        for canonical, data in FORENSIC_COLOR_PALETTE.items():
            if canonical.lower() == key:
                entry = data
                break
    if entry is None:
        return None
    return np.array(entry["lab"], dtype=np.float32)
