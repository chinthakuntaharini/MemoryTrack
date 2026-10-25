"""
Forensic Color Naming Utility (ISCC-NBS / CIELAB)
Maps segmented garment pixels to standardized forensic color descriptors using CIELAB space.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional

# Standard forensic color reference palette in CIELAB space
# Reference colors according to forensic surveillance standards:
# ['Black', 'White', 'Navy Blue', 'Royal Blue', 'Khaki', 'Beige', 'Charcoal', 'Red', 'Maroon', 'Olive Green', 'Yellow', 'Brown']
FORENSIC_COLOR_PALETTE = {
    "Black": {
        "rgb": (20, 20, 20),
        "hex": "#141414",
        "lab": (8.0, 0.0, 0.0),
        "max_L": 25.0
    },
    "Charcoal": {
        "rgb": (54, 57, 63),
        "hex": "#36393F",
        "lab": (24.0, 0.0, -2.0),
        "L_range": (20.0, 38.0)
    },
    "White": {
        "rgb": (240, 240, 240),
        "hex": "#F0F0F0",
        "lab": (94.0, 0.0, 0.0),
        "min_L": 85.0
    },
    "Navy Blue": {
        "rgb": (15, 23, 62),
        "hex": "#0F173E",
        "lab": (10.0, 4.0, -22.0)
    },
    "Royal Blue": {
        "rgb": (29, 78, 216),
        "hex": "#1D4ED8",
        "lab": (36.0, 24.0, -68.0)
    },
    "Beige": {
        "rgb": (224, 212, 187),
        "hex": "#E0D4BB",
        "lab": (85.0, 0.5, 14.0)
    },
    "Khaki": {
        "rgb": (189, 172, 129),
        "hex": "#BDAC81",
        "lab": (70.0, 1.0, 25.0)
    },
    "Brown": {
        "rgb": (101, 55, 30),
        "hex": "#65371E",
        "lab": (28.0, 18.0, 24.0)
    },
    "Maroon": {
        "rgb": (115, 16, 29),
        "hex": "#73101D",
        "lab": (22.0, 42.0, 22.0)
    },
    "Red": {
        "rgb": (220, 38, 38),
        "hex": "#DC2626",
        "lab": (46.0, 68.0, 47.0)
    },
    "Olive Green": {
        "rgb": (74, 93, 35),
        "hex": "#4A5D23",
        "lab": (38.0, -18.0, 30.0)
    },
    "Yellow": {
        "rgb": (234, 179, 8),
        "hex": "#EAB308",
        "lab": (76.0, 8.0, 80.0)
    }
}


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


def delta_e_cielab(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Compute Euclidean Delta-E (CIE76) color difference in CIELAB space."""
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def classify_cielab_color(lab_val: np.ndarray) -> Tuple[str, str]:
    """
    Classify a single (L, a, b) vector into standard forensic color name and hex.
    Returns (color_name, hex_code).
    """
    if lab_val is None or len(lab_val) < 3:
        return "Unknown", "#808080"

    L, a, b = float(lab_val[0]), float(lab_val[1]), float(lab_val[2])

    # Rule-based achromatic override for extreme highlights and shadows
    if L >= 88.0 and abs(a) < 10.0 and abs(b) < 14.0:
        return "White", FORENSIC_COLOR_PALETTE["White"]["hex"]

    if L <= 22.0:
        # Check if deep navy or black
        if b < -14.0 and abs(a) < 10.0:
            return "Navy Blue", FORENSIC_COLOR_PALETTE["Navy Blue"]["hex"]
        if a > 18.0 and b > 8.0:
            return "Maroon", FORENSIC_COLOR_PALETTE["Maroon"]["hex"]
        return "Black", FORENSIC_COLOR_PALETTE["Black"]["hex"]

    if 20.0 < L <= 36.0 and abs(a) < 7.0 and abs(b) < 7.0:
        return "Charcoal", FORENSIC_COLOR_PALETTE["Charcoal"]["hex"]

    # Minimum Delta-E matching across reference palette
    best_name = "Black"
    min_dist = float("inf")

    for name, data in FORENSIC_COLOR_PALETTE.items():
        ref_lab = np.array(data["lab"], dtype=np.float32)
        dist = delta_e_cielab(lab_val, ref_lab)
        if dist < min_dist:
            min_dist = dist
            best_name = name

    return best_name, FORENSIC_COLOR_PALETTE[best_name]["hex"]
