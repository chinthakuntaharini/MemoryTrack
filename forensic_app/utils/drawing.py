"""
Drawing & Visual Overlay Utilities for Forensic Re-ID
Renders bounding boxes, forensic entity tags, 17-keypoint skeletons, and cloth masks.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any

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
    "emerald": (80, 200, 120),   # Confirmed active
    "amber": (36, 179, 234),     # Carried accessories
    "purple": (220, 100, 180),   # Biometrics
    "navy_dark": (30, 25, 20),   # Background tag box
    "white": (255, 255, 255),
    "red": (60, 60, 230),
    "skeleton_joint": (0, 255, 255),
    "skeleton_limb": (255, 128, 0)
}


def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    """Convert HEX string (#RRGGBB) to BGR tuple."""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        return (128, 128, 128)
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (b, g, r)


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, conf_thresh: float = 0.3) -> None:
    """
    Render 17-keypoint skeleton on image frame.
    keypoints shape: (17, 2) or (17, 3) where columns are (x, y, [conf]).
    """
    if keypoints is None or len(keypoints) < 17:
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

    # Draw limb connections
    for idx1, idx2 in SKELETON_EDGES:
        if idx1 < len(pts) and idx2 < len(pts):
            if confs[idx1] >= conf_thresh and confs[idx2] >= conf_thresh:
                pt1 = pts[idx1]
                pt2 = pts[idx2]
                if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                    cv2.line(frame, pt1, pt2, PALETTE["skeleton_limb"], 2, cv2.LINE_AA)

    # Draw joint points
    for idx, (px, py) in enumerate(pts):
        if confs[idx] >= conf_thresh:
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(frame, (px, py), 4, PALETTE["skeleton_joint"], -1, cv2.LINE_AA)


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
                       is_selected: bool = False) -> None:
    """
    Render a forensic HUD bounding badge around a person with color swatches and accessory chips.
    """
    x1, y1, x2, y2 = bbox
    h_frame, w_frame = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame - 1, x2), min(h_frame - 1, y2)
    w_box = x2 - x1
    h_box = y2 - y1

    if w_box <= 0 or h_box <= 0:
        return

    # Box Border & Corner Brackets
    box_color = PALETTE["cyan"] if is_selected else PALETTE["emerald"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Corner HUD brackets
    c_len = min(16, w_box // 3, h_box // 3)
    cv2.line(frame, (x1, y1), (x1 + c_len, y1), box_color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + c_len), box_color, 3)
    cv2.line(frame, (x2, y1), (x2 - c_len, y1), box_color, 3)
    cv2.line(frame, (x2, y1), (x2, y1 + c_len), box_color, 3)
    cv2.line(frame, (x1, y2), (x1 + c_len, y2), box_color, 3)
    cv2.line(frame, (x1, y2), (x1, y2 - c_len), box_color, 3)
    cv2.line(frame, (x2, y2), (x2 - c_len, y2), box_color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - c_len), box_color, 3)

    # Top Tag Header
    header_text = f"ID #{track_id} | {person_uid}"
    (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

    tag_y1 = max(0, y1 - 24)
    tag_y2 = y1
    tag_w = max(w_box, tw + 20)

    # Dark banner backdrop
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, tag_y1), (x1 + tag_w, tag_y2), PALETTE["navy_dark"], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, header_text, (x1 + 4, tag_y1 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, PALETTE["white"], 1, cv2.LINE_AA)

    # Bottom Attribute Swatches & Labels
    sub_y1 = y2
    sub_y2 = min(h_frame, y2 + 22)
    sub_w = max(w_box, 130)

    cv2.rectangle(overlay, (x1, sub_y1), (x1 + sub_w, sub_y2), PALETTE["navy_dark"], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Upper Swatch
    u_bgr = hex_to_bgr(upper_hex)
    cv2.rectangle(frame, (x1 + 4, sub_y1 + 4), (x1 + 16, sub_y1 + 16), u_bgr, -1)
    cv2.rectangle(frame, (x1 + 4, sub_y1 + 4), (x1 + 16, sub_y1 + 16), PALETTE["white"], 1)

    # Lower Swatch
    l_bgr = hex_to_bgr(lower_hex)
    cv2.rectangle(frame, (x1 + 22, sub_y1 + 4), (x1 + 34, sub_y1 + 16), l_bgr, -1)
    cv2.rectangle(frame, (x1 + 22, sub_y1 + 4), (x1 + 34, sub_y1 + 16), PALETTE["white"], 1)

    # Text summary
    acc_str = f" +{','.join(accessories[:2])}" if accessories else ""
    attr_text = f"{upper_color[:5]}/{lower_color[:5]}{acc_str}"
    cv2.putText(frame, attr_text, (x1 + 40, sub_y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, PALETTE["cyan"], 1, cv2.LINE_AA)
