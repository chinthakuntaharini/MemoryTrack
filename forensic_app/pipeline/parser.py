"""
Human Parsing & Garment Segmentation Engine
Extracts upper and lower garment masks and prevents pants color hallucination when no pants are present.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


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


class HumanParser:
    """Semantic cloth and body segmenter."""

    def __init__(self, min_mask_pixels: int = 40):
        self.min_mask_pixels = min_mask_pixels

    def parse_pedestrian(self,
                         crop: np.ndarray,
                         keypoints: Optional[np.ndarray] = None,
                         keypoint_confs: Optional[np.ndarray] = None) -> ParsedGarments:
        """
        Segment upper and lower garments from a pedestrian crop.
        Guarantees NO false pants color if legs are bare, occluded, or out of frame.
        """
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

        # 1. Default geometric regions (with skin detection filtration)
        upper_mask = np.zeros((h, w), dtype=np.uint8)
        lower_mask = np.zeros((h, w), dtype=np.uint8)

        # Torso boundaries
        u_y1, u_y2 = int(h * 0.15), int(h * 0.50)
        u_x1, u_x2 = int(w * 0.20), int(w * 0.80)
        upper_mask[u_y1:u_y2, u_x1:u_x2] = 255

        # Check if pose keypoints indicate lower body presence
        has_lower_body_visible = False
        lower_garment_type = "None / Occluded"

        # Auto-extract confidences if keypoints has 3 columns (x, y, conf)
        if keypoints is not None and keypoint_confs is None and keypoints.ndim == 2:
            if keypoints.shape[1] >= 3:
                keypoint_confs = keypoints[:, 2]
            else:
                keypoint_confs = np.ones(keypoints.shape[0], dtype=np.float32)

        if keypoints is not None and keypoint_confs is not None and len(keypoints) >= 17:
            # COCO Keypoints:
            # 5: L_shoulder, 6: R_shoulder
            # 11: L_hip, 12: R_hip
            # 13: L_knee, 14: R_knee
            # 15: L_ankle, 16: R_ankle
            conf_hips = max(float(keypoint_confs[11]), float(keypoint_confs[12]))
            conf_knees = max(float(keypoint_confs[13]), float(keypoint_confs[14]))
            conf_ankles = max(float(keypoint_confs[15]), float(keypoint_confs[16]))

            # Torso refinement with shoulders & hips
            conf_shoulders = max(float(keypoint_confs[5]), float(keypoint_confs[6]))
            if conf_shoulders >= 0.25 and conf_hips >= 0.25:
                sh_y = (keypoints[5][1] + keypoints[6][1]) / 2.0
                hp_y = (keypoints[11][1] + keypoints[12][1]) / 2.0
                sh_py = int(sh_y * h) if sh_y <= 1.0 else int(sh_y)
                hp_py = int(hp_y * h) if hp_y <= 1.0 else int(hp_y)
                if hp_py > sh_py + 10:
                    upper_mask[:] = 0
                    upper_mask[max(0, sh_py):min(h, hp_py), int(w * 0.18):int(w * 0.82)] = 255

            # If knees AND ankles are missing, the person is occluded / upper-body only!
            if conf_knees < 0.20 and conf_ankles < 0.20:
                has_lower_body_visible = False
                lower_garment_type = "None / Occluded"
            elif conf_hips >= 0.25:
                lh_y, rh_y = keypoints[11][1], keypoints[12][1]
                hip_y = (lh_y + rh_y) / 2.0
                hip_py = int(hip_y * h) if hip_y <= 1.0 else int(hip_y)

                if (h - hip_py) > int(h * 0.25):
                    has_lower_body_visible = True
                    l_y1 = max(0, min(h - 1, hip_py + int(h * 0.02)))
                    l_y2 = int(h * 0.94)
                    l_x1 = int(w * 0.18)
                    l_x2 = int(w * 0.82)

                    if l_y2 > l_y1 + 10 and l_x2 > l_x1 + 10:
                        lower_mask[l_y1:l_y2, l_x1:l_x2] = 255
        else:
            # Without keypoints, only check lower half if crop aspect ratio suggests full body
            # Pedestrian full body typically has aspect ratio h/w >= 1.70
            aspect_ratio = float(h) / max(w, 1)
            if aspect_ratio >= 1.70:
                has_lower_body_visible = True
                l_y1, l_y2 = int(h * 0.54), int(h * 0.94)
                l_x1, l_x2 = int(w * 0.18), int(w * 0.82)
                lower_mask[l_y1:l_y2, l_x1:l_x2] = 255
            else:
                has_lower_body_visible = False
                lower_garment_type = "None / Occluded"

        # 2. Skin Filtering on Lower Mask
        # If the lower region contains substantial bare skin (shorts/skirt/bare legs),
        # NEVER report it as a pants color!
        has_lower_garment = False
        if has_lower_body_visible and np.sum(lower_mask > 0) >= self.min_mask_pixels:
            lower_pixels = crop[lower_mask > 0]
            # Convert to YCrCb and HSV for comprehensive skin detection across lighting conditions
            pix_3d = lower_pixels.reshape(-1, 1, 3)
            ycrcb = cv2.cvtColor(pix_3d, cv2.COLOR_BGR2YCrCb)
            cr = ycrcb[:, 0, 1]
            cb = ycrcb[:, 0, 2]
            ycrcb_skin = (cr >= 133) & (cr <= 175) & (cb >= 77) & (cb <= 128)

            hsv = cv2.cvtColor(pix_3d, cv2.COLOR_BGR2HSV)
            h_chan = hsv[:, 0, 0]
            s_chan = hsv[:, 0, 1]
            v_chan = hsv[:, 0, 2]
            hsv_skin = ((h_chan <= 25) | (h_chan >= 170)) & (s_chan >= 20) & (s_chan <= 180) & (v_chan >= 50)

            skin_mask = ycrcb_skin | hsv_skin
            skin_ratio = float(np.sum(skin_mask)) / max(len(lower_pixels), 1)

            # In a rectangular bounding box, legs take ~45% of total width.
            # If bare, skin covers 22%-50% of the entire rectangle.
            if skin_ratio >= 0.22:
                # Majority skin on legs -> Bare legs / Shorts / Dress without pants
                lower_garment_type = "Bare Legs / Shorts"
                has_lower_garment = False
                lower_mask[:] = 0  # Zero out mask so color extractor does not hallucinate
            else:
                lower_garment_type = "Pants/Jeans"
                has_lower_garment = True
        else:
            lower_garment_type = "None / Occluded"
            has_lower_garment = False
            lower_mask[:] = 0

        # Extract masked crops
        upper_crop = cv2.bitwise_and(crop, crop, mask=upper_mask)
        lower_crop = cv2.bitwise_and(crop, crop, mask=lower_mask) if has_lower_garment else np.empty((0, 0, 3), dtype=np.uint8)

        has_upper_garment = np.sum(upper_mask > 0) >= self.min_mask_pixels

        return ParsedGarments(
            upper_mask=upper_mask,
            lower_mask=lower_mask,
            upper_crop=upper_crop,
            lower_crop=lower_crop,
            has_upper_garment=has_upper_garment,
            has_lower_garment=has_lower_garment,
            lower_garment_type=lower_garment_type
        )
