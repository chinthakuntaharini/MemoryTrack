"""
Forensic Attribute Extraction Engine
Extracts illumination-invariant garment colors (CIELAB K-Means) and carried accessories.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans

from forensic_app.utils.color_naming import bgr_to_cielab, classify_cielab_color, FORENSIC_COLOR_PALETTE
from forensic_app.pipeline.parser import ParsedGarments


class AttributeExtractor:
    """Extracts dominant CIELAB garment colors and carried accessories."""

    def __init__(self,
                 k_clusters: int = 3,
                 min_luminance: float = 20.0,
                 max_luminance: float = 95.0):
        self.k_clusters = k_clusters
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance

        # COCO accessory class IDs and names
        # 24: backpack, 26: handbag, 28: suitcase, 25: umbrella, 39: bottle
        self.coco_accessory_map = {
            24: "backpack",
            26: "handbag",
            28: "suitcase",
            25: "umbrella",
            39: "bottle"
        }

    def extract_attributes(self,
                           frame: np.ndarray,
                           person_bbox: Tuple[int, int, int, int],
                           parsed_garments: ParsedGarments,
                           yolo_model=None,
                           keypoints: Optional[np.ndarray] = None,
                           keypoints_crop: Optional[np.ndarray] = None,
                           initial_accessories: Optional[List[str]] = None) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
        """
        Extract upper garment color, lower garment color, and carried accessories.
        Returns: (upper_garment_dict, lower_garment_dict, accessories_list)
        """
        # 1. Upper Garment Color
        if parsed_garments.has_upper_garment and parsed_garments.upper_mask is not None:
            upper_name, upper_hex = self._extract_dominant_cielab_color(
                parsed_garments.upper_crop, parsed_garments.upper_mask
            )
        else:
            upper_name, upper_hex = "Unknown", "#808080"

        # 2. Lower Garment Color (Strict presence enforcement)
        # Prevents hallucinating pants colors when person has bare legs, shorts, or is occluded
        if parsed_garments.has_lower_garment and parsed_garments.lower_mask is not None and np.sum(parsed_garments.lower_mask > 0) > 20:
            lower_name, lower_hex = self._extract_dominant_cielab_color(
                parsed_garments.lower_crop, parsed_garments.lower_mask
            )
        else:
            lower_name = parsed_garments.lower_garment_type
            lower_hex = "#D4B996" if "Bare" in lower_name else "#404040"

        # 3. Accessory Extraction (Multi-Scale Context + Frame Detections + Headwear)
        accessories = self._detect_accessories(
            frame, person_bbox, yolo_model, keypoints, keypoints_crop, initial_accessories
        )

        upper_dict = {"color": upper_name, "hex": upper_hex}
        lower_dict = {"color": lower_name, "hex": lower_hex}

        return upper_dict, lower_dict, accessories

    def _extract_dominant_cielab_color(self, crop: np.ndarray, mask: np.ndarray) -> Tuple[str, str]:
        """
        Extract dominant color using K-Means in CIELAB space, filtering shadow & highlight extremes.
        """
        if crop is None or crop.size == 0 or mask is None:
            return "Unknown", "#808080"

        pixels = crop[mask > 0]
        if len(pixels) < 15 or pixels.ndim != 2 or pixels.shape[1] != 3:
            return "Unknown", "#808080"

        # Convert to CIELAB space using safe wrapper
        lab_pixels = bgr_to_cielab(pixels)
        if len(lab_pixels) == 0:
            return "Unknown", "#808080"

        # Discard shadow pixels (L* < 20) and highlights (L* > 95) as per SKILL.md
        valid_idx = (lab_pixels[:, 0] >= self.min_luminance) & (lab_pixels[:, 0] <= self.max_luminance)
        filtered_lab = lab_pixels[valid_idx]

        # If too dark (almost all shadow), label as Black
        if len(filtered_lab) < (len(lab_pixels) * 0.20):
            mean_L = np.mean(lab_pixels[:, 0])
            if mean_L < self.min_luminance:
                return "Black", FORENSIC_COLOR_PALETTE["Black"]["hex"]
            elif mean_L > self.max_luminance:
                return "White", FORENSIC_COLOR_PALETTE["White"]["hex"]
            filtered_lab = lab_pixels

        # Run K-Means (k=3) clustering to find dominant color center
        n_clusters = min(self.k_clusters, len(filtered_lab))
        if n_clusters <= 0:
            return "Unknown", "#808080"

        try:
            kmeans = KMeans(n_clusters=n_clusters, n_init=3, max_iter=20, random_state=42)
            labels = kmeans.fit_predict(filtered_lab)
            # Find cluster with highest frequency
            counts = np.bincount(labels)
            dominant_cluster_idx = int(np.argmax(counts))
            dominant_centroid_lab = kmeans.cluster_centers_[dominant_cluster_idx]

            return classify_cielab_color(dominant_centroid_lab)
        except Exception:
            # Fallback to median
            med_lab = np.median(filtered_lab, axis=0)
            return classify_cielab_color(med_lab)

    def _detect_accessories(self,
                           frame: np.ndarray,
                           person_bbox: Tuple[int, int, int, int],
                           yolo_model=None,
                           keypoints: Optional[np.ndarray] = None,
                           keypoints_crop: Optional[np.ndarray] = None,
                           initial_accessories: Optional[List[str]] = None) -> List[str]:
        """
        Detect accessories carried by or attached to the person.
        Combines full-frame associations, ROI object detection, and crown/headwear inspection.
        """
        detected_accessories = list(initial_accessories) if initial_accessories else []
        if frame is None:
            return detected_accessories

        px1, py1, px2, py2 = person_bbox
        pw = px2 - px1
        ph = py2 - py1

        if pw < 10 or ph < 10:
            return detected_accessories

        # Expand region of interest slightly (accessories often hang at the side/back)
        roi_x1 = max(0, px1 - int(pw * 0.20))
        roi_y1 = max(0, py1 - int(ph * 0.10))
        roi_x2 = min(frame.shape[1], px2 + int(pw * 0.20))
        roi_y2 = min(frame.shape[0], py2 + int(ph * 0.10))

        person_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Only run ROI detection if we haven't already identified accessories and model is available
        if yolo_model is not None and person_roi.size > 0 and person_roi.ndim == 3 and person_roi.shape[2] == 3:
            try:
                results = yolo_model(person_roi, conf=0.18, iou=0.45, verbose=False)
                for r in results:
                    if r.boxes is None:
                        continue
                    for b in r.boxes:
                        cls_id = int(b.cls[0])
                        if cls_id in self.coco_accessory_map:
                            acc_name = self.coco_accessory_map[cls_id]
                            if acc_name not in detected_accessories:
                                detected_accessories.append(acc_name)
            except Exception:
                pass

        # Headwear / Cap inspection (inspect crown above eyes/nose)
        kpts_for_head = keypoints_crop if keypoints_crop is not None else keypoints
        if kpts_for_head is not None and len(kpts_for_head) >= 17 and "cap" not in detected_accessories:
            try:
                # Nose = 0, Eyes = 1, 2
                conf_face = min(float(kpts_for_head[0][2]), float(kpts_for_head[1][2]), float(kpts_for_head[2][2])) if kpts_for_head.shape[1] > 2 else 1.0
                if conf_face >= 0.25:
                    if keypoints_crop is not None:
                        nose_py = int(kpts_for_head[0][1])
                        head_roi = frame[py1:py2, px1:px2]
                    else:
                        nose_y = float(kpts_for_head[0][1])
                        nose_py = int(nose_y - py1) if nose_y > 1.0 else int(nose_y * ph)
                        head_roi = frame[py1:py2, px1:px2]

                    if 6 < nose_py < head_roi.shape[0]:
                        crown_crop = head_roi[0:nose_py, :]
                        if crown_crop.size > 50 and crown_crop.shape[0] > 6 and crown_crop.shape[1] > 6:
                            ycrcb = cv2.cvtColor(crown_crop, cv2.COLOR_BGR2YCrCb)
                            cr = ycrcb[:, :, 1]
                            cb = ycrcb[:, :, 2]
                            skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
                            skin_ratio = np.sum(skin_mask) / float(crown_crop.shape[0] * crown_crop.shape[1])
                            # If crown region is non-skin fabric (cap/hat)
                            if skin_ratio < 0.20 and crown_crop.shape[0] > 10:
                                detected_accessories.append("cap")
            except Exception:
                pass

        return detected_accessories
