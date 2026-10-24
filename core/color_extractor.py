"""
Color extractor module for MemoryTrack system.
Implements HSV color histogram extraction from clothing regions.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ColorFeatures:
    """Color features extracted from a person."""
    upper_body_histogram: np.ndarray
    lower_body_histogram: np.ndarray
    dominant_colors_upper: List[Tuple[int, int, int]]
    dominant_colors_lower: List[Tuple[int, int, int]]
    confidence: float

    @property
    def upper_body_hist(self) -> np.ndarray:
        return self.upper_body_histogram

    @property
    def lower_body_hist(self) -> np.ndarray:
        return self.lower_body_histogram



class ColorExtractor:
    """HSV color histogram extraction from clothing regions."""
    
    def __init__(self, hist_bins: int = 16,
                 regions: Optional[List[str]] = None,
                 hsv_channels: Optional[List[int]] = None):
        """
        Initialize color extractor.
        
        Args:
            hist_bins: Number of histogram bins per channel
            regions: Body regions to extract ('upper_body', 'lower_body')
            hsv_channels: HSV channels to use (0=H, 1=S, 2=V)
        """
        self.hist_bins = hist_bins
        self.regions = regions if regions else ['upper_body', 'lower_body']
        self.hsv_channels = hsv_channels if hsv_channels else [0, 1, 2]
        
        logger.info(f"ColorExtractor initialized with {hist_bins} bins")
    
    def extract(self, frame: np.ndarray,
                bbox: Tuple[float, float, float, float],
                pose_keypoints: Optional[np.ndarray] = None,
                pose_confidences: Optional[np.ndarray] = None) -> ColorFeatures:
        """
        Extract color features from frame.
        
        Args:
            frame: Input frame (BGR format)
            bbox: Bounding box (x1, y1, x2, y2)
            pose_keypoints: Optional pose keypoints for region segmentation
            pose_confidences: Optional pose keypoint confidences
            
        Returns:
            ColorFeatures object
        """
        try:
            # Crop to bounding box
            x1, y1, x2, y2 = bbox
            person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            
            if person_crop.size == 0:
                logger.warning("Empty crop for color extraction")
                return self._get_default_features()
            
            # Segment upper and lower body regions
            regions = self._segment_body_regions(
                person_crop,
                pose_keypoints,
                pose_confidences,
                bbox
            )
            
            # Extract histograms for each region
            upper_hist = self._extract_histogram(regions['upper_body'])
            lower_hist = self._extract_histogram(regions['lower_body'])
            
            # Extract dominant colors
            upper_colors = self._extract_dominant_colors(regions['upper_body'], top_k=3)
            lower_colors = self._extract_dominant_colors(regions['lower_body'], top_k=3)
            
            # Calculate confidence based on region sizes
            confidence = self._calculate_confidence(regions)
            
            return ColorFeatures(
                upper_body_histogram=upper_hist,
                lower_body_histogram=lower_hist,
                dominant_colors_upper=upper_colors,
                dominant_colors_lower=lower_colors,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            return self._get_default_features()
    
    def _segment_body_regions(self, crop: np.ndarray,
                             pose_keypoints: Optional[np.ndarray],
                             pose_confidences: Optional[np.ndarray],
                             bbox: Tuple[float, float, float, float]) -> Dict[str, np.ndarray]:
        """
        Segment upper and lower body regions with tight torso and leg isolation.
        
        Args:
            crop: Cropped person image
            pose_keypoints: Pose keypoints (33, 3) normalized [0, 1] relative to crop
            pose_confidences: Pose confidences
            bbox: Original bounding box
            
        Returns:
            Dictionary with 'upper_body' and 'lower_body' regions
        """
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return {'upper_body': crop, 'lower_body': crop}

        # Default fallback: central 60% of crop to avoid background edges/tables
        upper_x1, upper_x2 = int(w * 0.20), int(w * 0.80)
        upper_y1, upper_y2 = int(h * 0.15), int(h * 0.50)
        
        lower_x1, lower_x2 = int(w * 0.20), int(w * 0.80)
        lower_y1, lower_y2 = int(h * 0.52), int(h * 0.95)

        upper_body = crop[upper_y1:upper_y2, upper_x1:upper_x2]
        lower_body = crop[lower_y1:lower_y2, lower_x1:lower_x2]

        if pose_keypoints is not None and pose_confidences is not None:
            try:
                # MediaPipe keypoint indices
                left_shoulder_idx = 11
                right_shoulder_idx = 12
                left_hip_idx = 23
                right_hip_idx = 24
                left_knee_idx = 25
                right_knee_idx = 26

                conf_shoulders = min(pose_confidences[left_shoulder_idx], pose_confidences[right_shoulder_idx])
                conf_hips = min(pose_confidences[left_hip_idx], pose_confidences[right_hip_idx])

                if conf_shoulders > 0.20 and conf_hips > 0.20:
                    # Shoulder coordinates
                    ls_x, ls_y = pose_keypoints[left_shoulder_idx, 0], pose_keypoints[left_shoulder_idx, 1]
                    rs_x, rs_y = pose_keypoints[right_shoulder_idx, 0], pose_keypoints[right_shoulder_idx, 1]
                    
                    # Hip coordinates
                    lh_x, lh_y = pose_keypoints[left_hip_idx, 0], pose_keypoints[left_hip_idx, 1]
                    rh_x, rh_y = pose_keypoints[right_hip_idx, 0], pose_keypoints[right_hip_idx, 1]

                    # Isolate Torso polygon bounding box
                    torso_min_x = min(ls_x, rs_x, lh_x, rh_x)
                    torso_max_x = max(ls_x, rs_x, lh_x, rh_x)
                    torso_min_y = min(ls_y, rs_y)
                    torso_max_y = max(lh_y, rh_y)

                    torso_w = torso_max_x - torso_min_x
                    torso_h = torso_max_y - torso_min_y
                    
                    if torso_w > 0.05 and torso_h > 0.05:
                        # Inset 8% horizontally and 5% vertically to avoid arms, neck skin and background
                        tx1 = int(np.clip((torso_min_x + 0.08 * torso_w) * w, 0, w - 1))
                        tx2 = int(np.clip((torso_max_x - 0.08 * torso_w) * w, tx1 + 5, w))
                        ty1 = int(np.clip((torso_min_y + 0.05 * torso_h) * h, 0, h - 1))
                        ty2 = int(np.clip((torso_max_y - 0.02 * torso_h) * h, ty1 + 5, h))

                        torso_crop = crop[ty1:ty2, tx1:tx2]
                        if torso_crop.size > 100:
                            upper_body = torso_crop

                    # Isolate Lower Body / Legs / Pants region strictly centered on the hips
                    hip_center_y = (lh_y + rh_y) / 2.0
                    hip_min_x = min(lh_x, rh_x)
                    hip_max_x = max(lh_x, rh_x)
                    hip_span = max(hip_max_x - hip_min_x, 0.15)
                    
                    knee_y = 0.95
                    if pose_confidences[left_knee_idx] > 0.20 and pose_confidences[right_knee_idx] > 0.20:
                        knee_y = max(pose_keypoints[left_knee_idx, 1], pose_keypoints[right_knee_idx, 1])
                    
                    leg_y1 = int(np.clip((hip_center_y + 0.02) * h, 0, h - 1))
                    leg_y2 = int(np.clip(min(knee_y + 0.08, 0.98) * h, leg_y1 + 5, h))

                    # Bounded to the hip/leg column to completely ignore background tables/chairs
                    leg_x1 = int(np.clip((hip_min_x - 0.08 * hip_span) * w, 0, w - 1))
                    leg_x2 = int(np.clip((hip_max_x + 0.08 * hip_span) * w, leg_x1 + 5, w))

                    lower_crop = crop[leg_y1:leg_y2, leg_x1:leg_x2]
                    if lower_crop.size > 100:
                        lower_body = lower_crop

            except Exception as e:
                logger.debug(f"Pose-based segmentation error: {e}")

        return {
            'upper_body': upper_body if upper_body.size > 0 else crop,
            'lower_body': lower_body if lower_body.size > 0 else crop
        }
    
    def _extract_histogram(self, region: np.ndarray) -> np.ndarray:
        """
        Extract HSV histogram from region.
        
        Args:
            region: Image region
            
        Returns:
            Flattened histogram vector
        """
        if region.size == 0:
            total_bins = len(self.hsv_channels) * self.hist_bins
            return np.zeros(total_bins, dtype=np.float32)
        
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        histograms = []
        
        for channel in self.hsv_channels:
            if channel == 0:  # Hue
                ranges = [0, 180]
            else:  # Saturation or Value
                ranges = [0, 256]
            
            hist = cv2.calcHist(
                [hsv],
                [channel],
                None,
                [self.hist_bins],
                ranges
            )
            
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)
        
        combined_hist = np.concatenate(histograms)
        return combined_hist.astype(np.float32)
    
    def _extract_dominant_colors(self, region: np.ndarray,
                                top_k: int = 3) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors using k-means clustering.
        
        Args:
            region: Image region
            top_k: Number of dominant colors to extract
            
        Returns:
            List of (B, G, R) tuples
        """
        if region.size == 0:
            return [(0, 0, 0)] * top_k
        
        try:
            small_region = cv2.resize(region, (64, 64))
            pixels = small_region.reshape(-1, 3).astype(np.float32)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(
                pixels,
                min(top_k, len(pixels)),
                None,
                criteria,
                10,
                cv2.KMEANS_RANDOM_CENTERS
            )
            
            centers = centers.astype(np.uint8)
            unique_labels, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]
            
            dominant_colors = []
            for i in sorted_indices[:top_k]:
                color = tuple(centers[i].tolist())
                dominant_colors.append(color)
            
            return dominant_colors
            
        except Exception as e:
            logger.debug(f"Dominant color extraction failed: {e}")
            avg_color = tuple(map(int, cv2.mean(region)[:3]))
            return [avg_color] * top_k
    
    def _calculate_confidence(self, regions: Dict[str, np.ndarray]) -> float:
        """Calculate confidence based on region sizes."""
        upper_size = regions['upper_body'].size
        lower_size = regions['lower_body'].size
        total_size = upper_size + lower_size
        
        if total_size == 0:
            return 0.0
        
        min_region_ratio = min(upper_size, lower_size) / total_size
        confidence = min(1.0, min_region_ratio * 2)
        return float(confidence)
    
    def _get_default_features(self) -> ColorFeatures:
        """Return default zero features when extraction fails."""
        total_bins = len(self.hsv_channels) * self.hist_bins
        
        return ColorFeatures(
            upper_body_histogram=np.zeros(total_bins, dtype=np.float32),
            lower_body_histogram=np.zeros(total_bins, dtype=np.float32),
            dominant_colors_upper=[(0, 0, 0)] * 3,
            dominant_colors_lower=[(0, 0, 0)] * 3,
            confidence=0.0
        )
    
    def extract_feature_vector(self, color_features: ColorFeatures) -> np.ndarray:
        """Convert color features to a feature vector for fusion."""
        combined = np.concatenate([
            color_features.upper_body_histogram,
            color_features.lower_body_histogram
        ])
        
        target_dim = 96
        if len(combined) < target_dim:
            combined = np.pad(combined, (0, target_dim - len(combined)))
        elif len(combined) > target_dim:
            combined = combined[:target_dim]
        
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined
    
    def compute_similarity(self, features1: np.ndarray,
                         features2: np.ndarray) -> float:
        """Compute histogram similarity using correlation."""
        feat1_norm = features1 / (np.linalg.norm(features1) + 1e-6)
        feat2_norm = features2 / (np.linalg.norm(features2) + 1e-6)
        correlation = np.corrcoef(feat1_norm, feat2_norm)[0, 1]
        similarity = max(0.0, correlation)
        return float(similarity)

    @classmethod
    def classify_region_color(cls, region_bgr: np.ndarray) -> str:
        """
        Robustly classify clothing color from an image crop using skin filtering and HSV distribution.
        """
        if region_bgr is None or region_bgr.size < 20:
            return "Unknown"

        try:
            hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0].flatten()
            s = hsv[:, :, 1].flatten()
            v = hsv[:, :, 2].flatten()

            # Skin detection filter (skip face/arms/skin pixels from clothing stats)
            skin_mask = (h <= 25) & (s >= 35) & (s <= 175) & (v >= 60)
            non_skin_count = np.sum(~skin_mask)
            
            if non_skin_count > (len(h) * 0.25):
                h = h[~skin_mask]
                s = s[~skin_mask]
                v = v[~skin_mask]

            total_pixels = len(v)
            if total_pixels == 0:
                return "Unknown"

            # 1. White / Off-White Detection (Calibrated for indoor lighting)
            # High brightness with low saturation
            white_mask = (s < 45) & (v >= 115)
            white_ratio = np.sum(white_mask) / total_pixels

            # 2. Black Detection
            # Very low brightness
            black_mask = (v < 55) & ((s < 90) | (v < 40))
            black_ratio = np.sum(black_mask) / total_pixels

            # 3. Gray Detection
            gray_mask = (s < 42) & (v >= 55) & (v < 115)
            gray_ratio = np.sum(gray_mask) / total_pixels

            # Decision logic with pattern handling (e.g. white shirt with black texture)
            if white_ratio >= 0.32:
                return "White"
            if black_ratio >= 0.35:
                return "Black"
            if gray_ratio >= 0.40:
                mean_v = np.mean(v[gray_mask])
                return "Dark Gray" if mean_v < 85 else "Gray"

            # 4. Chromatic Color Evaluation
            chromatic_mask = ~white_mask & ~black_mask & ~gray_mask
            if np.sum(chromatic_mask) < (total_pixels * 0.20):
                # Fallback to dominant achromatic
                if white_ratio >= black_ratio and white_ratio >= gray_ratio:
                    return "White"
                elif black_ratio >= gray_ratio:
                    return "Black"
                else:
                    return "Gray"

            med_h = float(np.median(h[chromatic_mask]))
            med_s = float(np.median(s[chromatic_mask]))
            med_v = float(np.median(v[chromatic_mask]))

            if med_h <= 10 or med_h >= 170:
                return "Burgundy / Dark Red" if med_v < 90 else "Red"
            elif 11 <= med_h <= 25:
                return "Brown / Tan" if med_v < 100 else "Orange"
            elif 26 <= med_h <= 35:
                return "Khaki / Beige" if med_s < 85 else "Yellow"
            elif 36 <= med_h <= 85:
                return "Olive / Dark Green" if med_v < 90 else "Green"
            elif 86 <= med_h <= 100:
                return "Teal / Cyan"
            elif 101 <= med_h <= 135:
                return "Navy Blue" if med_v < 100 else "Blue"
            elif 136 <= med_h <= 155:
                return "Purple / Violet"
            else:
                return "Pink"

        except Exception as e:
            logger.debug(f"Color classification error: {e}")
            return "Unknown"

    @staticmethod
    def bgr_to_color_name(bgr: Tuple[int, int, int]) -> str:
        """Convert BGR color tuple to human-readable color name."""
        try:
            b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
            mat = np.uint8([[[b, g, r]]])
            return ColorExtractor.classify_region_color(mat)
        except Exception:
            return "Unknown"

    def get_clothing_description(self, color_features: Optional[ColorFeatures],
                                 upper_crop: Optional[np.ndarray] = None,
                                 lower_crop: Optional[np.ndarray] = None) -> Dict[str, str]:
        """Extract human-readable clothing descriptions for upper and lower body."""
        # Prefer direct pixel crop classification if provided
        if upper_crop is not None and upper_crop.size > 20:
            upper_name = self.classify_region_color(upper_crop)
        elif color_features and color_features.dominant_colors_upper:
            upper_name = self.bgr_to_color_name(color_features.dominant_colors_upper[0])
        else:
            upper_name = "Unknown"

        if lower_crop is not None and lower_crop.size > 20:
            lower_name = self.classify_region_color(lower_crop)
        elif color_features and color_features.dominant_colors_lower:
            lower_name = self.bgr_to_color_name(color_features.dominant_colors_lower[0])
        else:
            lower_name = "Unknown"

        return {
            'upper_color': upper_name,
            'lower_color': lower_name,
            'description': f"Wearing {upper_name} top and {lower_name} bottom"
        }


