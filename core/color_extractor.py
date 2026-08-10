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
        Segment upper and lower body regions.
        
        Args:
            crop: Cropped person image
            pose_keypoints: Pose keypoints
            pose_confidences: Pose confidences
            bbox: Original bounding box
            
        Returns:
            Dictionary with 'upper_body' and 'lower_body' regions
        """
        h, w = crop.shape[:2]
        
        # Default segmentation: split at 45% height
        split_y = int(h * 0.45)
        
        upper_body = crop[:split_y, :]
        lower_body = crop[split_y:, :]
        
        # If pose keypoints available, use them for better segmentation
        if pose_keypoints is not None and pose_confidences is not None:
            try:
                # MediaPipe keypoint indices
                left_shoulder_idx = 11
                right_shoulder_idx = 12
                left_hip_idx = 23
                right_hip_idx = 24
                
                # Check if keypoints are confident
                if (pose_confidences[left_shoulder_idx] > 0.5 and
                    pose_confidences[right_shoulder_idx] > 0.5 and
                    pose_confidences[left_hip_idx] > 0.5 and
                    pose_confidences[right_hip_idx] > 0.5):
                    
                    # Get hip position in crop coordinates
                    hip_y = (pose_keypoints[left_hip_idx, 1] + 
                            pose_keypoints[right_hip_idx, 1]) / 2
                    
                    # Convert to crop coordinates
                    hip_y_crop = int(hip_y * h)
                    
                    # Adjust split point
                    if 0 < hip_y_crop < h:
                        upper_body = crop[:hip_y_crop, :]
                        lower_body = crop[hip_y_crop:, :]
                        
            except Exception as e:
                logger.debug(f"Pose-based segmentation failed, using default: {e}")
        
        return {
            'upper_body': upper_body,
            'lower_body': lower_body
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
            # Return zero histogram
            total_bins = len(self.hsv_channels) * self.hist_bins
            return np.zeros(total_bins, dtype=np.float32)
        
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        histograms = []
        
        for channel in self.hsv_channels:
            # Define range based on channel
            if channel == 0:  # Hue
                ranges = [0, 180]
            else:  # Saturation or Value
                ranges = [0, 256]
            
            # Compute histogram
            hist = cv2.calcHist(
                [hsv],
                [channel],
                None,
                [self.hist_bins],
                ranges
            )
            
            # Normalize
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)
        
        # Concatenate histograms
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
            # Resize for faster processing
            small_region = cv2.resize(region, (64, 64))
            
            # Reshape for k-means
            pixels = small_region.reshape(-1, 3).astype(np.float32)
            
            # K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(
                pixels,
                min(top_k, len(pixels)),
                None,
                criteria,
                10,
                cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Convert centers to integers
            centers = centers.astype(np.uint8)
            
            # Sort by cluster size
            unique_labels, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]
            
            # Get top-k dominant colors
            dominant_colors = []
            for i in sorted_indices[:top_k]:
                color = tuple(centers[i].tolist())
                dominant_colors.append(color)
            
            return dominant_colors
            
        except Exception as e:
            logger.debug(f"Dominant color extraction failed: {e}")
            # Fallback: return average color
            avg_color = tuple(map(int, cv2.mean(region)[:3]))
            return [avg_color] * top_k
    
    def _calculate_confidence(self, regions: Dict[str, np.ndarray]) -> float:
        """
        Calculate confidence based on region sizes.
        
        Args:
            regions: Dictionary of body regions
            
        Returns:
            Confidence score between 0 and 1
        """
        upper_size = regions['upper_body'].size
        lower_size = regions['lower_body'].size
        total_size = upper_size + lower_size
        
        if total_size == 0:
            return 0.0
        
        # Penalize if one region is too small
        min_region_ratio = min(upper_size, lower_size) / total_size
        
        # Confidence is higher when both regions have reasonable sizes
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
        """
        Convert color features to a feature vector for fusion.
        
        Args:
            color_features: ColorFeatures object
            
        Returns:
            Feature vector (96-dimensional)
        """
        # Concatenate upper and lower body histograms
        combined = np.concatenate([
            color_features.upper_body_histogram,
            color_features.lower_body_histogram
        ])
        
        # Pad or truncate to 96 dimensions
        target_dim = 96
        if len(combined) < target_dim:
            combined = np.pad(combined, (0, target_dim - len(combined)))
        elif len(combined) > target_dim:
            combined = combined[:target_dim]
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined
    
    def compute_similarity(self, features1: np.ndarray,
                         features2: np.ndarray) -> float:
        """
        Compute histogram similarity using correlation.
        
        Args:
            features1: First color feature vector
            features2: Second color feature vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize features
        feat1_norm = features1 / (np.linalg.norm(features1) + 1e-6)
        feat2_norm = features2 / (np.linalg.norm(features2) + 1e-6)
        
        # Compute correlation
        correlation = np.corrcoef(feat1_norm, feat2_norm)[0, 1]
        
        # Ensure result is between 0 and 1
        similarity = max(0.0, correlation)
        
        return float(similarity)
