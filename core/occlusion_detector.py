"""
Occlusion detector module for MemoryTrack system.
Detects partial/full occlusion and identifies which modalities are affected.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Modalities that can be affected by occlusion
MODALITIES = ['reid', 'pose', 'color', 'accessory', 'motion']


@dataclass
class OcclusionResult:
    """Result of occlusion detection."""
    is_occluded: bool
    occlusion_level: float  # 0.0 (none) to 1.0 (fully occluded)
    occlusion_flags: Dict[str, bool] = field(default_factory=dict)
    affected_modalities: List[str] = field(default_factory=list)
    reason: str = ""


class OcclusionDetector:
    """Detect occlusion of persons and determine affected modalities."""

    def __init__(self,
                 overlap_threshold: float = 0.4,
                 aspect_ratio_range: Tuple[float, float] = (0.2, 1.5),
                 min_track_overlap: float = 0.5):
        """
        Initialize occlusion detector.

        Args:
            overlap_threshold: IoU threshold between persons to flag occlusion
            aspect_ratio_range: Valid aspect ratio range for a person bbox
            min_track_overlap: IoU threshold for partial-overlap detection
        """
        self.overlap_threshold = overlap_threshold
        self.aspect_ratio_range = aspect_ratio_range
        self.min_track_overlap = min_track_overlap

        logger.info(
            f"OcclusionDetector initialized with overlap_threshold={overlap_threshold}"
        )

    def detect(self,
               bbox: Tuple[float, float, float, float],
               other_bboxes: List[Tuple[float, float, float, float]],
               pose_confidences: Optional[np.ndarray] = None,
               keypoint_confidences: Optional[np.ndarray] = None) -> OcclusionResult:
        """
        Detect occlusion for a person given other detected persons.

        Args:
            bbox: Person bounding box (x1, y1, x2, y2)
            other_bboxes: Bounding boxes of other detected persons
            pose_confidences: Optional pose confidence values
            keypoint_confidences: Optional keypoint visibility array

        Returns:
            OcclusionResult with occlusion flags per modality
        """
        flags: Dict[str, bool] = {m: False for m in MODALITIES}
        affected: List[str] = []
        reason_parts: List[str] = []

        # 1. Person-on-person overlap (IoU)
        max_overlap = 0.0
        full_overlap = False

        for other in other_bboxes:
            iou = self._compute_iou(bbox, other)
            if iou > max_overlap:
                max_overlap = iou

        if max_overlap >= self.overlap_threshold:
            full_overlap = True
            reason_parts.append(f"overlap={max_overlap:.2f}")

        # 2. Determine modality effects
        if full_overlap:
            # Full occlusion: most appearance-based modalities degrade
            for m in ['reid', 'color', 'accessory']:
                flags[m] = True
                if m not in affected:
                    affected.append(m)
            flags['pose'] = True
            if 'pose' not in affected:
                affected.append('pose')
            # Motion may still be estimated from track history
            flags['motion'] = False
        elif max_overlap >= self.min_track_overlap:
            # Partial occlusion: some modalities partially affected
            flags['reid'] = True
            flags['color'] = True
            if 'reid' not in affected:
                affected.append('reid')
            if 'color' not in affected:
                affected.append('color')

        # 3. Pose-based occlusion detection (low keypoint visibility)
        if keypoint_confidences is not None and len(keypoint_confidences) > 0:
            visibility = float(np.mean(keypoint_confidences))
            if visibility < 0.3:
                flags['pose'] = True
                if 'pose' not in affected:
                    affected.append('pose')
                reason_parts.append(f"pose_visibility={visibility:.2f}")

        # 4. Aspect ratio sanity check indicates partial-body occlusion
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1
        if h > 0 and w > 0:
            aspect_ratio = h / w
            if (aspect_ratio < self.aspect_ratio_range[0] or
                    aspect_ratio > self.aspect_ratio_range[1]):
                # Likely a partial crop - color & pose unreliable
                flags['color'] = True
                flags['pose'] = True
                if 'color' not in affected:
                    affected.append('color')
                if 'pose' not in affected:
                    affected.append('pose')
                reason_parts.append(f"aspect_ratio={aspect_ratio:.2f}")

        # Compute overall occlusion level
        affected_count = len(affected)
        occlusion_level = min(1.0, affected_count / len(MODALITIES))
        # Boost if full overlap
        if full_overlap:
            occlusion_level = max(occlusion_level, 0.9)

        is_occluded = occlusion_level > 0.1

        reason = "; ".join(reason_parts) if reason_parts else "no occlusion detected"

        return OcclusionResult(
            is_occluded=is_occluded,
            occlusion_level=occlusion_level,
            occlusion_flags=flags,
            affected_modalities=affected,
            reason=reason
        )

    def detect_batch(self,
                     bboxes: List[Tuple[float, float, float, float]],
                     keypoint_confidences_list: Optional[List[Optional[np.ndarray]]] = None
                     ) -> Dict[int, OcclusionResult]:
        """
        Detect occlusion for multiple persons.

        Args:
            bboxes: List of person bounding boxes
            keypoint_confidences_list: Optional list of keypoint confidence arrays

        Returns:
            Dictionary of index -> OcclusionResult
        """
        results: Dict[int, OcclusionResult] = {}
        for i, bbox in enumerate(bboxes):
            other = [b for j, b in enumerate(bboxes) if j != i]
            kc = None
            if keypoint_confidences_list is not None and i < len(keypoint_confidences_list):
                kc = keypoint_confidences_list[i]
            results[i] = self.detect(bbox, other, keypoint_confidences=kc)
        return results

    def _compute_iou(self, bbox1: Tuple[float, float, float, float],
                     bbox2: Tuple[float, float, float, float]) -> float:
        """IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter_area

        if union <= 0:
            return 0.0
        return inter_area / union
