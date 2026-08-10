"""
Accessory extractor module for MemoryTrack system.
Implements accessory detection and feature vector extraction using YOLO detections.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Standard set of accessory categories that are tracked by the system.
# Each category maps to a one-hot index in the 32-dimensional feature vector.
ACCESSORY_CLASSES = [
    'backpack',
    'handbag',
    'suitcase',
    'umbrella',
    'cap',
    'bottle',
    'tie',
    'umbrella2',
    'wine_glass',
    'skateboard',
    'surfboard',
    'tennis_racket',
    'book',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot_dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted_plant',
    'bed'
]

# The first N classes are the "primary" accessories explicitly configured.
PRIMARY_ACCESSORIES = ['backpack', 'handbag', 'suitcase', 'umbrella', 'cap', 'bottle']


@dataclass
class AccessoryDetection:
    """A single accessory detection."""
    accessory_type: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int


@dataclass
class AccessoryFeatures:
    """Accessory features extracted for a person."""
    accessory_vector: np.ndarray   # 32-dimensional one-hot style vector
    accessories: List[AccessoryDetection] = field(default_factory=list)
    confidence: float = 0.0


class AccessoryExtractor:
    """Extract accessory feature vectors from YOLO detections."""

    def __init__(self,
                 confidence_threshold: float = 0.4,
                 accessory_classes: Optional[List[str]] = None,
                 vector_dim: int = 32,
                 iou_threshold: float = 0.3):
        """
        Initialize accessory extractor.

        Args:
            confidence_threshold: Minimum detection confidence
            accessory_classes: Which classes to treat as accessories
            vector_dim: Dimension of the output feature vector
            iou_threshold: Minimum IoU for assigning an accessory to a person
        """
        self.confidence_threshold = confidence_threshold
        self.accessory_classes = accessory_classes or PRIMARY_ACCESSORIES
        self.vector_dim = vector_dim
        self.iou_threshold = iou_threshold

        self.class_to_index = {name: idx for idx, name in enumerate(ACCESSORY_CLASSES)}

        logger.info(
            f"AccessoryExtractor initialized with threshold={confidence_threshold}, "
            f"classes={self.accessory_classes}"
        )

    def extract(self,
                frame: np.ndarray,
                person_bbox: Tuple[float, float, float, float],
                detections: Optional[Dict[str, List]] = None,
                detector=None) -> AccessoryFeatures:
        """
        Extract accessory features for a person.

        Args:
            frame: Input frame (unused here; kept for API symmetry)
            person_bbox: Person bounding box (x1, y1, x2, y2)
            detections: Pre-computed accessory detections dict from
                        PersonDetector.detect_with_accessories. If None, the detector
                        will be called (if provided).
            detector: Optional PersonDetector instance to run detection

        Returns:
            AccessoryFeatures with a 32-dimensional vector
        """
        # Build detection source
        all_accessory_detections: List[AccessoryDetection] = []

        if detections is None and detector is not None:
            detections = detector.detect_with_accessories(
                frame,
                accessory_classes=self.accessory_classes
            )

        if detections:
            for acc_type, bboxes in detections.items():
                if acc_type not in self.accessory_classes:
                    continue
                for bbox in bboxes:
                    conf = getattr(bbox, 'confidence', 0.5)
                    if conf < self.confidence_threshold:
                        continue
                    all_accessory_detections.append(AccessoryDetection(
                        accessory_type=acc_type,
                        confidence=float(conf),
                        bbox=(
                            float(getattr(bbox, 'x1', 0)),
                            float(getattr(bbox, 'y1', 0)),
                            float(getattr(bbox, 'x2', 0)),
                            float(getattr(bbox, 'y2', 0))
                        ),
                        class_id=getattr(bbox, 'class_id', -1)
                    ))

        # Filter detections that overlap with the person bbox
        matched = [
            d for d in all_accessory_detections
            if self._compute_iou(person_bbox, d.bbox) >= self.iou_threshold
        ]

        # Build one-hot style vector
        vector = np.zeros(self.vector_dim, dtype=np.float32)
        if matched:
            best_by_type: Dict[str, AccessoryDetection] = {}
            for d in matched:
                if (d.accessory_type not in best_by_type or
                        d.confidence > best_by_type[d.accessory_type].confidence):
                    best_by_type[d.accessory_type] = d

            for d in best_by_type.values():
                idx = self.class_to_index.get(d.accessory_type)
                if idx is not None and idx < self.vector_dim:
                    # Weight by confidence
                    vector[idx] = float(d.confidence)

        # Overall confidence is the max individual accessory confidence (saturated)
        confidence = float(max([d.confidence for d in matched], default=0.0))
        confidence = min(1.0, confidence)

        return AccessoryFeatures(
            accessory_vector=vector,
            accessories=matched,
            confidence=confidence
        )

    def compute_similarity(self, features1: np.ndarray,
                          features2: np.ndarray) -> float:
        """
        Compute accessory similarity between two feature vectors.

        Args:
            features1: First accessory feature vector
            features2: Second accessory feature vector

        Returns:
            Similarity score between 0 and 1
        """
        # Cosine similarity; both vectors are non-negative
        norm1 = np.linalg.norm(features1) + 1e-6
        norm2 = np.linalg.norm(features2) + 1e-6
        cos_sim = float(np.dot(features1, features2) / (norm1 * norm2))
        return max(0.0, min(1.0, cos_sim))

    def extract_feature_vector(self, accessory_features: AccessoryFeatures) -> np.ndarray:
        """
        Return the feature vector from AccessoryFeatures.

        Args:
            accessory_features: AccessoryFeatures object

        Returns:
            32-dimensional vector
        """
        vec = accessory_features.accessory_vector
        # Ensure exact dimension
        if len(vec) < self.vector_dim:
            vec = np.pad(vec, (0, self.vector_dim - len(vec)))
        elif len(vec) > self.vector_dim:
            vec = vec[:self.vector_dim]
        return vec

    def _compute_iou(self, bbox1: Tuple[float, float, float, float],
                     bbox2: Tuple[float, float, float, float]) -> float:
        """
        Compute Intersection over Union between two bounding boxes.

        Args:
            bbox1: First bbox (x1, y1, x2, y2)
            bbox2: Second bbox (x1, y1, x2, y2)

        Returns:
            IoU score
        """
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

