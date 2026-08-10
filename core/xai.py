"""
Explainable AI (XAI) module for MemoryTrack system.
Computes per-modality similarity scores and generates natural language match explanations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Modality human-readable labels
MODALITY_LABELS = {
    'reid': 'appearance',
    'pose': 'body shape',
    'color': 'clothing color',
    'accessory': 'accessories',
    'motion': 'movement pattern'
}


@dataclass
class Explanation:
    """Natural language match explanation."""
    summary: str
    modality_contributions: Dict[str, float] = field(default_factory=dict)
    dominant_modalities: List[str] = field(default_factory=list)
    confidence_level: str = ""


class ExplanationGenerator:
    """Generate human-readable match explanations from modality similarities."""

    def __init__(self, modality_labels: Optional[Dict[str, str]] = None,
                 high_threshold: float = 0.7,
                 mid_threshold: float = 0.4):
        """
        Initialize explanation generator.

        Args:
            modality_labels: Optional custom labels for modalities
            high_threshold: Threshold to consider a modality strongly matching
            mid_threshold: Threshold to consider a modality moderately matching
        """
        self.modality_labels = modality_labels or MODALITY_LABELS
        self.high_threshold = high_threshold
        self.mid_threshold = mid_threshold

        logger.info("ExplanationGenerator initialized")

    def generate(self,
                 query_features: Dict[str, np.ndarray],
                 stored_features: Dict[str, np.ndarray],
                 overall_confidence: float) -> Explanation:
        """
        Generate an explanation for a match.

        Args:
            query_features: Query modality feature vectors
            stored_features: Stored profile modality feature vectors
            overall_confidence: Overall match confidence

        Returns:
            Explanation object
        """
        # Compute per-modality similarity
        similarities = self.compute_similarities(query_features, stored_features)

        # Identify dominant modalities
        strong = [m for m, s in similarities.items() if s >= self.high_threshold]
        moderate = [m for m, s in similarities.items() if self.mid_threshold <= s < self.high_threshold]
        weak = [m for m, s in similarities.items() if s < self.mid_threshold]

        dominant = strong if strong else (moderate if moderate else weak)

        # Determine confidence level
        if overall_confidence >= 0.75:
            level = "high"
        elif overall_confidence >= 0.5:
            level = "moderate"
        else:
            level = "low"

        # Build summary sentence
        parts = []
        if strong:
            labels = [self.modality_labels.get(m, m) for m in strong[:3]]
            parts.append(f"strongly matches on {' and '.join(labels)}")
        if moderate and len(strong) == 0:
            labels = [self.modality_labels.get(m, m) for m in moderate[:3]]
            parts.append(f"moderately matches on {' and '.join(labels)}")
        if weak and len(strong) == 0 and len(moderate) == 0:
            labels = [self.modality_labels.get(m, m) for m in weak[:3]]
            parts.append(f"weakly matches on {' and '.join(labels)}")

        summary = f"Match confidence is {overall_confidence:.0%}. "
        summary += "; ".join(parts) if parts else "No strong modality similarities."

        if len(weak) >= len(similarities) and overall_confidence < 0.5:
            summary += " This is a low-confidence match and should be verified."

        return Explanation(
            summary=summary,
            modality_contributions=similarities,
            dominant_modalities=dominant,
            confidence_level=level
        )

    def compute_similarities(self,
                             query_features: Dict[str, np.ndarray],
                             stored_features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute cosine similarity per modality.

        Args:
            query_features: Query modality vectors
            stored_features: Stored modality vectors

        Returns:
            Dict of modality -> similarity score (0 to 1)
        """
        similarities = {}
        for modality in MODALITY_LABELS:
            if modality in query_features and modality in stored_features:
                q = query_features[modality]
                s = stored_features[modality]
                if q is not None and s is not None and len(q) > 0 and len(s) > 0:
                    similarities[modality] = self._cosine_similarity(q, s)
                else:
                    similarities[modality] = 0.0
            else:
                similarities[modality] = 0.0
        return similarities

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        norm_a = np.linalg.norm(a) + 1e-6
        norm_b = np.linalg.norm(b) + 1e-6
        return float(max(0.0, min(1.0, float(np.dot(a, b) / (norm_a * norm_b)))))

    def format_contributions(self, similarities: Dict[str, float],
                            top_k: int = 3) -> str:
        """
        Format modality contributions as readable text.

        Args:
            similarities: Modality similarity dict
            top_k: Number of top contributions to include

        Returns:
            Formatted string
        """
        sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        parts = [
            f"{self.modality_labels.get(m, m)}: {s:.0%}"
            for m, s in sorted_items if s > 0
        ]
        return ", ".join(parts) if parts else "No modality contributions"

