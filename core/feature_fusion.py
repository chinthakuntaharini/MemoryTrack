"""
Feature fusion module for MemoryTrack system.
Implements multi-modal feature fusion with dynamic weight adjustment.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result of feature fusion."""
    fused_vector: np.ndarray
    weights_used: Dict[str, float]
    occlusion_flags: Dict[str, bool]
    confidence: float


class FeatureFusion:
    """Multi-modal feature fusion with dynamic weighting."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 dimensions: Optional[Dict[str, int]] = None):
        """
        Initialize feature fusion module.
        
        Args:
            weights: Initial weights for each modality
            dimensions: Expected dimensions for each modality
        """
        self.default_weights = weights or {
            'reid': 0.4,
            'pose': 0.2,
            'color': 0.15,
            'accessory': 0.15,
            'motion': 0.1
        }
        
        self.dimensions = dimensions or {
            'reid': 512,
            'pose': 64,
            'color': 96,
            'accessory': 32,
            'motion': 16
        }
        
        self.current_weights = self.default_weights.copy()
        
        logger.info(f"FeatureFusion initialized with weights: {self.default_weights}")
    
    def fuse(self, features: Dict[str, np.ndarray],
             confidences: Optional[Dict[str, float]] = None,
             occlusion_flags: Optional[Dict[str, bool]] = None) -> FusionResult:
        """
        Fuse multiple feature vectors into a unified embedding.
        
        Args:
            features: Dictionary of modality -> feature vector
            confidences: Dictionary of modality -> confidence score
            occlusion_flags: Dictionary of modality -> is_occluded
            
        Returns:
            FusionResult with fused vector and metadata
        """
        # Initialize confidences if not provided
        if confidences is None:
            confidences = {modality: 1.0 for modality in features}
        
        # Initialize occlusion flags if not provided
        if occlusion_flags is None:
            occlusion_flags = {modality: False for modality in features}
        
        # Adjust weights based on occlusion and confidence
        adjusted_weights = self.adjust_weights(
            self.current_weights,
            confidences,
            occlusion_flags
        )
        
        # Normalize and weight each feature
        weighted_features = []
        for modality, feature in features.items():
            if modality in adjusted_weights:
                # Normalize feature
                normalized_feature = self.normalize(feature)
                
                # Apply weight
                weighted_feature = normalized_feature * adjusted_weights[modality]
                weighted_features.append(weighted_feature)
        
        # Concatenate weighted features
        if weighted_features:
            fused_vector = np.concatenate(weighted_features)
        else:
            # Fallback: return zero vector
            total_dim = sum(self.dimensions.values())
            fused_vector = np.zeros(total_dim, dtype=np.float32)
        
        # Normalize final vector
        fused_vector = self.normalize(fused_vector)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            confidences,
            adjusted_weights
        )
        
        return FusionResult(
            fused_vector=fused_vector,
            weights_used=adjusted_weights,
            occlusion_flags=occlusion_flags,
            confidence=overall_confidence
        )
    
    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        L2 normalize a vector.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def adjust_weights(self, weights: Dict[str, float],
                      confidences: Dict[str, float],
                      occlusion_flags: Dict[str, bool]) -> Dict[str, float]:
        """
        Dynamically adjust weights based on confidence and occlusion.
        
        Args:
            weights: Original weights
            confidences: Confidence scores for each modality
            occlusion_flags: Occlusion flags for each modality
            
        Returns:
            Adjusted weights
        """
        adjusted_weights = weights.copy()
        
        for modality in weights:
            # Reduce weight if modality is occluded
            if occlusion_flags.get(modality, False):
                adjusted_weights[modality] *= 0.3
                logger.debug(f"Reduced weight for occluded {modality}")
            
            # Adjust weight based on confidence
            confidence = confidences.get(modality, 1.0)
            if confidence < 0.5:
                # Low confidence: reduce weight
                adjusted_weights[modality] *= confidence
                logger.debug(f"Reduced weight for low-confidence {modality}: {confidence}")
        
        # Renormalize weights to sum to 1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for modality in adjusted_weights:
                adjusted_weights[modality] /= total_weight
        else:
            # Fallback to equal weights
            num_modalities = len(adjusted_weights)
            for modality in adjusted_weights:
                adjusted_weights[modality] = 1.0 / num_modalities
        
        return adjusted_weights
    
    def _calculate_overall_confidence(self, confidences: Dict[str, float],
                                     weights: Dict[str, float]) -> float:
        """
        Calculate overall confidence as weighted average.
        
        Args:
            confidences: Confidence scores
            weights: Weights used
            
        Returns:
            Overall confidence score
        """
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for modality in confidences:
            if modality in weights:
                weighted_confidence += confidences[modality] * weights[modality]
                total_weight += weights[modality]
        
        if total_weight > 0:
            return weighted_confidence / total_weight
        return 0.0
    
    def compute_modality_similarities(self, query_features: Dict[str, np.ndarray],
                                    stored_features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute similarity scores for each modality individually.
        
        Args:
            query_features: Query feature vectors
            stored_features: Stored feature vectors
            
        Returns:
            Dictionary of modality -> similarity score
        """
        similarities = {}
        
        for modality in query_features:
            if modality in stored_features:
                query_vec = self.normalize(query_features[modality])
                stored_vec = self.normalize(stored_features[modality])
                
                # Compute cosine similarity
                similarity = np.dot(query_vec, stored_vec)
                similarities[modality] = float(max(0.0, similarity))
            else:
                similarities[modality] = 0.0
        
        return similarities
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Update fusion weights.
        
        Args:
            weights: New weights dictionary
        """
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            self.current_weights = {k: v / total for k, v in weights.items()}
        else:
            logger.warning("Invalid weights provided, keeping current weights")
        
        logger.info(f"Updated fusion weights: {self.current_weights}")
    
    def get_weights(self) -> Dict[str, float]:
        """Get current fusion weights."""
        return self.current_weights.copy()
    
    def get_expected_dimensions(self) -> Dict[str, int]:
        """Get expected dimensions for each modality."""
        return self.dimensions.copy()
    
    def get_total_dimension(self) -> int:
        """Get total dimension of fused vector."""
        return sum(self.dimensions.values())


class AdaptiveFeatureFusion(FeatureFusion):
    """Adaptive feature fusion that learns optimal weights over time."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 dimensions: Optional[Dict[str, int]] = None,
                 learning_rate: float = 0.01):
        """
        Initialize adaptive feature fusion.
        
        Args:
            weights: Initial weights
            dimensions: Expected dimensions
            learning_rate: Learning rate for weight adaptation
        """
        super().__init__(weights, dimensions)
        self.learning_rate = learning_rate
        self.match_history = []
        
        logger.info("AdaptiveFeatureFusion initialized")
    
    def update_weights(self, match_success: bool,
                      modality_contributions: Dict[str, float]) -> None:
        """
        Update weights based on match success and modality contributions.
        
        Args:
            match_success: Whether the match was successful
            modality_contributions: Contribution of each modality to the match
        """
        # Record match
        self.match_history.append({
            'success': match_success,
            'contributions': modality_contributions.copy()
        })
        
        # Keep only recent history
        if len(self.match_history) > 100:
            self.match_history = self.match_history[-100:]
        
        # Calculate success rate for each modality
        modality_success = {}
        for modality in self.current_weights:
            successes = []
            contributions = []
            
            for match in self.match_history:
                if modality in match['contributions']:
                    successes.append(match['success'])
                    contributions.append(match['contributions'][modality])
            
            if successes:
                # Weighted success rate
                weighted_success = sum(s * c for s, c in zip(successes, contributions))
                total_contribution = sum(contributions)
                modality_success[modality] = weighted_success / (total_contribution + 1e-6)
            else:
                modality_success[modality] = 0.5  # Default
        
        # Update weights using success rates
        for modality in self.current_weights:
            target_weight = modality_success[modality]
            current_weight = self.current_weights[modality]
            
            # Gradual adjustment
            new_weight = current_weight + self.learning_rate * (target_weight - current_weight)
            self.current_weights[modality] = max(0.01, min(0.9, new_weight))
        
        # Renormalize
        total = sum(self.current_weights.values())
        if total > 0:
            for modality in self.current_weights:
                self.current_weights[modality] /= total
        
        logger.debug(f"Updated adaptive weights: {self.current_weights}")
    
    def get_modality_importance(self) -> Dict[str, float]:
        """
        Get importance scores for each modality based on history.
        
        Returns:
            Dictionary of modality -> importance score
        """
        if not self.match_history:
            return self.current_weights.copy()
        
        importance = {}
        for modality in self.current_weights:
            contributions = []
            for match in self.match_history:
                if modality in match['contributions']:
                    contributions.append(match['contributions'][modality])
            
            if contributions:
                importance[modality] = np.mean(contributions)
            else:
                importance[modality] = 0.0
        
        return importance
