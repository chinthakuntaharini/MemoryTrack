"""
Unit tests for feature_fusion module.
Tests FeatureFusion and AdaptiveFeatureFusion functionality.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.feature_fusion import (
    FeatureFusion,
    AdaptiveFeatureFusion,
    FusionResult
)


class TestFeatureFusion:
    """Tests for FeatureFusion class."""
    
    @pytest.fixture
    def fusion(self):
        """Create a FeatureFusion instance for testing."""
        weights = {
            'reid': 0.4,
            'pose': 0.2,
            'color': 0.15,
            'accessory': 0.15,
            'motion': 0.1
        }
        return FeatureFusion(weights=weights)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature vectors."""
        return {
            'reid': np.random.rand(512).astype(np.float32),
            'pose': np.random.rand(64).astype(np.float32),
            'color': np.random.rand(96).astype(np.float32),
            'accessory': np.random.rand(32).astype(np.float32),
            'motion': np.random.rand(16).astype(np.float32)
        }
    
    def test_initialization(self, fusion):
        """Test fusion module initialization."""
        assert fusion.default_weights['reid'] == 0.4
        assert fusion.default_weights['pose'] == 0.2
        assert len(fusion.dimensions) == 5
        assert fusion.get_total_dimension() == 720
    
    def test_fuse_all_modalities(self, fusion, sample_features):
        """Test fusing all modalities."""
        result = fusion.fuse(sample_features)
        
        assert isinstance(result, FusionResult)
        assert result.fused_vector.shape == (720,)
        assert len(result.weights_used) == 5
        assert result.confidence > 0
    
    def test_fuse_partial_modalities(self, fusion):
        """Test fusing with only some modalities available."""
        partial_features = {
            'reid': np.random.rand(512).astype(np.float32),
            'pose': np.random.rand(64).astype(np.float32)
        }
        
        result = fusion.fuse(partial_features)
        
        assert isinstance(result, FusionResult)
        assert result.fused_vector.shape[0] > 0
        # Weights should be renormalized
        total_weight = sum(result.weights_used.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_fuse_with_confidences(self, fusion, sample_features):
        """Test fusion with confidence scores."""
        confidences = {
            'reid': 0.9,
            'pose': 0.7,
            'color': 0.5,
            'accessory': 0.8,
            'motion': 0.6
        }
        
        result = fusion.fuse(sample_features, confidences=confidences)
        
        assert isinstance(result, FusionResult)
        # Low confidence modalities should have reduced weights
        assert result.weights_used['color'] < fusion.default_weights['color']
    
    def test_fuse_with_occlusion(self, fusion, sample_features):
        """Test fusion with occlusion flags."""
        occlusion_flags = {
            'reid': False,
            'pose': True,  # Pose is occluded
            'color': False,
            'accessory': False,
            'motion': False
        }
        
        result = fusion.fuse(sample_features, occlusion_flags=occlusion_flags)
        
        assert isinstance(result, FusionResult)
        # Occluded modality should have reduced weight
        assert result.weights_used['pose'] < fusion.default_weights['pose']
        assert result.occlusion_flags['pose'] is True
    
    def test_normalization(self, fusion):
        """Test vector normalization."""
        vector = np.array([3.0, 4.0, 0.0])
        normalized = fusion.normalize(vector)
        
        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-6
    
    def test_adjust_weights(self, fusion):
        """Test dynamic weight adjustment."""
        weights = fusion.default_weights.copy()
        confidences = {
            'reid': 0.5,
            'pose': 0.9,
            'color': 0.3,
            'accessory': 0.8,
            'motion': 0.7
        }
        occlusion_flags = {
            'reid': False,
            'pose': False,
            'color': True,  # Occluded
            'accessory': False,
            'motion': False
        }
        
        adjusted = fusion.adjust_weights(weights, confidences, occlusion_flags)
        
        # Weights should sum to 1
        total = sum(adjusted.values())
        assert abs(total - 1.0) < 0.01
        
        # Occluded should have significantly reduced weight
        assert adjusted['color'] < weights['color']
        
        # High confidence modalities should get relatively higher weight
        assert adjusted['pose'] > weights['pose'] or adjusted['accessory'] > weights['accessory']
    
    def test_compute_modality_similarities(self, fusion, sample_features):
        """Test computing individual modality similarities."""
        stored_features = {
            'reid': np.random.rand(512).astype(np.float32),
            'pose': np.random.rand(64).astype(np.float32),
            'color': np.random.rand(96).astype(np.float32),
            'accessory': np.random.rand(32).astype(np.float32),
            'motion': np.random.rand(16).astype(np.float32)
        }
        
        similarities = fusion.compute_modality_similarities(
            sample_features,
            stored_features
        )
        
        assert len(similarities) == 5
        assert all(0 <= s <= 1 for s in similarities.values())
    
    def test_set_weights(self, fusion):
        """Test setting custom weights."""
        new_weights = {
            'reid': 0.5,
            'pose': 0.3,
            'color': 0.1,
            'accessory': 0.05,
            'motion': 0.05
        }
        
        fusion.set_weights(new_weights)
        
        current = fusion.get_weights()
        assert current['reid'] == 0.5
        assert current['pose'] == 0.3
    
    def test_get_expected_dimensions(self, fusion):
        """Test getting expected dimensions."""
        dims = fusion.get_expected_dimensions()
        
        assert dims['reid'] == 512
        assert dims['pose'] == 64
        assert dims['color'] == 96
        assert dims['accessory'] == 32
        assert dims['motion'] == 16
    
    def test_get_total_dimension(self, fusion):
        """Test getting total dimension."""
        total = fusion.get_total_dimension()
        assert total == 720


class TestAdaptiveFeatureFusion:
    """Tests for AdaptiveFeatureFusion class."""
    
    @pytest.fixture
    def adaptive_fusion(self):
        """Create an AdaptiveFeatureFusion instance for testing."""
        weights = {
            'reid': 0.4,
            'pose': 0.2,
            'color': 0.15,
            'accessory': 0.15,
            'motion': 0.1
        }
        return AdaptiveFeatureFusion(weights=weights, learning_rate=0.1)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature vectors."""
        return {
            'reid': np.random.rand(512).astype(np.float32),
            'pose': np.random.rand(64).astype(np.float32),
            'color': np.random.rand(96).astype(np.float32),
            'accessory': np.random.rand(32).astype(np.float32),
            'motion': np.random.rand(16).astype(np.float32)
        }
    
    def test_initialization(self, adaptive_fusion):
        """Test adaptive fusion initialization."""
        assert adaptive_fusion.learning_rate == 0.1
        assert len(adaptive_fusion.match_history) == 0
    
    def test_update_weights_success(self, adaptive_fusion):
        """Test weight update after successful match."""
        contributions = {
            'reid': 0.5,
            'pose': 0.3,
            'color': 0.1,
            'accessory': 0.05,
            'motion': 0.05
        }
        
        initial_weights = adaptive_fusion.get_weights().copy()
        
        adaptive_fusion.update_weights(match_success=True, modality_contributions=contributions)
        
        updated_weights = adaptive_fusion.get_weights()
        
        # Weights should have changed
        assert updated_weights != initial_weights
        
        # Match history should be updated
        assert len(adaptive_fusion.match_history) == 1
    
    def test_update_weights_failure(self, adaptive_fusion):
        """Test weight update after failed match."""
        contributions = {
            'reid': 0.1,
            'pose': 0.1,
            'color': 0.6,
            'accessory': 0.1,
            'motion': 0.1
        }
        
        adaptive_fusion.update_weights(match_success=False, modality_contributions=contributions)
        
        # Match history should be updated
        assert len(adaptive_fusion.match_history) == 1
        assert adaptive_fusion.match_history[0]['success'] is False
    
    def test_get_modality_importance(self, adaptive_fusion):
        """Test getting modality importance scores."""
        # Add some match history
        for i in range(5):
            contributions = {
                'reid': 0.4 + i * 0.05,
                'pose': 0.2,
                'color': 0.15,
                'accessory': 0.15,
                'motion': 0.1
            }
            adaptive_fusion.update_weights(
                match_success=True,
                modality_contributions=contributions
            )
        
        importance = adaptive_fusion.get_modality_importance()
        
        assert len(importance) == 5
        assert all(0 <= v <= 1 for v in importance.values())
    
    def test_history_limit(self, adaptive_fusion):
        """Test that match history is limited."""
        # Add more than limit matches
        for i in range(150):
            contributions = {
                'reid': 0.4,
                'pose': 0.2,
                'color': 0.15,
                'accessory': 0.15,
                'motion': 0.1
            }
            adaptive_fusion.update_weights(
                match_success=True,
                modality_contributions=contributions
            )
        
        # History should be limited to 100
        assert len(adaptive_fusion.match_history) <= 100


class TestFusionResult:
    """Tests for FusionResult dataclass."""
    
    def test_fusion_result_creation(self):
        """Test creating a fusion result."""
        fused_vector = np.random.rand(720).astype(np.float32)
        weights_used = {'reid': 0.4, 'pose': 0.2}
        occlusion_flags = {'reid': False, 'pose': True}
        
        result = FusionResult(
            fused_vector=fused_vector,
            weights_used=weights_used,
            occlusion_flags=occlusion_flags,
            confidence=0.85
        )
        
        assert result.fused_vector.shape == (720,)
        assert result.weights_used == weights_used
        assert result.occlusion_flags == occlusion_flags
        assert result.confidence == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
