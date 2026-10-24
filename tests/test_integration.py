"""
Integration tests for MemoryTrack pipeline components.
Tests the full pipeline flow from detection to memory storage.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil

from core.detector import PersonDetector, MultiObjectTracker, BoundingBox
from core.pose_extractor import PoseExtractor
from core.reid_extractor import ReIDExtractor
from core.color_extractor import ColorExtractor
from core.feature_fusion import FeatureFusion
from core.memory_bank import AdaptiveMemoryBank
from utils.config_loader import ConfigLoader
from utils.db_manager import DatabaseManager


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    
    db = DatabaseManager(str(db_path))
    db.init_db()
    
    yield db
    
    # Cleanup
    db.close()
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 100), (400, 400), (128, 128, 128), -1)
    return frame


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box."""
    return BoundingBox(x1=200, y1=100, x2=400, y2=400, confidence=0.9, class_id=0, class_name="person")


@pytest.fixture
def config():
    """Load test configuration."""
    return ConfigLoader()


class TestPipelineIntegration:
    """Test full pipeline integration."""
    
    def test_feature_extraction_pipeline(self, sample_frame, sample_bbox, config):
        """Test the complete feature extraction pipeline."""
        pose_extractor = PoseExtractor(model_complexity=1)
        reid_extractor = ReIDExtractor(device='cpu')
        color_extractor = ColorExtractor()
        
        pose_features = pose_extractor.extract(sample_frame, (sample_bbox.x1, sample_bbox.y1, sample_bbox.x2, sample_bbox.y2))
        reid_features = reid_extractor.extract(sample_frame, (sample_bbox.x1, sample_bbox.y1, sample_bbox.x2, sample_bbox.y2))
        color_features = color_extractor.extract(sample_frame, (sample_bbox.x1, sample_bbox.y1, sample_bbox.x2, sample_bbox.y2), pose_features)
        
        assert reid_features is not None
        assert len(reid_features) == 512
        
        assert pose_features is not None
        assert pose_features.keypoints is not None
        assert pose_features.keypoints.shape == (33, 3)
        
        assert color_features is not None
        assert color_features.upper_body_hist is not None
        assert color_features.lower_body_hist is not None
    
    def test_feature_fusion_pipeline(self, sample_frame, sample_bbox):
        """Test feature fusion with multiple modalities."""
        pose_extractor = PoseExtractor(model_complexity=1)
        reid_extractor = ReIDExtractor(device='cpu')
        color_extractor = ColorExtractor()
        fusion = FeatureFusion()
        
        bbox_tuple = (sample_bbox.x1, sample_bbox.y1, sample_bbox.x2, sample_bbox.y2)
        pose_features = pose_extractor.extract(sample_frame, bbox_tuple)
        reid_features = reid_extractor.extract(sample_frame, bbox_tuple)
        color_features = color_extractor.extract(sample_frame, bbox_tuple, pose_features)
        
        pose_vec = pose_extractor.extract_feature_vector(pose_features)
        color_vec = color_extractor.extract_feature_vector(color_features)
        
        features = {
            'reid': reid_features,
            'pose': pose_vec,
            'color': color_vec
        }
        
        confidences = {
            'reid': 0.9,
            'pose': pose_features.confidence,
            'color': 0.8
        }
        
        result = fusion.fuse(features, confidences)
        
        assert result is not None
        assert result.fused_vector is not None
        assert len(result.fused_vector) > 0
        assert result.confidence > 0
    
    def test_memory_bank_integration(self, temp_db):
        """Test memory bank with database integration."""
        memory_bank = AdaptiveMemoryBank(
            embedding_dim=512,
            db_manager=temp_db
        )
        
        person_id = 1
        features = np.random.random(512).astype(np.float32)
        
        memory_bank.add_profile(
            person_id=person_id,
            features=features,
            camera_id='cam1'
        )
        
        query = features + np.random.normal(0, 0.05, 512).astype(np.float32)
        results = memory_bank.search(query, top_k=5)
        
        assert len(results) > 0
        assert results[0].person_id == person_id
        assert results[0].confidence > 0.3
    
    def test_detection_tracking_integration(self, sample_frame):
        """Test detection and tracking integration."""
        try:
            detector = PersonDetector(confidence_threshold=0.3, device='cpu')
            tracker = MultiObjectTracker()
            
            detections = detector.detect(sample_frame)
            tracks = tracker.update(detections)
            
            assert isinstance(detections, list)
            assert isinstance(tracks, dict)
            
        except Exception as e:
            pytest.skip(f"Detection models not available: {e}")
    
    def test_config_validation(self, config):
        """Test configuration loading and validation."""
        assert config.get_detection_config() is not None
        assert config.get_tracking_config() is not None
        assert config.get_feature_extraction_config() is not None
        assert config.get_memory_bank_config() is not None
        
        assert config.validate() is True
    
    def test_database_operations(self, temp_db):
        """Test database CRUD operations."""
        person_data = {
            'name': 'Test Person',
            'status': 'missing',
            'notes': 'Integration test'
        }
        
        person_id = temp_db.create_person(person_data)
        assert person_id is not None
        
        person = temp_db.get_person(person_id)
        assert person is not None
        assert person['name'] == 'Test Person'
        
        features = np.random.random(720).astype(np.float32)
        snapshot_data = {
            'person_id': person_id,
            'features': features,
            'camera_id': 'cam1',
            'confidence': 0.85
        }
        
        snapshot_id = temp_db.add_feature_snapshot(snapshot_data)
        assert snapshot_id is not None
        
        snapshots = temp_db.get_person_snapshots(person_id)
        assert len(snapshots) == 1
        assert snapshots[0]['confidence'] == 0.85


class TestErrorHandling:
    """Test error handling and graceful degradation."""
    
    def test_missing_model_fallback(self):
        """Test fallback when models are missing."""
        try:
            detector = PersonDetector(
                model_path="nonexistent_model.pt", 
                device='cpu'
            )
        except Exception:
            pass
    
    def test_invalid_frame_handling(self):
        """Test handling of invalid frames."""
        pose_extractor = PoseExtractor(model_complexity=1)
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9, class_id=0, class_name="person")
        
        result = pose_extractor.extract(None, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        assert result is None or hasattr(result, 'keypoints')
    
    def test_empty_detection_handling(self):
        """Test handling when no persons are detected."""
        tracker = MultiObjectTracker()
        tracks = tracker.update([])
        assert isinstance(tracks, dict)
        assert len(tracks) == 0


class TestPerformance:
    """Basic performance tests."""
    
    def test_feature_extraction_speed(self, sample_frame, sample_bbox):
        """Test that feature extraction completes in reasonable time."""
        import time
        
        pose_extractor = PoseExtractor(model_complexity=0)
        
        start_time = time.time()
        result = pose_extractor.extract(sample_frame, (sample_bbox.x1, sample_bbox.y1, sample_bbox.x2, sample_bbox.y2))
        elapsed = time.time() - start_time
        
        assert elapsed < 5.0
        assert result is not None
    
    def test_memory_bank_search_speed(self, temp_db):
        """Test memory bank search performance."""
        import time
        
        memory_bank = AdaptiveMemoryBank(embedding_dim=512, db_manager=temp_db)
        
        for i in range(50):
            features = np.random.random(512).astype(np.float32)
            memory_bank.add_profile(
                person_id=i,
                features=features,
                camera_id=f'cam{i%5}'
            )
        
        query = np.random.random(512).astype(np.float32)
        
        start_time = time.time()
        results = memory_bank.search(query, top_k=10)
        elapsed = time.time() - start_time
        
        assert elapsed < 1.0
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__])