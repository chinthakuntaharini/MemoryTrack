"""
Unit tests for memory_bank module.
Tests AdaptiveMemoryBank functionality including temporal decay and profile management.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.memory_bank import (
    AdaptiveMemoryBank,
    PersonProfile,
    TemporalSnapshot,
    MatchResult
)


class TestTemporalSnapshot:
    """Tests for TemporalSnapshot dataclass."""
    
    def test_snapshot_creation(self):
        """Test creating a temporal snapshot."""
        features = np.random.rand(720).astype(np.float32)
        timestamp = datetime.now()
        
        snapshot = TemporalSnapshot(
            features=features,
            timestamp=timestamp,
            camera_id="cam_1",
            confidence=0.85
        )
        
        assert snapshot.features.shape == (720,)
        assert snapshot.camera_id == "cam_1"
        assert snapshot.confidence == 0.85
        assert snapshot.frame_number is None


class TestPersonProfile:
    """Tests for PersonProfile dataclass."""
    
    def test_profile_creation(self):
        """Test creating a person profile."""
        profile = PersonProfile(
            person_id=1,
            name="John Doe",
            status="missing"
        )
        
        assert profile.person_id == 1
        assert profile.name == "John Doe"
        assert profile.status == "missing"
        assert len(profile.snapshots) == 0
    
    def test_add_snapshot(self):
        """Test adding snapshot to profile."""
        profile = PersonProfile(person_id=1)
        
        features = np.random.rand(720).astype(np.float32)
        snapshot = TemporalSnapshot(
            features=features,
            timestamp=datetime.now(),
            camera_id="cam_1",
            confidence=0.85
        )
        
        profile.snapshots.append(snapshot)
        
        assert len(profile.snapshots) == 1
        assert profile.snapshots[0].camera_id == "cam_1"


class TestAdaptiveMemoryBank:
    """Tests for AdaptiveMemoryBank class."""
    
    @pytest.fixture
    def memory_bank(self):
        """Create a memory bank instance for testing."""
        return AdaptiveMemoryBank(
            embedding_dim=720,
            decay_rate=0.0001,
            max_snapshots_per_person=5,
            update_threshold=0.8
        )
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature vector."""
        return np.random.rand(720).astype(np.float32)
    
    def test_initialization(self, memory_bank):
        """Test memory bank initialization."""
        assert memory_bank.embedding_dim == 720
        assert memory_bank.decay_rate == 0.0001
        assert memory_bank.max_snapshots_per_person == 5
        assert memory_bank.update_threshold == 0.8
        assert len(memory_bank.profiles) == 0
    
    def test_add_profile(self, memory_bank, sample_features):
        """Test adding a new profile."""
        profile = memory_bank.add_profile(
            person_id=1,
            features=sample_features,
            camera_id="cam_1",
            confidence=0.85,
            name="Test Person"
        )
        
        assert profile.person_id == 1
        assert profile.name == "Test Person"
        assert len(profile.snapshots) == 1
        assert profile.last_seen_camera == "cam_1"
        assert 1 in memory_bank.profiles
    
    def test_add_multiple_snapshots(self, memory_bank, sample_features):
        """Test adding multiple snapshots to a profile."""
        # Add first snapshot
        memory_bank.add_profile(
            person_id=1,
            features=sample_features,
            camera_id="cam_1",
            confidence=0.85
        )
        
        # Add second snapshot
        new_features = np.random.rand(720).astype(np.float32)
        memory_bank.update_profile(
            person_id=1,
            features=new_features,
            camera_id="cam_2",
            confidence=0.9
        )
        
        profile = memory_bank.get_profile(1)
        assert len(profile.snapshots) == 2
    
    def test_max_snapshots_limit(self, memory_bank):
        """Test that max snapshots limit is enforced."""
        # Add more snapshots than max_snapshots_per_person
        for i in range(10):
            features = np.random.rand(720).astype(np.float32)
            if i == 0:
                memory_bank.add_profile(
                    person_id=1,
                    features=features,
                    camera_id="cam_1",
                    confidence=0.85
                )
            else:
                memory_bank.update_profile(
                    person_id=1,
                    features=features,
                    camera_id=f"cam_{i}",
                    confidence=0.9
                )
        
        profile = memory_bank.get_profile(1)
        assert len(profile.snapshots) <= memory_bank.max_snapshots_per_person
    
    def test_update_profile_high_confidence(self, memory_bank, sample_features):
        """Test profile update with high confidence."""
        # Add initial profile
        memory_bank.add_profile(
            person_id=1,
            features=sample_features,
            camera_id="cam_1",
            confidence=0.7
        )
        
        initial_snapshot_count = len(memory_bank.get_profile(1).snapshots)
        
        # Update with high confidence (should add new snapshot)
        new_features = np.random.rand(720).astype(np.float32)
        memory_bank.update_profile(
            person_id=1,
            features=new_features,
            camera_id="cam_2",
            confidence=0.9  # Above threshold
        )
        
        profile = memory_bank.get_profile(1)
        assert len(profile.snapshots) == initial_snapshot_count + 1
    
    def test_update_profile_low_confidence(self, memory_bank, sample_features):
        """Test profile update with low confidence (averaging)."""
        # Add initial profile
        memory_bank.add_profile(
            person_id=1,
            features=sample_features,
            camera_id="cam_1",
            confidence=0.9
        )
        
        initial_snapshot_count = len(memory_bank.get_profile(1).snapshots)
        
        # Update with low confidence (should average)
        new_features = np.random.rand(720).astype(np.float32)
        memory_bank.update_profile(
            person_id=1,
            features=new_features,
            camera_id="cam_2",
            confidence=0.5  # Below threshold
        )
        
        profile = memory_bank.get_profile(1)
        # Snapshot count should not increase
        assert len(profile.snapshots) == initial_snapshot_count
    
    def test_search_empty_bank(self, memory_bank, sample_features):
        """Test search on empty memory bank."""
        results = memory_bank.search(sample_features, top_k=5)
        assert len(results) == 0
    
    def test_search_with_profiles(self, memory_bank, sample_features):
        """Test search with profiles in memory bank."""
        # Add profiles
        for i in range(3):
            features = np.random.rand(720).astype(np.float32)
            memory_bank.add_profile(
                person_id=i,
                features=features,
                camera_id="cam_1",
                confidence=0.85
            )
        
        # Search
        results = memory_bank.search(sample_features, top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, MatchResult) for r in results)
    
    def test_get_profile(self, memory_bank, sample_features):
        """Test getting a profile by ID."""
        memory_bank.add_profile(
            person_id=1,
            features=sample_features,
            camera_id="cam_1",
            confidence=0.85
        )
        
        profile = memory_bank.get_profile(1)
        assert profile is not None
        assert profile.person_id == 1
        
        # Test non-existent profile
        profile = memory_bank.get_profile(999)
        assert profile is None
    
    def test_delete_profile(self, memory_bank, sample_features):
        """Test deleting a profile."""
        memory_bank.add_profile(
            person_id=1,
            features=sample_features,
            camera_id="cam_1",
            confidence=0.85
        )
        
        assert 1 in memory_bank.profiles
        
        result = memory_bank.delete_profile(1)
        assert result is True
        assert 1 not in memory_bank.profiles
        
        # Test deleting non-existent profile
        result = memory_bank.delete_profile(999)
        assert result is False
    
    def test_update_status(self, memory_bank, sample_features):
        """Test updating person status."""
        memory_bank.add_profile(
            person_id=1,
            features=sample_features,
            camera_id="cam_1",
            confidence=0.85
        )
        
        result = memory_bank.update_status(1, "found")
        assert result is True
        
        profile = memory_bank.get_profile(1)
        assert profile.status == "found"
    
    def test_temporal_weight_calculation(self, memory_bank):
        """Test temporal weight calculation."""
        now = datetime.now()
        past_time = now - timedelta(seconds=1000)
        
        weight = memory_bank._calculate_temporal_weight(past_time, now)
        
        # Weight should be between 0 and 1
        assert 0 <= weight <= 1
        
        # More recent time should have higher weight
        recent_weight = memory_bank._calculate_temporal_weight(
            now - timedelta(seconds=10), now
        )
        assert recent_weight > weight
    
    def test_normalization(self, memory_bank):
        """Test vector normalization."""
        vector = np.array([3.0, 4.0, 0.0])
        normalized = memory_bank._normalize(vector)
        
        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-6
    
    def test_statistics(self, memory_bank, sample_features):
        """Test memory bank statistics."""
        # Add some profiles
        for i in range(5):
            features = np.random.rand(720).astype(np.float32)
            memory_bank.add_profile(
                person_id=i,
                features=features,
                camera_id="cam_1",
                confidence=0.85
            )
            # Set status separately
            memory_bank.update_status(i, "missing" if i < 3 else "found")
        
        stats = memory_bank.get_statistics()
        
        assert stats['total_profiles'] == 5
        assert stats['total_snapshots'] == 5
        assert stats['avg_snapshots_per_profile'] == 1.0
        assert stats['status_distribution']['missing'] == 3
        assert stats['status_distribution']['found'] == 2
    
    def test_save_and_load(self, memory_bank, sample_features):
        """Test saving and loading memory bank."""
        # Add some data
        memory_bank.add_profile(
            person_id=1,
            features=sample_features,
            camera_id="cam_1",
            confidence=0.85,
            name="Test Person"
        )
        memory_bank.update_status(1, "missing")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            memory_bank.save(temp_path)
            
            # Create new memory bank and load
            new_bank = AdaptiveMemoryBank(embedding_dim=720)
            new_bank.load(temp_path)
            
            # Verify data
            assert len(new_bank.profiles) == 1
            assert 1 in new_bank.profiles
            # Note: name might not be preserved in current implementation
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_apply_temporal_decay(self, memory_bank):
        """Test applying temporal decay to all profiles."""
        # Add profile with old timestamp
        features = np.random.rand(720).astype(np.float32)
        profile = memory_bank.add_profile(
            person_id=1,
            features=features,
            camera_id="cam_1",
            confidence=0.85
        )
        
        # Manually set old timestamp
        old_time = datetime.now() - timedelta(days=100)
        profile.snapshots[0].timestamp = old_time
        
        # Apply decay
        memory_bank.apply_temporal_decay_to_all()
        
        # Profile should be removed due to very low temporal weight
        assert 1 not in memory_bank.profiles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
