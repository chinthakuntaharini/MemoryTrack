"""
Memory bank module for MemoryTrack system.
Implements adaptive memory bank with FAISS indexing and temporal decay.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
import pickle
import logging

try:
    import faiss
except ImportError:
    raise ImportError("FAISS not found. Install with: pip install faiss-cpu")

logger = logging.getLogger(__name__)


@dataclass
class TemporalSnapshot:
    """Single temporal snapshot of a person's features."""
    features: np.ndarray
    timestamp: datetime
    camera_id: str
    confidence: float
    frame_number: Optional[int] = None
    metadata: Dict = field(default_factory=dict)
    modality_features: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class PersonProfile:
    """Complete profile for a person with multiple temporal snapshots."""
    person_id: int
    name: Optional[str] = None
    status: str = "missing"
    snapshots: List[TemporalSnapshot] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_seen_camera: Optional[str] = None
    last_seen_time: Optional[datetime] = None


@dataclass
class MatchResult:
    """Result of a similarity search."""
    person_id: int
    confidence: float
    distance: float
    profile: PersonProfile
    snapshot_used: TemporalSnapshot
    modality_similarities: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


class AdaptiveMemoryBank:
    """FAISS-backed adaptive memory bank with temporal decay."""

    def __init__(self, embedding_dim: int = 720,
                 decay_rate: float = 0.0001,
                 max_snapshots_per_person: int = 10,
                 update_threshold: float = 0.8,
                 index_type: str = "IndexFlatIP"):
        """
        Initialize adaptive memory bank.

        Args:
            embedding_dim: Dimension of feature vectors
            decay_rate: Temporal decay rate (per second)
            max_snapshots_per_person: Maximum snapshots to keep per person
            update_threshold: Confidence threshold for profile updates
            index_type: FAISS index type
        """
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        self.max_snapshots_per_person = max_snapshots_per_person
        self.update_threshold = update_threshold

        # Initialize FAISS index
        self.index = self._create_index(index_type)

        # Store profiles
        self.profiles: Dict[int, PersonProfile] = {}

        # Mapping from FAISS index to profile/snapshot
        self.index_to_profile: Dict[int, Tuple[int, int]] = {}  # index -> (person_id, snapshot_idx)
        self.next_index = 0

        logger.info(f"AdaptiveMemoryBank initialized with dim={embedding_dim}, decay={decay_rate}")

    def _create_index(self, index_type: str):
        """
        Create FAISS index.

        Args:
            index_type: Type of FAISS index

        Returns:
            FAISS index object
        """
        if index_type == "IndexFlatIP":
            # Inner product index (for cosine similarity with normalized vectors)
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == "IndexFlatL2":
            # L2 distance index
            index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            logger.warning(f"Unknown index type {index_type}, using IndexFlatIP")
            index = faiss.IndexFlatIP(self.embedding_dim)

        return index

    def add_profile(self, person_id: int, features: np.ndarray,
                    camera_id: str, confidence: float,
                    name: Optional[str] = None,
                    frame_number: Optional[int] = None,
                    metadata: Optional[Dict] = None,
                    modality_features: Optional[Dict[str, np.ndarray]] = None) -> PersonProfile:
        """
        Add a new person profile or update existing one.

        Args:
            person_id: Person ID
            features: Feature vector
            camera_id: Camera ID where person was seen
            confidence: Detection confidence
            name: Person name (optional)
            frame_number: Frame number (optional)
            metadata: Additional metadata
            modality_features: Optional dict of modality -> feature vector

        Returns:
            PersonProfile object
        """
        # Create or get profile
        if person_id not in self.profiles:
            profile = PersonProfile(
                person_id=person_id,
                name=name,
                created_at=datetime.now()
            )
            self.profiles[person_id] = profile
            logger.info(f"Created new profile for person {person_id}")
        else:
            profile = self.profiles[person_id]
            if name:
                profile.name = name

        # Create snapshot
        snapshot = TemporalSnapshot(
            features=features.copy(),
            timestamp=datetime.now(),
            camera_id=camera_id,
            confidence=confidence,
            frame_number=frame_number,
            metadata=metadata or {},
            modality_features=modality_features or {}
        )

        # Add snapshot to profile
        profile.snapshots.append(snapshot)
        profile.updated_at = datetime.now()
        profile.last_seen_camera = camera_id
        profile.last_seen_time = snapshot.timestamp

        # Manage snapshot count
        if len(profile.snapshots) > self.max_snapshots_per_person:
            # Remove oldest snapshot
            profile.snapshots.pop(0)
            # Remove from FAISS index (this is complex, so we'll rebuild)
            self._rebuild_index()
            logger.debug(f"Removed oldest snapshot for person {person_id}")

        # Add to FAISS index
        self._add_to_index(person_id, len(profile.snapshots) - 1, features)

        return profile

    def _add_to_index(self, person_id: int, snapshot_idx: int, features: np.ndarray) -> None:
        """
        Add feature vector to FAISS index.

        Args:
            person_id: Person ID
            snapshot_idx: Snapshot index within profile
            features: Feature vector
        """
        # Normalize for cosine similarity
        features_normalized = self._normalize(features)

        # Add to index
        features_array = features_normalized.reshape(1, -1).astype('float32')
        self.index.add(features_array)

        # Store mapping
        self.index_to_profile[self.next_index] = (person_id, snapshot_idx)
        self.next_index += 1

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from current profiles."""
        # Reset index
        index_type = type(self.index).__name__
        self.index = self._create_index(index_type.replace('Index', 'Index'))
        self.index_to_profile.clear()
        self.next_index = 0

        # Re-add all snapshots
        for person_id, profile in self.profiles.items():
            for snapshot_idx, snapshot in enumerate(profile.snapshots):
                self._add_to_index(person_id, snapshot_idx, snapshot.features)

        logger.info("Rebuilt FAISS index")

    def update_profile(self, person_id: int, features: np.ndarray,
                       camera_id: str, confidence: float,
                       frame_number: Optional[int] = None,
                       modality_features: Optional[Dict[str, np.ndarray]] = None) -> Optional[PersonProfile]:
        """
        Update existing profile with new features.

        Args:
            person_id: Person ID
            features: New feature vector
            camera_id: Camera ID
            confidence: Detection confidence
            frame_number: Frame number
            modality_features: Optional dict of modality -> feature vector

        Returns:
            Updated profile or None if person not found
        """
        if person_id not in self.profiles:
            logger.warning(f"Person {person_id} not found in memory bank")
            return None

        profile = self.profiles[person_id]

        if confidence >= self.update_threshold:
            # High confidence: add as new snapshot
            snapshot = TemporalSnapshot(
                features=features.copy(),
                timestamp=datetime.now(),
                camera_id=camera_id,
                confidence=confidence,
                frame_number=frame_number,
                modality_features=modality_features or {}
            )
            profile.snapshots.append(snapshot)

            # Manage snapshot count
            if len(profile.snapshots) > self.max_snapshots_per_person:
                profile.snapshots.pop(0)
                self._rebuild_index()
            else:
                self._add_to_index(person_id, len(profile.snapshots) - 1, features)

            logger.debug(f"Added new snapshot for person {person_id} (confidence: {confidence:.2f})")
        else:
            # Low confidence: average with most recent snapshot
            if profile.snapshots:
                latest_snapshot = profile.snapshots[-1]
                # Weighted average
                alpha = 0.7  # Weight for existing
                beta = 0.3   # Weight for new
                updated_features = alpha * latest_snapshot.features + beta * features
                updated_features = self._normalize(updated_features)

                latest_snapshot.features = updated_features
                latest_snapshot.timestamp = datetime.now()
                latest_snapshot.confidence = max(latest_snapshot.confidence, confidence)

                # Rebuild index to reflect changes
                self._rebuild_index()

                logger.debug(f"Averaged features for person {person_id} (confidence: {confidence:.2f})")

        profile.updated_at = datetime.now()
        profile.last_seen_camera = camera_id
        profile.last_seen_time = datetime.now()

        return profile

    def search(self, query: np.ndarray, top_k: int = 5,
               apply_temporal_decay: bool = True) -> List[MatchResult]:
        """
        Search for similar profiles in memory bank.

        Args:
            query: Query feature vector
            top_k: Number of top matches to return
            apply_temporal_decay: Whether to apply temporal decay to scores

        Returns:
            List of MatchResult objects
        """
        if self.index.ntotal == 0:
            logger.warning("Memory bank is empty")
            return []

        # Normalize query
        query_normalized = self._normalize(query).reshape(1, -1).astype('float32')

        # Search
        distances, indices = self.index.search(query_normalized, top_k)

        # Process results
        results = []
        current_time = datetime.now()

        for i, idx in enumerate(indices[0]):
            if idx == -1:  # No result
                continue

            if idx not in self.index_to_profile:
                logger.warning(f"Index {idx} not found in mapping")
                continue

            person_id, snapshot_idx = self.index_to_profile[idx]
            distance = float(distances[0][i])

            if person_id not in self.profiles:
                continue

            profile = self.profiles[person_id]
            snapshot = profile.snapshots[snapshot_idx]

            # Calculate confidence (convert distance to similarity)
            confidence = self._distance_to_similarity(distance)

            # Apply temporal decay
            if apply_temporal_decay:
                time_weight = self._calculate_temporal_weight(snapshot.timestamp, current_time)
                confidence *= time_weight

            # Create match result
            result = MatchResult(
                person_id=person_id,
                confidence=confidence,
                distance=distance,
                profile=profile,
                snapshot_used=snapshot,
                modality_similarities=snapshot.metadata.get('modality_similarities', {}),
                explanation=snapshot.metadata.get('explanation', '')
            )

            results.append(result)

        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)

        return results

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        L2 normalize vector.

        Args:
            vector: Input vector

        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert FAISS distance to similarity score.

        Args:
            distance: Distance from FAISS

        Returns:
            Similarity score between 0 and 1
        """
        # For IndexFlatIP (cosine similarity), distance is actually similarity
        # For IndexFlatL2, we need to convert
        if isinstance(self.index, faiss.IndexFlatIP):
            similarity = max(0.0, distance)
        else:
            # Convert L2 distance to similarity
            similarity = 1.0 / (1.0 + distance)

        return min(1.0, similarity)

    def _calculate_temporal_weight(self, snapshot_time: datetime,
                                   current_time: datetime) -> float:
        """
        Calculate temporal decay weight.

        Args:
            snapshot_time: Timestamp of snapshot
            current_time: Current time

        Returns:
            Temporal weight between 0 and 1
        """
        delta_seconds = (current_time - snapshot_time).total_seconds()
        weight = math.exp(-self.decay_rate * delta_seconds)
        return max(0.0, min(1.0, weight))

    def get_profile(self, person_id: int) -> Optional[PersonProfile]:
        """
        Get profile by person ID.

        Args:
            person_id: Person ID

        Returns:
            PersonProfile or None if not found
        """
        return self.profiles.get(person_id)

    def delete_profile(self, person_id: int) -> bool:
        """
        Delete profile from memory bank.

        Args:
            person_id: Person ID

        Returns:
            True if deleted, False if not found
        """
        if person_id not in self.profiles:
            return False

        del self.profiles[person_id]
        self._rebuild_index()
        logger.info(f"Deleted profile for person {person_id}")
        return True

    def update_status(self, person_id: int, status: str) -> bool:
        """
        Update person status.

        Args:
            person_id: Person ID
            status: New status ('missing', 'found', 'safe')

        Returns:
            True if updated, False if not found
        """
        if person_id not in self.profiles:
            return False

        self.profiles[person_id].status = status
        logger.info(f"Updated person {person_id} status to {status}")
        return True

    def apply_temporal_decay_to_all(self) -> None:
        """Apply temporal decay cleanup to all profiles."""
        current_time = datetime.now()
        profiles_to_delete = []

        for person_id, profile in self.profiles.items():
            # Remove snapshots with very low temporal weight
            valid_snapshots = []
            for snapshot in profile.snapshots:
                weight = self._calculate_temporal_weight(snapshot.timestamp, current_time)
                if weight > 0.1:  # Keep if weight > 10%
                    valid_snapshots.append(snapshot)

            if valid_snapshots:
                profile.snapshots = valid_snapshots
            else:
                profiles_to_delete.append(person_id)

        # Delete empty profiles
        for person_id in profiles_to_delete:
            del self.profiles[person_id]

        # Rebuild index if changes were made
        if profiles_to_delete:
            self._rebuild_index()
            logger.info(f"Applied temporal decay, removed {len(profiles_to_delete)} profiles")

    def get_statistics(self) -> Dict:
        """
        Get memory bank statistics.

        Returns:
            Dictionary of statistics
        """
        total_profiles = len(self.profiles)
        total_snapshots = sum(len(p.snapshots) for p in self.profiles.values())
        avg_snapshots = total_snapshots / total_profiles if total_profiles > 0 else 0

        status_counts = {}
        for profile in self.profiles.values():
            status = profile.status
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'total_profiles': total_profiles,
            'total_snapshots': total_snapshots,
            'avg_snapshots_per_profile': avg_snapshots,
            'index_size': self.index.ntotal,
            'status_distribution': status_counts,
            'decay_rate': self.decay_rate,
            'max_snapshots_per_person': self.max_snapshots_per_person
        }

    def save(self, filepath: str) -> None:
        """
        Save memory bank to file.

        Args:
            filepath: Path to save file
        """
        data = {
            'profiles': self.profiles,
            'embedding_dim': self.embedding_dim,
            'decay_rate': self.decay_rate,
            'max_snapshots_per_person': self.max_snapshots_per_person,
            'update_threshold': self.update_threshold
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved memory bank to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load memory bank from file.

        Args:
            filepath: Path to load file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.profiles = data['profiles']
        self.embedding_dim = data['embedding_dim']
        self.decay_rate = data['decay_rate']
        self.max_snapshots_per_person = data['max_snapshots_per_person']
        self.update_threshold = data['update_threshold']

        # Rebuild index
        self._rebuild_index()

        logger.info(f"Loaded memory bank from {filepath}")
