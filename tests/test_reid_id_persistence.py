"""
Test suite for Person Re-Identification & Global Identity Persistence
Verifies:
1. Same person re-appearing with different tracker IDs maintains persistent Person UID.
2. Distinct persons get unique Person UIDs without ID collision.
3. Tracker spatial association prevents ID switching during normal motion.
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forensic_app.core.signature import ForensicSignature
from forensic_app.core.database import ForensicDatabase
from forensic_app.pipeline.detector import PedestrianDetector, Tracklet


class TestReIdPersistence(unittest.TestCase):
    """Test suite for Identity Consistency across tracklet interruptions."""

    def setUp(self):
        self.db = ForensicDatabase(db_path=":memory:")

    def tearDown(self):
        self.db.close()

    def test_reid_matching_same_person(self):
        """
        Verify that a person who was previously seen matches their original UID
        even when assigned a new tracker track_id.
        """
        # Create base embedding for Person 1 (unit normalized)
        np.random.seed(42)
        base_emb_person1 = np.random.randn(512).astype(np.float32)
        base_emb_person1 /= np.linalg.norm(base_emb_person1)

        # Register Person 1 into database
        sig1 = ForensicSignature(
            person_uid="UID-CAM_01-0001",
            camera_id="CAM_01",
            upper_garment={"color": "Navy Blue", "hex": "#0F173E"},
            lower_garment={"color": "Beige", "hex": "#E0D4BB"},
            accessories=["backpack"],
            embedding=base_emb_person1.tolist()
        )
        self.db.save_signature(sig1)

        # Later frame: Person 1 re-appears from occlusion
        # Perturbed embedding (simulating pose/illumination variation with cosine sim ~0.90)
        noise = np.random.randn(512).astype(np.float32)
        noise = (noise / np.linalg.norm(noise)) * 0.40
        query_emb_person1 = base_emb_person1 + noise
        query_emb_person1 /= np.linalg.norm(query_emb_person1)

        # Call match_person
        matched_uid, conf = self.db.match_person(
            query_embedding=query_emb_person1,
            upper_color="Navy Blue",
            lower_color="Beige",
            min_similarity=0.58
        )

        self.assertEqual(matched_uid, "UID-CAM_01-0001", "Must re-associate with original person UID!")
        self.assertGreaterEqual(conf, 0.65, f"Confidence should be high, got {conf:.2f}")
        print(f"[PASS] Re-ID matched same person as {matched_uid} with confidence {conf:.2f}")

    def test_reid_distinct_persons(self):
        """
        Verify that a second, different person is NOT confused with Person 1.
        """
        np.random.seed(42)
        base_emb_person1 = np.random.randn(512).astype(np.float32)
        base_emb_person1 /= np.linalg.norm(base_emb_person1)

        sig1 = ForensicSignature(
            person_uid="UID-CAM_01-0001",
            camera_id="CAM_01",
            upper_garment={"color": "Navy Blue", "hex": "#0F173E"},
            lower_garment={"color": "Beige", "hex": "#E0D4BB"},
            accessories=["backpack"],
            embedding=base_emb_person1.tolist()
        )
        self.db.save_signature(sig1)

        # Person 2 (orthogonal / distinct vector and different clothing)
        np.random.seed(123)
        emb_person2 = np.random.randn(512).astype(np.float32)
        emb_person2 /= np.linalg.norm(emb_person2)

        matched_uid, conf = self.db.match_person(
            query_embedding=emb_person2,
            upper_color="Red",
            lower_color="Bare Legs / Shorts",
            min_similarity=0.58
        )

        self.assertIsNone(matched_uid, "Person 2 must NOT match Person 1!")
        print("[PASS] Distinct person correctly recognized as new (no false positive match)")

    def test_tracker_spatial_continuity(self):
        """
        Verify that PedestrianDetector maintains the same tracklet ID when
        a person moves between consecutive frames (center proximity & IoU fallback).
        """
        detector = PedestrianDetector()

        # Frame 1: Person at (100, 100, 160, 260)
        det_f1 = [((100, 100, 160, 260), 0.90)]
        active_f1, _ = detector._update_tracks(det_f1)
        self.assertEqual(len(active_f1), 1)
        initial_track_id = active_f1[0].track_id

        # Frame 2: Person moves 25 pixels right (125, 100, 185, 260)
        # Note: IoU drops due to fast step, but center proximity matches!
        det_f2 = [((125, 100, 185, 260), 0.88)]
        active_f2, _ = detector._update_tracks(det_f2)
        self.assertEqual(len(active_f2), 1)
        self.assertEqual(active_f2[0].track_id, initial_track_id, "Track ID must remain continuous across movement!")

        # Frame 3: Person moves another 25 pixels right (150, 100, 210, 260)
        det_f3 = [((150, 100, 210, 260), 0.87)]
        active_f3, _ = detector._update_tracks(det_f3)
        self.assertEqual(len(active_f3), 1)
        self.assertEqual(active_f3[0].track_id, initial_track_id, "Track ID must not change during walking!")
        print(f"[PASS] Tracker continuity verified: track #{initial_track_id} preserved across all motion steps")


if __name__ == "__main__":
    unittest.main()
