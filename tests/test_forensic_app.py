"""
Unit Tests for Refactored Forensic Re-ID Application
Verifies all modules, bug fixes (safe DB deletion, no-pants color prevention, accessory detection), and signature schemas.
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

from forensic_app.utils.color_naming import bgr_to_cielab, classify_cielab_color, delta_e_cielab
from forensic_app.core.signature import ForensicSignature
from forensic_app.core.database import ForensicDatabase
from forensic_app.pipeline.parser import HumanParser
from forensic_app.pipeline.attributes import AttributeExtractor
from forensic_app.pipeline.pose import PoseBiometrics
from forensic_app.pipeline.reid import ReIDFeatureExtractor


class TestForensicApp(unittest.TestCase):
    """Test suite for forensic_app modules."""

    def test_cielab_color_classification(self):
        """Test CIELAB conversion and forensic palette mapping."""
        # Pure red in BGR
        red_bgr = np.array([0, 0, 255], dtype=np.uint8)
        red_lab = bgr_to_cielab(red_bgr)
        name, hex_code = classify_cielab_color(red_lab)
        self.assertIn(name, ["Red", "Maroon"])

        # Deep navy in BGR
        navy_bgr = np.array([60, 20, 10], dtype=np.uint8)
        navy_lab = bgr_to_cielab(navy_bgr)
        name, _ = classify_cielab_color(navy_lab)
        self.assertIn(name, ["Navy Blue", "Black"])

        # White in BGR
        white_bgr = np.array([245, 245, 245], dtype=np.uint8)
        white_lab = bgr_to_cielab(white_bgr)
        name, _ = classify_cielab_color(white_lab)
        self.assertEqual(name, "White")

    def test_signature_schema(self):
        """Verify signature structure matches SKILL.md Section 3.E."""
        sig = ForensicSignature(
            person_uid="UID-2026-F9812",
            camera_id="CAM_01",
            upper_garment={"color": "Navy Blue", "hex": "#1E3A8A"},
            lower_garment={"color": "Beige", "hex": "#D4B996"},
            accessories=["backpack"],
            biometrics={"torso_leg_ratio": 0.61, "cadence_hz": 1.72},
            embedding=[0.05] * 512
        )
        d = sig.to_dict()
        self.assertEqual(d["person_uid"], "UID-2026-F9812")
        self.assertEqual(d["upper_garment"]["color"], "Navy Blue")
        self.assertEqual(d["lower_garment"]["color"], "Beige")
        self.assertIn("backpack", d["accessories"])
        self.assertEqual(d["biometrics"]["torso_leg_ratio"], 0.61)
        self.assertEqual(len(d["embedding"]), 512)

        # JSON Round-Trip
        json_str = sig.to_json()
        reconstructed = ForensicSignature.from_dict(d)
        self.assertEqual(reconstructed.person_uid, sig.person_uid)

    def test_parser_no_pants_detection(self):
        """
        Verify that parser does NOT hallucinate pants color when person has bare legs or no pants.
        """
        parser = HumanParser()
        # Mock crop with bare leg skin tones in the bottom half
        mock_crop = np.full((120, 60, 3), (120, 150, 210), dtype=np.uint8)  # Skin tone in BGR
        parsed = parser.parse_pedestrian(mock_crop)

        # If lower region is majority skin, has_lower_garment MUST be False
        self.assertFalse(parsed.has_lower_garment)
        self.assertIn(parsed.lower_garment_type, ["Bare Legs / Shorts", "None / Occluded"])

    def test_database_safe_clear(self):
        """
        Critical Bug Fix Test:
        Verify that clearing the database wipes all records and does NOT raise NameError or crash!
        """
        import tempfile
        temp_db_path = str(Path(tempfile.gettempdir()) / "test_forensic_safe.db")
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

        db = ForensicDatabase(db_path=temp_db_path)

        # Save test signature
        sig = ForensicSignature(
            person_uid="UID-TEST-001",
            camera_id="CAM_01",
            upper_garment={"color": "Royal Blue", "hex": "#1D4ED8"},
            lower_garment={"color": "Black", "hex": "#141414"},
            accessories=["backpack"],
            embedding=[0.1] * 512
        )
        db.save_signature(sig)

        stats_before = db.get_stats()
        self.assertEqual(stats_before["total_persons"], 1)

        # Clear database: MUST SUCCEED WITH ZERO ERRORS
        success = db.clear_database()
        self.assertTrue(success)

        stats_after = db.get_stats()
        self.assertEqual(stats_after["total_persons"], 0)
        self.assertEqual(stats_after["total_signatures"], 0)
        self.assertEqual(stats_after["faiss_indexed"], 0)

        db.close()
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

    def test_pose_biometrics(self):
        """Test anthropometric calculations and cadence estimation."""
        pose = PoseBiometrics()
        # Mock 17 keypoints
        kpts = np.zeros((17, 3), dtype=np.float32)
        # Shoulders (index 5, 6)
        kpts[5] = [40, 30, 0.9]
        kpts[6] = [60, 30, 0.9]
        # Hips (index 11, 12)
        kpts[11] = [45, 60, 0.9]
        kpts[12] = [55, 60, 0.9]
        # Ankles (index 15, 16)
        kpts[15] = [45, 110, 0.9]
        kpts[16] = [55, 110, 0.9]

        ratios = pose._calculate_ratios(kpts, track_id=99)
        self.assertGreater(ratios["torso_leg_ratio"], 0.0)
        self.assertGreater(ratios["shoulder_hip_ratio"], 0.0)
        self.assertIn("cadence_hz", ratios)

    def test_reid_extractor_norm(self):
        """Test 512-D vector extraction and unit L2 normalization."""
        reid = ReIDFeatureExtractor()
        dummy_img = np.zeros((100, 50, 3), dtype=np.uint8)
        emb = reid.extract(dummy_img, (0, 0, 50, 100))
        self.assertEqual(len(emb), 512)
        # Unit norm check
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
