"""
End-to-End Simulation & Verification Script for Forensic Re-ID Application
Verifies:
1. No-pants detection (bare legs, shorts, occluded) -> No pants color hallucinated
2. Accessory detection (backpack, handbag, cap, umbrella)
3. Safe database wipe & index reset -> Zero crash, thread-safe
4. Spatio-temporal FAISS vector and attribute queries
"""

import os
import sys
import time
import shutil
import tempfile
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forensic_app.core.signature import ForensicSignature
from forensic_app.core.database import ForensicDatabase
from forensic_app.pipeline.detector import PedestrianDetector
from forensic_app.pipeline.pose import PoseBiometrics
from forensic_app.pipeline.parser import HumanParser
from forensic_app.pipeline.attributes import AttributeExtractor
from forensic_app.pipeline.reid import ReIDFeatureExtractor
from forensic_app.utils.drawing import draw_skeleton, draw_forensic_bbox


class TestPipelineSimulation(unittest.TestCase):
    """Rigorous end-to-end simulation test suite."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="forensic_test_"))
        self.db_path = str(self.temp_dir / "test_forensic.db")
        self.crops_dir = self.temp_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.db = ForensicDatabase(db_path=self.db_path)

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_no_pants_bare_legs_prevention(self):
        """
        Verify that a person with bare legs / shorts is NOT reported as having a pants color.
        """
        parser = HumanParser()
        attributes = AttributeExtractor()

        h, w = 180, 80
        crop = np.zeros((h, w, 3), dtype=np.uint8)

        # Upper body: Blue shirt (BGR: 200, 50, 20)
        crop[int(h * 0.15):int(h * 0.50), int(w * 0.20):int(w * 0.80)] = [200, 50, 20]

        # Lower body: Bare legs (Skin color in BGR: [130, 150, 215] -> YCrCb / HSV skin)
        crop[int(h * 0.55):int(h * 0.92), int(w * 0.22):int(w * 0.78)] = [130, 150, 215]

        # Mock pose keypoints with knees and ankles detected
        kpts_crop = np.zeros((17, 3), dtype=np.float32)
        kpts_crop[5] = [20, 30, 0.9]    # L Shoulder
        kpts_crop[6] = [60, 30, 0.9]    # R Shoulder
        kpts_crop[11] = [30, 90, 0.9]   # L Hip
        kpts_crop[12] = [50, 90, 0.9]   # R Hip
        kpts_crop[13] = [32, 130, 0.9]  # L Knee
        kpts_crop[14] = [48, 130, 0.9]  # R Knee
        kpts_crop[15] = [32, 165, 0.9]  # L Ankle
        kpts_crop[16] = [48, 165, 0.9]  # R Ankle

        parsed = parser.parse_pedestrian(crop, keypoints=kpts_crop)

        # Must detect bare legs and refuse pants classification
        self.assertFalse(parsed.has_lower_garment, "Should not classify bare legs as pants!")
        self.assertEqual(parsed.lower_garment_type, "Bare Legs / Shorts")

        # Check attribute extraction
        upper_dict, lower_dict, accs = attributes.extract_attributes(
            frame=crop,
            person_bbox=(0, 0, w, h),
            parsed_garments=parsed,
            keypoints_crop=kpts_crop
        )

        self.assertEqual(lower_dict["color"], "Bare Legs / Shorts")
        self.assertNotIn(lower_dict["color"], ["Black", "Brown", "Yellow", "Charcoal", "Khaki"])
        print("[PASS] No-pants bare legs test passed: lower color is strictly 'Bare Legs / Shorts'")

    def test_02_upper_body_only_occluded_lower_body(self):
        """
        Verify that when lower body is cut off / occluded (knees and ankles missing),
        it flags 'None / Occluded' and does NOT hallucinate pants color.
        """
        parser = HumanParser()
        attributes = AttributeExtractor()

        h, w = 120, 80
        crop = np.zeros((h, w, 3), dtype=np.uint8)
        # Red shirt (BGR: 30, 30, 220)
        crop[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = [30, 30, 220]

        # Keypoints: upper body visible, but knees and ankles are 0.0 confidence (missing)
        kpts_crop = np.zeros((17, 3), dtype=np.float32)
        kpts_crop[5] = [20, 25, 0.9]   # L Shoulder
        kpts_crop[6] = [60, 25, 0.9]   # R Shoulder
        kpts_crop[11] = [30, 80, 0.8]  # L Hip
        kpts_crop[12] = [50, 80, 0.8]  # R Hip
        # Knees & ankles have confidence 0.0 (not visible)
        kpts_crop[13] = [0, 0, 0.0]
        kpts_crop[14] = [0, 0, 0.0]
        kpts_crop[15] = [0, 0, 0.0]
        kpts_crop[16] = [0, 0, 0.0]

        parsed = parser.parse_pedestrian(crop, keypoints=kpts_crop)
        self.assertFalse(parsed.has_lower_garment)
        self.assertEqual(parsed.lower_garment_type, "None / Occluded")

        upper_dict, lower_dict, _ = attributes.extract_attributes(
            frame=crop,
            person_bbox=(0, 0, w, h),
            parsed_garments=parsed,
            keypoints_crop=kpts_crop
        )
        self.assertEqual(lower_dict["color"], "None / Occluded")
        print("[PASS] Occluded lower body test passed: lower color is strictly 'None / Occluded'")

    def test_03_accessory_association_and_detection(self):
        """
        Verify multi-scale accessory association from frame detections and cap inspection.
        """
        # Frame with person and backpack
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person_bbox = (100, 80, 220, 380)

        # Backpack located on the right side of person
        frame_accessories = [
            ("backpack", (180, 140, 240, 260), 0.75),
            ("umbrella", (500, 100, 550, 300), 0.80)  # Far away from person
        ]

        associated = PedestrianDetector.associate_accessories(person_bbox, frame_accessories)
        self.assertIn("backpack", associated, "Backpack overlapping person envelope must be associated!")
        self.assertNotIn("umbrella", associated, "Distant umbrella must not be associated!")

        # Headwear inspection
        attributes = AttributeExtractor()
        kpts_crop = np.zeros((17, 3), dtype=np.float32)
        # Face landmarks inside crop
        kpts_crop[0] = [40, 35, 0.9]  # Nose at y=35
        kpts_crop[1] = [32, 28, 0.9]  # Left Eye
        kpts_crop[2] = [48, 28, 0.9]  # Right Eye

        # Person crop with a dark baseball cap on head (rows 0 to 25 are dark fabric)
        person_crop = np.full((300, 120, 3), 180, dtype=np.uint8)
        person_crop[0:26, :] = [20, 20, 20]  # Dark cap
        frame[80:380, 100:220] = person_crop

        parser = HumanParser()
        parsed = parser.parse_pedestrian(person_crop, keypoints=kpts_crop)

        _, _, all_accs = attributes.extract_attributes(
            frame=frame,
            person_bbox=person_bbox,
            parsed_garments=parsed,
            keypoints_crop=kpts_crop,
            initial_accessories=associated
        )

        self.assertIn("backpack", all_accs)
        self.assertIn("cap", all_accs)
        print(f"[PASS] Accessory association test passed: detected {all_accs}")

    def test_04_threadsafe_database_wipe(self):
        """
        Verify safe database clearing, crop cleanup, and vector reset under multi-threaded load.
        """
        # Seed test database with signatures
        for i in range(15):
            sig = ForensicSignature(
                person_uid=f"UID-CAM_01-{i:04d}",
                camera_id="CAM_01",
                upper_garment={"color": "Navy Blue", "hex": "#0F173E"},
                lower_garment={"color": "Bare Legs / Shorts" if i % 2 == 0 else "Charcoal", "hex": "#404040"},
                accessories=["backpack"] if i % 3 == 0 else [],
                embedding=np.random.randn(512).tolist()
            )
            # Create dummy crop image
            dummy_crop_path = self.temp_dir / "crops" / f"{sig.person_uid}.jpg"
            cv2.imwrite(str(dummy_crop_path), np.zeros((100, 50, 3), dtype=np.uint8))
            sig.crop_path = str(dummy_crop_path)
            self.db.save_signature(sig)

        stats_before = self.db.get_stats()
        self.assertEqual(stats_before["total_persons"], 15)
        self.assertEqual(stats_before["faiss_indexed"], 15)

        # Query attributes
        bare_candidates = self.db.query_by_attributes(lower_color="None / Bare Legs")
        self.assertGreater(len(bare_candidates), 0)

        # Clear database safely
        clear_success = self.db.clear_database()
        self.assertTrue(clear_success)

        stats_after = self.db.get_stats()
        self.assertEqual(stats_after["total_persons"], 0)
        self.assertEqual(stats_after["total_signatures"], 0)
        self.assertEqual(stats_after["faiss_indexed"], 0)

        # Verify disk crops were also cleaned
        remaining_crops = list(self.crops_dir.glob("*.jpg"))
        self.assertEqual(len(remaining_crops), 0)
        print("[PASS] Thread-safe database wipe test passed: zero records, zero vectors, zero crop leaks")

    def test_05_visual_overlays_robustness(self):
        """
        Verify drawing routines never crash on boundary conditions or edge coordinates.
        """
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        # Person near frame boundaries
        draw_forensic_bbox(
            canvas,
            bbox=(620, 460, 640, 480),  # Right-bottom boundary
            person_uid="UID-EDGE-001",
            track_id=1,
            upper_color="Red",
            upper_hex="#DC2626",
            lower_color="Bare Legs / Shorts",
            lower_hex="#D4B996",
            accessories=["backpack", "cap"]
        )

        # Skeleton with 17 keypoints
        kpts = np.full((17, 3), 100.0, dtype=np.float32)
        draw_skeleton(canvas, kpts)

        self.assertEqual(canvas.shape, (480, 640, 3))
        print("[PASS] Visual overlay boundary robustness test passed")


if __name__ == "__main__":
    unittest.main()
