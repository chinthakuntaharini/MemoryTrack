"""
Headless GUI Component & Interaction Test
Verifies ForensicMainWindow, GalleryWidget, QueryWidget, TrajectoryWidget,
Search Execution, Table Population, and Safe DB Reset.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Handle PyTorch/Qt order
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from forensic_app.ui.main_window import ForensicMainWindow
from forensic_app.core.database import ForensicDatabase
from forensic_app.core.signature import ForensicSignature


class TestHeadlessGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)

    def test_main_window_full_interaction(self):
        test_db_path = PROJECT_ROOT / "database" / "test_gui_records.db"
        if test_db_path.exists():
            test_db_path.unlink()

        db = ForensicDatabase(db_path=str(test_db_path))

        # Add 3 diverse signatures
        # Person 1: Bare legs / Shorts
        sig1 = ForensicSignature(
            person_uid="UID-CAM_01-0001",
            camera_id="CAM_01",
            upper_garment={"color": "Red", "hex": "#DC2626"},
            lower_garment={"color": "Bare Legs / Shorts", "hex": "#D4B996"},
            accessories=["cap"],
            biometrics={"torso_leg_ratio": 0.65, "cadence_hz": 1.2},
            embedding=[0.05] * 512
        )
        # Person 2: Navy / Beige with backpack
        sig2 = ForensicSignature(
            person_uid="UID-CAM_01-0002",
            camera_id="CAM_01",
            upper_garment={"color": "Navy Blue", "hex": "#0F173E"},
            lower_garment={"color": "Beige", "hex": "#E0D4BB"},
            accessories=["backpack"],
            biometrics={"torso_leg_ratio": 0.58, "cadence_hz": 1.8},
            embedding=[0.10] * 512
        )
        # Person 3: Occluded / Sitting
        sig3 = ForensicSignature(
            person_uid="UID-CAM_02-0003",
            camera_id="CAM_02",
            upper_garment={"color": "White", "hex": "#F0F0F0"},
            lower_garment={"color": "None / Occluded", "hex": "#404040"},
            accessories=["handbag"],
            biometrics={"torso_leg_ratio": 0.0, "cadence_hz": 0.0},
            embedding=[0.02] * 512
        )
        db.save_signature(sig1)
        db.save_signature(sig2)
        db.save_signature(sig3)

        window = ForensicMainWindow(db=db)
        self.assertIsNotNone(window)

        # 1. Verify Database Table populated without error
        window._populate_db_table()
        self.assertEqual(window.db_table.rowCount(), 3)
        self.assertIn(window.db_table.item(0, 0).text(), ["UID-CAM_01-0001", "UID-CAM_01-0002", "UID-CAM_02-0003"])

        # 2. Test Search by Bare Legs
        query_bare = {
            "upper_color": "Any",
            "lower_color": "None / Bare Legs",
            "accessories": []
        }
        window._execute_search(query_bare)
        # Person 1 (Bare legs) and Person 3 (None / Occluded) should match!
        self.assertGreaterEqual(window.gallery_widget.cards_layout.count(), 2)

        # 3. Test Search by Backpack
        query_bp = {
            "upper_color": "Any",
            "lower_color": "Any",
            "accessories": ["backpack"]
        }
        window._execute_search(query_bp)
        self.assertGreaterEqual(window.gallery_widget.cards_layout.count(), 1)

        # 4. Test Trajectory Loading
        window.trajectory_widget.input_uid.setText("UID-CAM_01-0001")
        window.trajectory_widget.load_trajectory()
        self.assertEqual(window.trajectory_widget.table.rowCount(), 1)

        # 5. Test Safe Clear Database
        clear_success = db.clear_database()
        self.assertTrue(clear_success)
        window._populate_db_table()
        self.assertEqual(window.db_table.rowCount(), 0)

        window.close()
        db.close()
        if test_db_path.exists():
            test_db_path.unlink()

        print("[PASS] Full GUI interaction and component integration test passed with zero errors")


if __name__ == "__main__":
    unittest.main()
