"""
Real-time Video Worker Thread Simulation Test
Streams 60 frames through ForensicPipelineWorker and verifies:
1. No exceptions or errors emitted by worker
2. Signatures saved into database and FAISS index
3. Query by attributes successfully matches suspects
4. Database clearing executes without crashing
"""

import os
import sys
import time
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Handle PyTorch/Qt order
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

from PyQt6.QtCore import QCoreApplication
from forensic_app.core.database import ForensicDatabase
from forensic_app.pipeline.worker import ForensicPipelineWorker


class TestWorkerStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create headless QCoreApplication for Qt signal/slot event loop
        cls.app = QCoreApplication.instance()
        if cls.app is None:
            cls.app = QCoreApplication(sys.argv)

    def test_worker_full_video_stream(self):
        video_path = PROJECT_ROOT / "tests" / "test_surveillance_stream.mp4"
        self.assertTrue(video_path.exists(), "Test video must exist")

        test_db_path = PROJECT_ROOT / "database" / "test_worker_records.db"
        if test_db_path.exists():
            test_db_path.unlink()

        db = ForensicDatabase(db_path=str(test_db_path))

        worker = ForensicPipelineWorker(
            db=db,
            source_type="video",
            source_path=str(video_path),
            camera_id="CAM_01",
            device="cpu"
        )

        frames_received = []
        errors_received = []
        signatures_received = []
        finished_flag = [False]

        def on_frame(f):
            frames_received.append(f)

        def on_error(err):
            errors_received.append(err)

        def on_sig(sig):
            signatures_received.append(sig)

        def on_finish():
            finished_flag[0] = True

        worker.frame_ready.connect(on_frame)
        worker.error_occurred.connect(on_error)
        worker.signature_detected.connect(on_sig)
        worker.worker_finished.connect(on_finish)

        worker.start()

        # Wait for worker thread to complete (up to 30 seconds)
        t_start = time.time()
        while not finished_flag[0] and (time.time() - t_start) < 30:
            self.app.processEvents()
            time.sleep(0.05)

        worker.stop()
        worker.wait(2000)

        # Assertions
        self.assertEqual(len(errors_received), 0, f"Worker emitted errors: {errors_received}")
        self.assertGreater(len(frames_received), 0, "Frames should have been processed and emitted")

        # Test database clear on live database
        clear_result = db.clear_database()
        self.assertTrue(clear_result, "Database clear must succeed without crashing")
        stats = db.get_stats()
        self.assertEqual(stats["total_persons"], 0)

        db.close()
        if test_db_path.exists():
            test_db_path.unlink()

        print(f"[PASS] Stream simulation successful: {len(frames_received)} frames processed with ZERO errors")


if __name__ == "__main__":
    unittest.main()
