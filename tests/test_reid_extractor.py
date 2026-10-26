"""
Test suite for the ReID appearance extractor (forensic_app.pipeline.reid)
Verifies:
1. Batched extraction shapes, unit norm, zero rows for invalid boxes, batch == single extraction.
2. Horizontal-flip TTA consistency and the fixed, deterministic 2048->512 projection.
3. A brightness-jittered crop of a person is closer (after centering) than a differently dressed person.
4. backbone="auto" never triggers a network download; the heuristic fallback honours the same contract.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from forensic_app.pipeline.reid import ReIDFeatureExtractor, letterbox_crop, seeded_orthogonal_projection, TORCH_AVAILABLE


def synthetic_person(shirt_bgr, pants_bgr, h=256, w=128, brightness=1.0, seed=0, stripes=False) -> np.ndarray:
    """Simple pedestrian drawing: textured background, head, torso, arms, legs (BGR)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 128, dtype=np.float32)
    img += rng.normal(0, 6, img.shape)
    cx = w // 2
    cv2.circle(img, (cx, int(0.12 * h)), int(0.08 * h), (120, 150, 200), -1)                  # head
    cv2.rectangle(img, (cx - int(0.22 * w), int(0.2 * h)), (cx + int(0.22 * w), int(0.55 * h)), shirt_bgr, -1)
    cv2.rectangle(img, (cx - int(0.32 * w), int(0.22 * h)), (cx - int(0.22 * w), int(0.5 * h)), shirt_bgr, -1)
    cv2.rectangle(img, (cx + int(0.22 * w), int(0.22 * h)), (cx + int(0.32 * w), int(0.5 * h)), shirt_bgr, -1)
    if stripes:
        for y in range(int(0.22 * h), int(0.55 * h), 10):
            cv2.line(img, (cx - int(0.22 * w), y), (cx + int(0.22 * w), y), (240, 240, 240), 2)
    cv2.rectangle(img, (cx - int(0.2 * w), int(0.55 * h)), (cx - int(0.02 * w), int(0.95 * h)), pants_bgr, -1)
    cv2.rectangle(img, (cx + int(0.02 * w), int(0.55 * h)), (cx + int(0.2 * w), int(0.95 * h)), pants_bgr, -1)
    img = img * brightness + rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _centered_cos(embs: np.ndarray, i: int, j: int) -> float:
    c = embs - embs.mean(axis=0, keepdims=True)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    return float(c[i] @ c[j])


class _ExtractorContract:
    """Shared checks run for every backbone."""

    extractor: ReIDFeatureExtractor = None

    def _frame(self):
        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        frame[100:356, 50:178] = synthetic_person((90, 30, 20), (180, 200, 220), seed=1)
        frame[120:376, 300:428] = synthetic_person((30, 30, 200), (90, 90, 90), seed=2)
        return frame

    def test_shapes_norms_and_invalid_rows(self):
        frame = self._frame()
        boxes = [(50, 100, 178, 356), (300, 120, 428, 376), (10, 10, 14, 30), (700, 10, 800, 200), (5, 5, 5, 5)]
        out = self.extractor.extract_batch(frame, boxes)
        self.assertEqual(out.shape, (5, 512))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(out[:2], axis=1), 1.0, atol=1e-4)
        self.assertTrue(np.all(out[2:] == 0.0), "Invalid / too-small / outside boxes must be zero rows")
        self.assertEqual(self.extractor.extract_batch(frame, []).shape, (0, 512))
        self.assertTrue(isinstance(self.extractor.model_id, str) and self.extractor.model_id)

    def test_batch_matches_single(self):
        frame = self._frame()
        boxes = [(50, 100, 178, 356), (300, 120, 428, 376), (40, 90, 200, 300)]
        batch = self.extractor.extract_batch(frame, boxes)
        for k, box in enumerate(boxes):
            single = self.extractor.extract(frame, box)
            self.assertEqual(single.shape, (512,))
            self.assertGreater(float(single @ batch[k]), 0.9995)

    def test_flip_consistency(self):
        person = synthetic_person((60, 140, 30), (40, 40, 40), seed=3)
        e1 = self.extractor.extract(person, (0, 0, 128, 256))
        e2 = self.extractor.extract(np.ascontiguousarray(person[:, ::-1]), (0, 0, 128, 256))
        threshold = 0.999 if self.extractor.flip_tta and self.extractor.model is not None else 0.97
        self.assertGreater(float(e1 @ e2), threshold)

    def test_same_person_closer_than_other_outfit(self):
        crops = [
            synthetic_person((90, 30, 20), (180, 200, 220), seed=10),                     # 0 person A
            synthetic_person((90, 30, 20), (180, 200, 220), seed=11, brightness=1.15),   # 1 A, brighter
            synthetic_person((30, 30, 200), (90, 90, 90), seed=12),                      # 2 red / grey
            synthetic_person((40, 160, 40), (40, 40, 40), seed=13),                      # 3 green / black
            synthetic_person((220, 220, 220), (130, 60, 20), seed=14, stripes=True),     # 4 striped / denim
            synthetic_person((20, 200, 230), (60, 60, 120), seed=15),                    # 5 yellow / maroon
        ]
        embs = np.stack([self.extractor.extract(c, (0, 0, c.shape[1], c.shape[0])) for c in crops])
        same = _centered_cos(embs, 0, 1)
        for other in range(2, len(crops)):
            self.assertGreater(same, _centered_cos(embs, 0, other), f"other={other}")

    def test_wide_crop_letterbox(self):
        # Webcam-style upper-body crop (wider than tall) must still produce a valid embedding
        wide = cv2.resize(synthetic_person((90, 30, 20), (180, 200, 220), seed=4)[:128], (220, 160))
        e = self.extractor.extract(wide, (0, 0, 220, 160))
        self.assertAlmostEqual(float(np.linalg.norm(e)), 1.0, places=4)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestReIDAutoBackbone(_ExtractorContract, unittest.TestCase):
    """Default (auto) backbone - CNN when cached weights exist."""

    @classmethod
    def setUpClass(cls):
        import torch.hub
        # Constructing the extractor must never hit the network
        with mock.patch.object(torch.hub, "load_state_dict_from_url", side_effect=AssertionError("download")), \
                mock.patch.object(torch.hub, "download_url_to_file", side_effect=AssertionError("download")):
            cls.extractor = ReIDFeatureExtractor(backbone="auto")

    def test_model_id_describes_space(self):
        ex = self.extractor
        if ex.model is None:
            self.assertTrue(ex.model_id.startswith("heuristic"))
        else:
            self.assertIn(ex.backbone_name, ex.model_id)
            self.assertIn("gem", ex.model_id)


class TestReIDHeuristicBackbone(_ExtractorContract, unittest.TestCase):
    """Handcrafted fallback descriptor."""

    @classmethod
    def setUpClass(cls):
        cls.extractor = ReIDFeatureExtractor(backbone="heuristic")

    def test_is_heuristic(self):
        self.assertIsNone(self.extractor.model)
        self.assertTrue(self.extractor.model_id.startswith("heuristic"))

    def test_black_crop_unit_norm(self):
        emb = self.extractor.extract(np.zeros((100, 50, 3), dtype=np.uint8), (0, 0, 50, 100))
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=4)


class TestReIDHelpers(unittest.TestCase):
    """Deterministic projection and letterboxing."""

    def test_projection_is_orthonormal_and_deterministic(self):
        p1 = seeded_orthogonal_projection(2048, 512)
        p2 = seeded_orthogonal_projection(2048, 512)
        self.assertEqual(p1.shape, (2048, 512))
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_allclose(p1.T @ p1, np.eye(512), atol=1e-4)

    def test_letterbox_preserves_aspect(self):
        crop = np.full((100, 100, 3), 255, dtype=np.uint8)
        canvas, (y0, x0, h, w) = letterbox_crop(crop, 256, 128, pad_color=(1, 2, 3))
        self.assertEqual(canvas.shape, (256, 128, 3))
        self.assertEqual((h, w), (128, 128))
        self.assertEqual(y0, 64)
        self.assertTrue(np.all(canvas[0, 0] == (1, 2, 3)))
        self.assertTrue(np.all(canvas[y0 + 5, 5] == 255))


if __name__ == "__main__":
    unittest.main()
