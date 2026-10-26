"""
Attribute Accuracy Test Suite
Verifies garment color naming (incl. shading), pose-guided parsing (bare legs vs. khaki/brown/orange pants,
webcam upper-body crops), headwear heuristics, exclusive accessory assignment, per-class accessory
thresholds, velocity-aware tracking, and batched pose estimation.
"""

import os
import sys
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forensic_app.utils.color_naming import (
    bgr_to_cielab, classify_cielab_color, color_family, FORENSIC_COLOR_PALETTE, palette_lab
)
from forensic_app.pipeline.parser import HumanParser
from forensic_app.pipeline.attributes import AttributeExtractor, accessory_person_affinity
from forensic_app.core.catalog import LOWER_GARMENT_TYPES, accessory_threshold

SKIN_BGR = (120, 150, 210)        # light-medium skin
HAIR_BGR = (28, 26, 30)           # dark hair
BG_BGR = (100, 112, 104)          # muted background


def _rgb(r, g, b):
    return (b, g, r)


def _add_noise(img: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _draw_head(img, cx, ey, d, skin=SKIN_BGR, hair=True, cap_bgr=None, hair_bgr=HAIR_BGR, mask_bgr=None):
    """Head ellipse centred on the eye line; dark hair above the hairline or a cap with a brim."""
    h, w = img.shape[:2]
    head = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(head, (int(cx), int(ey)), (int(1.2 * d), int(1.9 * d)), 0, 0, 360, 255, -1)
    img[head > 0] = skin
    yy = np.arange(h)[:, None].repeat(w, axis=1)
    if cap_bgr is not None:
        cap = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(cap, (int(cx), int(ey)), (int(1.3 * d), int(2.0 * d)), 0, 0, 360, 255, -1)
        img[(cap > 0) & (yy < ey - 0.55 * d)] = cap_bgr
        cv2.rectangle(img, (int(cx - 1.6 * d), int(ey - 0.65 * d)), (int(cx + 1.6 * d), int(ey - 0.45 * d)),
                      cap_bgr, -1)
    elif hair:
        hair_m = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(hair_m, (int(cx), int(ey)), (int(1.25 * d), int(1.95 * d)), 0, 0, 360, 255, -1)
        img[(hair_m > 0) & (yy < ey - 1.0 * d)] = hair_bgr
    if mask_bgr is not None:
        face = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(face, (int(cx), int(ey)), (int(1.15 * d), int(1.85 * d)), 0, 0, 360, 255, -1)
        img[(face > 0) & (yy > ey + 0.45 * d)] = mask_bgr
    # eyes
    for ex in (cx - 0.5 * d, cx + 0.5 * d):
        cv2.circle(img, (int(ex), int(ey)), max(1, int(0.12 * d)), (40, 40, 40), -1)


def _face_kpts(kp, cx, ey, d, conf=0.9):
    kp[0] = [cx, ey + 0.7 * d, conf]           # nose
    kp[1] = [cx + 0.5 * d, ey, conf]           # left eye (image right)
    kp[2] = [cx - 0.5 * d, ey, conf]           # right eye
    kp[3] = [cx + 1.1 * d, ey + 0.2 * d, conf]  # left ear
    kp[4] = [cx - 1.1 * d, ey + 0.2 * d, conf]  # right ear


def make_full_body(shirt_bgr, pants_bgr, legs="pants", face=True, cap_bgr=None, seed=0, noise=3.0):
    """Synthetic 120x300 full-body pedestrian crop with COCO keypoints (crop coordinates)."""
    w, h = 120, 300
    img = np.full((h, w, 3), BG_BGR, dtype=np.uint8)
    cx, d, ey = 60.0, 10.0, 30.0
    kp = np.zeros((17, 3), dtype=np.float32)

    # neck
    cv2.rectangle(img, (int(cx - 6), int(ey + 15)), (int(cx + 6), 75), SKIN_BGR, -1)
    # torso
    ls, rs = (cx + 24, 72), (cx - 24, 72)
    lh, rh = (cx + 15, 150), (cx - 15, 150)
    poly = np.array([[cx - 27, 68], [cx + 27, 68], [cx + 19, 154], [cx - 19, 154]], dtype=np.int32)
    cv2.fillPoly(img, [poly], shirt_bgr)
    # legs
    lk, rk = (cx + 14, 222), (cx - 14, 222)
    la, ra = (cx + 14, 286), (cx - 14, 286)
    leg_color = SKIN_BGR if legs == "bare" else pants_bgr
    for hip, knee, ank in ((lh, lk, la), (rh, rk, ra)):
        if legs == "shorts":
            mid = (int(hip[0]), int(hip[1] + 0.45 * (knee[1] - hip[1])))
            cv2.line(img, (int(hip[0]), int(hip[1])), mid, pants_bgr, 20)
            cv2.line(img, mid, (int(knee[0]), int(knee[1])), SKIN_BGR, 18)
            cv2.line(img, (int(knee[0]), int(knee[1])), (int(ank[0]), int(ank[1])), SKIN_BGR, 15)
        else:
            cv2.line(img, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), leg_color, 20)
            cv2.line(img, (int(knee[0]), int(knee[1])), (int(ank[0]), int(ank[1])), leg_color, 16)
        cv2.ellipse(img, (int(ank[0]), int(ank[1]) + 6), (9, 5), 0, 0, 360, (30, 30, 30), -1)
    if legs in ("pants",):
        cv2.fillPoly(img, [np.array([[lh[0], lh[1] - 4], [rh[0], rh[1] - 4], [rh[0], rh[1] + 30],
                                     [lh[0], lh[1] + 30]], dtype=np.int32)], pants_bgr)
    # arms: sleeves + bare forearms
    le, re = (cx + 31, 110), (cx - 31, 110)
    lw, rw = (cx + 33, 146), (cx - 33, 146)
    for s, e, wr in ((ls, le, lw), (rs, re, rw)):
        cv2.line(img, (int(s[0]), int(s[1])), (int(e[0]), int(e[1])), shirt_bgr, 10)
        cv2.line(img, (int(e[0]), int(e[1])), (int(wr[0]), int(wr[1])), SKIN_BGR, 8)
    _draw_head(img, cx, ey, d, hair=True, cap_bgr=cap_bgr)

    for i, p in ((5, ls), (6, rs), (7, le), (8, re), (9, lw), (10, rw), (11, lh), (12, rh),
                 (13, lk), (14, rk), (15, la), (16, ra)):
        kp[i] = [p[0], p[1], 0.9]
    _face_kpts(kp, cx, ey, d, conf=0.9 if face else 0.05)
    return _add_noise(img, noise, seed), kp


def make_webcam(shirt_bgr, seed=0, hair=True, cap_bgr=None, noise=3.0, hair_bgr=HAIR_BGR, mask_bgr=None):
    """Synthetic 240x260 webcam crop: large face at the top, neck, shirt below, no hips/legs visible."""
    w, h = 260, 240
    img = np.full((h, w, 3), BG_BGR, dtype=np.uint8)
    cx, d, ey = 130.0, 30.0, 75.0
    cv2.rectangle(img, (int(cx - 18), 120), (int(cx + 18), 170), SKIN_BGR, -1)       # neck
    poly = np.array([[40, 160], [220, 160], [250, 240], [10, 240]], dtype=np.int32)
    cv2.fillPoly(img, [poly], shirt_bgr)
    # V-neck skin below the shoulder line
    cv2.fillPoly(img, [np.array([[cx - 16, 160], [cx + 16, 160], [cx, 190]], dtype=np.int32)], SKIN_BGR)
    _draw_head(img, cx, ey, d, hair=hair, cap_bgr=cap_bgr, hair_bgr=hair_bgr, mask_bgr=mask_bgr)
    kp = np.zeros((17, 3), dtype=np.float32)
    _face_kpts(kp, cx, ey, d)
    kp[5] = [195, 165, 0.92]
    kp[6] = [65, 165, 0.92]
    kp[7] = [215, 235, 0.4]
    kp[8] = [45, 235, 0.4]
    kp[11] = [170, 330, 0.08]   # hips hallucinated below the crop with low confidence
    kp[12] = [90, 330, 0.08]
    for i in (9, 10, 13, 14, 15, 16):
        kp[i] = [130, 400, 0.02]
    return _add_noise(img, noise, seed), kp


class _Arr:
    """Minimal tensor stand-in exposing .cpu().numpy()."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __len__(self):
        return len(self.a)


class _FakeBoxes:
    def __init__(self, dets):
        self.cls = _Arr([d[0] for d in dets])
        self.conf = _Arr([d[1] for d in dets])
        self.xyxy = _Arr([d[2] for d in dets])

    def __len__(self):
        return len(self.cls)


class _FakeResult:
    def __init__(self, dets):
        self.boxes = _FakeBoxes(dets)


class _FakeYolo:
    """Fake ultralytics model returning fixed (cls, conf, xyxy) detections in input-image coordinates."""
    names = {0: "person", 24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
             39: "bottle", 67: "cell phone", 73: "laptop"}

    def __init__(self, dets):
        self.dets = dets
        self.calls = []

    def __call__(self, img, **kwargs):
        self.calls.append(kwargs)
        classes = kwargs.get("classes")
        conf = kwargs.get("conf", 0.0)
        dets = [d for d in self.dets if (classes is None or d[0] in classes) and d[1] >= conf]
        return [_FakeResult(dets)]


class TestColorClassification(unittest.TestCase):
    """LCh palette naming, including realistic shading variants."""

    TABLE = [
        # name, RGB, accepted names (first = ideal)
        ("denim jeans", (70, 95, 140), ["Denim Blue"]),
        ("light denim", (100, 125, 165), ["Denim Blue"]),
        ("faded denim", (110, 130, 160), ["Denim Blue"]),
        ("navy", (25, 35, 70), ["Navy Blue"]),
        ("royal blue", (40, 80, 200), ["Royal Blue"]),
        ("sky blue", (135, 206, 235), ["Sky Blue"]),
        ("grey", (128, 128, 128), ["Grey"]),
        ("cool grey", (120, 125, 135), ["Grey"]),
        ("light grey", (190, 190, 195), ["Light Grey"]),
        ("charcoal", (60, 62, 66), ["Charcoal"]),
        ("black", (25, 25, 28), ["Black"]),
        ("white", (235, 235, 235), ["White"]),
        ("red", (200, 30, 35), ["Red"]),
        ("maroon", (110, 20, 30), ["Maroon"]),
        ("pink", (240, 130, 170), ["Pink"]),
        ("orange", (245, 130, 32), ["Orange"]),
        ("yellow", (240, 200, 30), ["Yellow"]),
        ("khaki", (189, 172, 129), ["Khaki"]),
        ("beige", (224, 212, 187), ["Beige"]),
        ("tan", (210, 180, 140), ["Tan"]),
        ("brown", (101, 55, 30), ["Brown"]),
        ("olive", (85, 100, 45), ["Olive Green"]),
        ("green", (34, 139, 34), ["Green"]),
        ("dark green", (20, 70, 40), ["Dark Green"]),
        ("purple", (112, 48, 160), ["Purple"]),
        ("teal", (0, 128, 128), ["Teal"]),
    ]

    @staticmethod
    def _name(rgb, factor=1.0):
        bgr = np.clip(np.array(rgb[::-1], dtype=np.float32) * factor, 0, 255).astype(np.uint8)
        return classify_cielab_color(bgr_to_cielab(bgr))[0]

    def test_palette_table(self):
        for label, rgb, accepted in self.TABLE:
            with self.subTest(color=label):
                self.assertIn(self._name(rgb), accepted, f"{label} {rgb}")

    def test_shading_keeps_family(self):
        """Mild shading (x0.8) and brightening (x1.1) keep the color family."""
        # boundary colors may legitimately drift into the neighbouring family (beige <-> cream)
        adjacent = {"beige": {"neutral-light"}, "light grey": {"neutral-mid"}}
        for label, rgb, accepted in self.TABLE:
            fams = {color_family(accepted[0])} | adjacent.get(label, set())
            for f in (0.8, 1.1):
                with self.subTest(color=label, factor=f):
                    got = self._name(rgb, f)
                    self.assertIn(color_family(got), fams, f"{label} x{f} -> {got}")

    def test_typical_jeans_are_blue(self):
        for L in (32, 40, 48):
            for b in (-12, -20, -28):
                with self.subTest(L=L, b=b):
                    name, _ = classify_cielab_color(np.array([L, 2.0, b], dtype=np.float32))
                    self.assertEqual(color_family(name), "blue", f"jeans L={L} b={b} -> {name}")
        self.assertEqual(classify_cielab_color(np.array([40.0, 3.0, -24.0]))[0], "Denim Blue")

    def test_khaki_pants_never_skin_named(self):
        for rgb in ((189, 172, 129), (170, 155, 115), (210, 180, 140), (195, 180, 150)):
            name = self._name(rgb)
            self.assertIn(name, ["Khaki", "Tan", "Beige"], f"{rgb} -> {name}")

    def test_legacy_expectations(self):
        self.assertIn(classify_cielab_color(bgr_to_cielab(np.array([0, 0, 255], np.uint8)))[0], ["Red", "Maroon"])
        self.assertIn(classify_cielab_color(bgr_to_cielab(np.array([60, 20, 10], np.uint8)))[0],
                      ["Navy Blue", "Black"])
        self.assertEqual(classify_cielab_color(bgr_to_cielab(np.array([245, 245, 245], np.uint8)))[0], "White")

    def test_palette_and_families(self):
        for legacy in ("Black", "White", "Navy Blue", "Royal Blue", "Khaki", "Beige", "Charcoal", "Red",
                       "Maroon", "Olive Green", "Yellow", "Brown"):
            self.assertIn(legacy, FORENSIC_COLOR_PALETTE)
        for name, data in FORENSIC_COLOR_PALETTE.items():
            with self.subTest(palette=name):
                self.assertEqual(classify_cielab_color(np.array(data["lab"]))[0], name)
                self.assertNotEqual(color_family(name), "unknown")
                self.assertTrue(data["hex"].startswith("#") and len(data["hex"]) == 7)
        self.assertEqual(color_family("Navy Blue"), "blue")
        self.assertEqual(color_family("blue"), "blue")
        self.assertEqual(color_family("Dark Gray"), "neutral-mid")
        self.assertEqual(color_family("Bare Legs / Shorts"), "skin")
        self.assertIsNotNone(palette_lab("denim blue"))
        self.assertIsNone(palette_lab("Bare Legs / Shorts"))


class TestParserAccuracy(unittest.TestCase):
    """Pose-guided masks and the bare-legs decision."""

    def setUp(self):
        self.parser = HumanParser()
        self.attrs = AttributeExtractor()

    def _colors(self, crop, kp):
        parsed = self.parser.parse_pedestrian(crop, keypoints=kp)
        h, w = crop.shape[:2]
        upper, lower, _ = self.attrs.extract_attributes_scored(crop, (0, 0, w, h), parsed, keypoints_crop=kp)
        return parsed, upper, lower

    def test_colored_pants_are_not_bare_legs(self):
        pants = {"khaki": _rgb(189, 172, 129), "tan": _rgb(210, 180, 140), "brown": _rgb(101, 55, 30),
                 "light brown": _rgb(139, 90, 60), "orange": _rgb(230, 120, 40), "red": _rgb(170, 30, 35)}
        for face in (True, False):
            for label, bgr in pants.items():
                if not face and label == "light brown":
                    continue  # indistinguishable from medium skin without a face reference
                with self.subTest(pants=label, face=face):
                    crop, kp = make_full_body(_rgb(30, 60, 150), bgr, legs="pants", face=face, seed=3)
                    parsed, upper, lower = self._colors(crop, kp)
                    self.assertTrue(parsed.has_lower_garment, f"{label} pants judged as {parsed.lower_garment_type}")
                    self.assertEqual(lower["type"], "Pants/Jeans")
                    self.assertNotEqual(lower["color"], "Bare Legs / Shorts")
                    self.assertIsNotNone(lower["lab"])
        crop, kp = make_full_body(_rgb(200, 200, 200), _rgb(189, 172, 129), seed=4)
        _, _, lower = self._colors(crop, kp)
        self.assertIn(lower["color"], ["Khaki", "Tan", "Beige"])

    def test_bare_legs_matching_face_skin(self):
        for legs in ("bare", "shorts"):
            with self.subTest(legs=legs):
                crop, kp = make_full_body(_rgb(200, 30, 35), _rgb(40, 40, 45), legs=legs, seed=5)
                parsed, upper, lower = self._colors(crop, kp)
                self.assertFalse(parsed.has_lower_garment)
                self.assertEqual(parsed.lower_garment_type, "Bare Legs / Shorts")
                self.assertEqual(lower["color"], "Bare Legs / Shorts")
                self.assertIsNone(lower["lab"])
                self.assertIsNotNone(parsed.skin_lab, "face skin should have been sampled")
                self.assertEqual(upper["color"], "Red")

    def test_full_body_colors(self):
        cases = [(_rgb(30, 60, 150), _rgb(70, 95, 140), "blue", "blue"),
                 (_rgb(34, 139, 34), _rgb(25, 25, 28), "green", "neutral-dark"),
                 (_rgb(128, 128, 128), _rgb(189, 172, 129), "neutral-mid", "brown"),
                 (_rgb(240, 130, 170), _rgb(20, 70, 40), "pink", "green")]
        for i, (shirt, pants, fam_u, fam_l) in enumerate(cases):
            with self.subTest(case=i):
                crop, kp = make_full_body(shirt, pants, seed=10 + i)
                parsed, upper, lower = self._colors(crop, kp)
                self.assertEqual(color_family(upper["color"]), fam_u, upper)
                self.assertEqual(color_family(lower["color"]), fam_l, lower)
                self.assertGreater(upper["confidence"], 0.3)
                self.assertGreater(parsed.upper_quality, 0.3)
                self.assertGreater(parsed.lower_quality, 0.3)
                self.assertIn(lower["type"], LOWER_GARMENT_TYPES)

    def test_webcam_upper_body_reports_shirt_not_skin(self):
        shirts = {"Navy Blue": _rgb(25, 35, 70), "Grey": _rgb(128, 128, 128), "Green": _rgb(34, 139, 34),
                  "White": _rgb(235, 235, 235), "Black": _rgb(25, 25, 28)}
        for expected, bgr in shirts.items():
            with self.subTest(shirt=expected):
                crop, kp = make_webcam(bgr, seed=7)
                parsed, upper, lower = self._colors(crop, kp)
                self.assertEqual(upper["color"], expected)
                self.assertEqual(lower["color"], "None / Occluded")
                self.assertFalse(parsed.has_lower_garment)
                # torso mask must not include the face
                ys = np.where(parsed.upper_mask > 0)[0]
                self.assertGreater(ys.min(), 150)

    def test_webcam_without_keypoints_falls_back_below_face(self):
        crop, _ = make_webcam(_rgb(40, 80, 200), seed=8)
        parsed = self.parser.parse_pedestrian(crop)
        upper, _, _ = self.attrs.extract_attributes_scored(crop, (0, 0, crop.shape[1], crop.shape[0]), parsed)
        self.assertEqual(color_family(upper["color"]), "blue")

    def test_no_keypoint_skin_crop_is_not_pants(self):
        mock_crop = np.full((120, 60, 3), (120, 150, 210), dtype=np.uint8)
        parsed = self.parser.parse_pedestrian(mock_crop)
        self.assertFalse(parsed.has_lower_garment)
        self.assertIn(parsed.lower_garment_type, ["Bare Legs / Shorts", "None / Occluded"])

    def test_garment_dict_schema(self):
        crop, kp = make_full_body(_rgb(200, 30, 35), _rgb(70, 95, 140), seed=9)
        parsed, upper, lower = self._colors(crop, kp)
        for dct in (upper, lower):
            self.assertTrue({"color", "hex", "lab", "confidence"} <= set(dct))
            self.assertEqual(len(dct["lab"]), 3)
            self.assertTrue(all(isinstance(v, float) for v in dct["lab"]))
            self.assertTrue(0.0 <= dct["confidence"] <= 1.0)
        self.assertEqual(lower["type"], "Pants/Jeans")
        # legacy wrapper
        u2, l2, accs = self.attrs.extract_attributes(crop, (0, 0, crop.shape[1], crop.shape[0]), parsed,
                                                     keypoints_crop=kp, initial_accessories=["backpack"])
        self.assertEqual(u2["color"], upper["color"])
        self.assertEqual(accs, ["backpack"])


class TestAccessories(unittest.TestCase):
    """Headwear heuristics, exclusive assignment, ROI validation and per-class thresholds."""

    def setUp(self):
        self.parser = HumanParser()
        self.attrs = AttributeExtractor()

    def _scored(self, crop, kp, **kw):
        parsed = self.parser.parse_pedestrian(crop, keypoints=kp)
        h, w = crop.shape[:2]
        return self.attrs.extract_attributes_scored(crop, (0, 0, w, h), parsed, keypoints_crop=kp, **kw)[2]

    def test_dark_hair_is_not_a_cap(self):
        for seed in range(3):
            crop, kp = make_webcam(_rgb(25, 35, 70), seed=seed, hair=True)
            self.assertNotIn("cap", self._scored(crop, kp))
            crop, kp = make_full_body(_rgb(128, 128, 128), _rgb(70, 95, 140), seed=seed)
            self.assertNotIn("cap", self._scored(crop, kp))

    def test_colored_cap_is_detected(self):
        for cap in (_rgb(200, 30, 35), _rgb(40, 80, 200), _rgb(240, 240, 240)):
            with self.subTest(cap=cap):
                crop, kp = make_webcam(_rgb(128, 128, 128), seed=1, cap_bgr=cap)
                scores = self._scored(crop, kp)
                self.assertIn("cap", scores)
                self.assertGreaterEqual(scores["cap"], accessory_threshold("cap"))
        crop, kp = make_full_body(_rgb(128, 128, 128), _rgb(70, 95, 140), cap_bgr=_rgb(200, 30, 35), seed=2)
        self.assertIn("cap", self._scored(crop, kp))

    def test_grey_and_brown_hair_are_not_caps(self):
        for hair in (_rgb(150, 150, 150), _rgb(205, 205, 200), _rgb(95, 60, 35), _rgb(190, 160, 110)):
            with self.subTest(hair=hair):
                crop, kp = make_webcam(_rgb(25, 35, 70), seed=4, hair_bgr=hair)
                self.assertNotIn("cap", self._scored(crop, kp))

    def test_face_mask(self):
        for mask in (_rgb(235, 235, 235), _rgb(150, 190, 225), _rgb(25, 25, 28)):
            with self.subTest(mask=mask):
                crop, kp = make_webcam(_rgb(128, 128, 128), seed=5, mask_bgr=mask)
                self.assertIn("face mask", self._scored(crop, kp))
        crop, kp = make_webcam(_rgb(128, 128, 128), seed=5)
        scores = self._scored(crop, kp)
        self.assertNotIn("face mask", scores)
        self.assertNotIn("glasses", scores)

    def test_no_face_no_headwear(self):
        crop, kp = make_full_body(_rgb(128, 128, 128), _rgb(70, 95, 140), face=False, cap_bgr=_rgb(200, 30, 35))
        self.assertNotIn("cap", self._scored(crop, kp))

    def test_assign_accessories_exclusive(self):
        from forensic_app.pipeline.detector import PedestrianDetector
        boxes = {1: (100, 100, 180, 400), 2: (200, 100, 280, 400)}
        kp1 = np.zeros((17, 3), np.float32)
        kp2 = np.zeros((17, 3), np.float32)
        for kp, x0 in ((kp1, 100), (kp2, 200)):
            kp[5] = [x0 + 60, 160, 0.9]
            kp[6] = [x0 + 20, 160, 0.9]
            kp[11] = [x0 + 52, 250, 0.9]
            kp[12] = [x0 + 28, 250, 0.9]
        kp1[9] = [176, 262, 0.9]   # person 1 holds the bag (right side of their box)
        kp1[10] = [108, 260, 0.9]
        kp2[9] = [276, 262, 0.9]
        kp2[10] = [212, 330, 0.9]
        handbag = ("handbag", (168, 250, 212, 300), 0.6)
        out = PedestrianDetector.assign_accessories(boxes, [handbag], {1: kp1, 2: kp2})
        self.assertIn("handbag", out[1])
        self.assertNotIn("handbag", out[2])
        # Without keypoints it still goes to exactly one person
        out = PedestrianDetector.assign_accessories(boxes, [handbag])
        self.assertEqual(sum("handbag" in v for v in out.values()), 1)
        # A backpack on person 2's torso goes to person 2 only
        backpack = ("backpack", (210, 150, 270, 260), 0.5)
        out = PedestrianDetector.assign_accessories(boxes, [handbag, backpack], {1: kp1, 2: kp2})
        self.assertIn("backpack", out[2])
        self.assertNotIn("backpack", out[1])
        # Far away item -> nobody
        out = PedestrianDetector.assign_accessories(boxes, [("suitcase", (500, 300, 560, 380), 0.8)])
        self.assertEqual(out, {1: {}, 2: {}})

    def test_affinity_geometry(self):
        person = (100, 100, 180, 400)
        self.assertGreater(accessory_person_affinity("umbrella", (90, 20, 190, 110), person), 0.5)
        self.assertEqual(accessory_person_affinity("tie", (300, 150, 320, 200), person), 0.0)
        self.assertEqual(accessory_person_affinity("backpack", (0, 0, 640, 480), person), 0.0)  # too large

    def test_per_class_thresholds_drop_weak_bottle(self):
        from forensic_app.pipeline.detector import detect_accessories_with_model, PedestrianDetector
        frame = np.zeros((480, 640, 3), np.uint8)
        fake = _FakeYolo([(39, 0.25, (10, 10, 30, 50)), (26, 0.32, (40, 60, 80, 110)),
                          (39, 0.55, (90, 60, 100, 90))])
        dets = detect_accessories_with_model(fake, frame, (100, 100, 300, 400))
        names = sorted((n, round(c, 2)) for n, _, c in dets)
        self.assertEqual(names, [("bottle", 0.55), ("handbag", 0.32)])
        # boxes are mapped back to frame coordinates
        hb = [b for n, b, _ in dets if n == "handbag"][0]
        self.assertEqual(hb, (140, 160, 180, 210))

        # Full-frame path: a 0.25 bottle and 0.30 person-less noise are dropped, person kept
        det = PedestrianDetector.__new__(PedestrianDetector)
        det.conf_thresh, det.iou_thresh, det.device = 0.45, 0.40, "cpu"
        det.archive_lost_threshold = 60
        det.tracklets, det.next_track_id = {}, 1
        det.model = _FakeYolo([(0, 0.9, (100, 100, 180, 400)), (39, 0.25, (150, 250, 160, 280)),
                               (24, 0.31, (110, 150, 170, 260))])
        det.class_map = {k: v for k, v in _FakeYolo.names.items() if k != 0}
        active, _, accs = det.detect_and_track(frame)
        self.assertEqual(len(active), 1)
        self.assertEqual([a[0] for a in accs], ["backpack"])

    def test_roi_pass_validates_against_neighbours(self):
        frame = np.full((480, 640, 3), 90, np.uint8)
        crop_box = (100, 100, 180, 400)
        neighbour = (190, 100, 270, 400)

        class _Det:
            def detect_accessories_in_roi(self, frame, roi):
                return [("handbag", (230, 250, 262, 300), 0.7),   # in the neighbour's hand
                        ("backpack", (110, 150, 170, 260), 0.5),  # on this person's back
                        ("bottle", (120, 200, 130, 230), 0.3)]    # below per-class threshold

        parsed = HumanParser().parse_pedestrian(frame[100:400, 100:180])
        _, _, accs = self.attrs.extract_attributes_scored(frame, crop_box, parsed, accessory_detector=_Det(),
                                                          exclude_boxes=[neighbour])
        self.assertIn("backpack", accs)
        self.assertNotIn("handbag", accs)
        self.assertNotIn("bottle", accs)
        # initial dict + ROI confirmation combine (noisy-or)
        _, _, accs2 = self.attrs.extract_attributes_scored(frame, crop_box, parsed, accessory_detector=_Det(),
                                                           initial_accessories={"backpack": 0.4},
                                                           exclude_boxes=[neighbour])
        self.assertGreater(accs2["backpack"], accs["backpack"])


class TestTrackerAndPose(unittest.TestCase):
    """Velocity-aware tracking and batched pose estimation."""

    def test_tracker_velocity_40px_per_frame(self):
        from forensic_app.pipeline.detector import PedestrianDetector
        det = PedestrianDetector.__new__(PedestrianDetector)
        det.tracklets, det.next_track_id, det.archive_lost_threshold = {}, 1, 60
        x = 100
        active, _ = det._update_tracks([((x, 100, x + 60, 260), 0.9)])
        tid = active[0].track_id
        for step in range(1, 8):
            x += 40
            active, _ = det._update_tracks([((x, 100, x + 60, 260), 0.9)])
            self.assertEqual([t.track_id for t in active], [tid], f"step {step}")
        self.assertAlmostEqual(det.tracklets[tid].velocity[0], 40.0, delta=4.0)
        # Two missed frames, then the person reappears where constant velocity predicts
        det._update_tracks([])
        det._update_tracks([])
        x += 3 * 40
        active, _ = det._update_tracks([((x, 100, x + 60, 260), 0.9)])
        self.assertIn(tid, [t.track_id for t in active])
        # A second person entering far away gets a new id
        active, _ = det._update_tracks([((x + 40, 100, x + 100, 260), 0.9), ((10, 100, 70, 260), 0.9)])
        ids = sorted(t.track_id for t in active)
        self.assertIn(tid, ids)
        self.assertEqual(len(ids), 2)

    def test_estimate_pose_batch_one_result_per_bbox(self):
        from forensic_app.pipeline.pose import PoseBiometrics
        pose = PoseBiometrics(fps=30.0)
        self.assertEqual(pose.fps, 30.0)
        if pose.model is None:
            self.skipTest("pose model not available locally")
        frame = np.full((480, 640, 3), 100, np.uint8)
        crop, _ = make_full_body(_rgb(200, 30, 35), _rgb(70, 95, 140), seed=1, noise=0)
        frame[100:400, 50:170] = crop
        frame[100:400, 300:420] = crop
        bboxes = [(50, 100, 170, 400), (300, 100, 420, 400), (600, 10, 610, 20)]
        out = pose.estimate_pose_batch(frame, bboxes, [1, 2, 3])
        self.assertEqual(len(out), 3)
        for kf, kc, bio, posture in out:
            self.assertIsInstance(bio, dict)
            self.assertIsInstance(posture, str)
            if kf is not None:
                self.assertEqual(kf.shape, (17, 3))
                self.assertEqual(kc.shape, (17, 3))
        self.assertIsNone(out[2][0])  # too small -> default result
        single = pose.estimate_pose(frame, bboxes[0], 1)
        self.assertEqual(len(single), 4)


if __name__ == "__main__":
    unittest.main()
