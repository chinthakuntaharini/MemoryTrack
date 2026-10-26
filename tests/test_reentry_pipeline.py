"""
End-to-End Re-Entry Identity Test
Drives the REAL ForensicPipelineWorker loop (real parser, attributes, ReID extractor, IdentityManager,
database and drawing) over a synthetic video in which a person leaves the camera view and comes back,
and verifies that the SAME persistent UID is shown again - within one session and after a restart.
Only detection and pose are replaced by ground-truth fakes (YOLO does not detect drawn figures).
"""

import os
import sys
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Handle PyTorch/Qt order (as the app does)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch  # noqa: F401,E402

from PyQt6.QtCore import QCoreApplication  # noqa: E402

from forensic_app.core.database import ForensicDatabase  # noqa: E402
from forensic_app.core.identity import IdentityManager, TENTATIVE_LABEL  # noqa: E402
from forensic_app.pipeline import worker as worker_mod  # noqa: E402
from forensic_app.pipeline.detector import PedestrianDetector as RealDetector, Tracklet  # noqa: E402
from forensic_app.utils.color_naming import color_family  # noqa: E402

FRAME_W, FRAME_H = 640, 480
SKIN_BGR = (120, 150, 210)
HAIR_BGR = (28, 26, 30)
BAG_BGR = (48, 46, 52)
SPRITE_W, SPRITE_H = 120, 300


# =====================================================================================  synthetic scene
@dataclass
class PersonSpec:
    """Appearance of one drawn person."""
    name: str
    shirt: Tuple[int, int, int]
    pants: Tuple[int, int, int]
    backpack: bool = False


PERSON_A = PersonSpec("A", shirt=(40, 40, 190), pants=(140, 95, 60), backpack=True)   # red shirt, denim, backpack
PERSON_B = PersonSpec("B", shirt=(60, 150, 60), pants=(130, 170, 195))                # green shirt, khaki


def render_person(spec: PersonSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Draw a full-body figure on a 120x300 sprite. Returns (bgr, mask, COCO-17 keypoints, backpack box)."""
    img = np.zeros((SPRITE_H, SPRITE_W, 3), dtype=np.uint8)
    mask = np.zeros((SPRITE_H, SPRITE_W), dtype=np.uint8)

    def paint(fn, color):
        fn(img, color)
        fn(mask, 255)

    cx, d, ey = 60.0, 10.0, 30.0
    kp = np.zeros((17, 3), dtype=np.float32)

    # neck + torso
    paint(lambda im, c: cv2.rectangle(im, (int(cx - 6), int(ey + 15)), (int(cx + 6), 75), c, -1), SKIN_BGR)
    torso = np.array([[cx - 27, 68], [cx + 27, 68], [cx + 19, 154], [cx - 19, 154]], dtype=np.int32)
    paint(lambda im, c: cv2.fillPoly(im, [torso], c), spec.shirt)

    # legs (pants)
    ls, rs = (cx + 24, 72), (cx - 24, 72)
    lh, rh = (cx + 15, 150), (cx - 15, 150)
    lk, rk = (cx + 14, 222), (cx - 14, 222)
    la, ra = (cx + 14, 286), (cx - 14, 286)
    for hip, knee, ank in ((lh, lk, la), (rh, rk, ra)):
        paint(lambda im, c, a=hip, b=knee: cv2.line(im, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c, 20),
              spec.pants)
        paint(lambda im, c, a=knee, b=ank: cv2.line(im, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c, 16),
              spec.pants)
        paint(lambda im, c, a=ank: cv2.ellipse(im, (int(a[0]), int(a[1]) + 6), (9, 5), 0, 0, 360, c, -1),
              (30, 30, 30))
    hip_band = np.array([[lh[0], lh[1] - 4], [rh[0], rh[1] - 4], [rh[0], rh[1] + 30], [lh[0], lh[1] + 30]],
                        dtype=np.int32)
    paint(lambda im, c: cv2.fillPoly(im, [hip_band], c), spec.pants)

    # backpack seen from the side (covers the side of the torso)
    bag = None
    if spec.backpack:
        bag = (int(cx + 14), 76, int(cx + 40), 148)
        paint(lambda im, c: cv2.rectangle(im, bag[:2], bag[2:], c, -1), BAG_BGR)
        paint(lambda im, c: cv2.line(im, (int(cx + 16), 72), (int(cx + 6), 120), c, 4), BAG_BGR)  # strap

    # arms: sleeves + bare forearms
    le, re = (cx + 31, 110), (cx - 31, 110)
    lw, rw = (cx + 33, 146), (cx - 33, 146)
    for s, e, wr in ((ls, le, lw), (rs, re, rw)):
        paint(lambda im, c, a=s, b=e: cv2.line(im, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c, 10),
              spec.shirt)
        paint(lambda im, c, a=e, b=wr: cv2.line(im, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c, 8),
              SKIN_BGR)

    # head: skin ellipse, dark hair above the hairline, eyes
    head = np.zeros((SPRITE_H, SPRITE_W), dtype=np.uint8)
    cv2.ellipse(head, (int(cx), int(ey)), (int(1.2 * d), int(1.9 * d)), 0, 0, 360, 255, -1)
    img[head > 0] = SKIN_BGR
    mask[head > 0] = 255
    hair = np.zeros_like(head)
    cv2.ellipse(hair, (int(cx), int(ey)), (int(1.25 * d), int(1.95 * d)), 0, 0, 360, 255, -1)
    yy = np.arange(SPRITE_H)[:, None].repeat(SPRITE_W, axis=1)
    img[(hair > 0) & (yy < ey - 1.0 * d)] = HAIR_BGR
    mask[hair > 0] = 255
    for ex in (cx - 0.5 * d, cx + 0.5 * d):
        cv2.circle(img, (int(ex), int(ey)), 1, (40, 40, 40), -1)

    for i, p in ((5, ls), (6, rs), (7, le), (8, re), (9, lw), (10, rw), (11, lh), (12, rh),
                 (13, lk), (14, rk), (15, la), (16, ra)):
        kp[i] = [p[0], p[1], 0.9]
    kp[0] = [cx, ey + 0.7 * d, 0.9]
    kp[1] = [cx + 0.5 * d, ey, 0.9]
    kp[2] = [cx - 0.5 * d, ey, 0.9]
    kp[3] = [cx + 1.1 * d, ey + 0.2 * d, 0.9]
    kp[4] = [cx - 1.1 * d, ey + 0.2 * d, 0.9]
    return img, mask, kp, bag


def make_background(seed: int = 7) -> np.ndarray:
    """Static corridor-like background (gradient wall, floor band, door, poster)."""
    rng = np.random.default_rng(seed)
    bg = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.float32)
    grad = np.linspace(0.0, 1.0, FRAME_H, dtype=np.float32)[:, None]
    bg[:] = np.stack([96 + 18 * grad, 104 + 10 * grad, 100 + 6 * grad], axis=-1)
    cv2.rectangle(bg, (0, 400), (FRAME_W, FRAME_H), (78, 82, 88), -1)       # floor
    cv2.rectangle(bg, (470, 120), (560, 400), (70, 88, 112), -1)            # door
    cv2.rectangle(bg, (250, 90), (330, 170), (150, 120, 90), -1)            # poster
    bg += rng.normal(0.0, 4.0, bg.shape).astype(np.float32)
    return np.clip(bg, 0, 255).astype(np.uint8)


def paste_person(frame: np.ndarray, sprite, cx: float, top: float, scale: float) -> Optional[Dict[str, Any]]:
    """Composite a sprite into the frame (clipped at the borders). Returns ground truth or None if unseen."""
    img, mask, kp, bag = sprite
    sw, sh = int(round(SPRITE_W * scale)), int(round(SPRITE_H * scale))
    im = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    mk = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    x0, y0 = int(round(cx - sw / 2.0)), int(round(top))
    fx1, fy1 = max(0, x0), max(0, y0)
    fx2, fy2 = min(FRAME_W, x0 + sw), min(FRAME_H, y0 + sh)
    if fx2 - fx1 < 4 or fy2 - fy1 < 4:
        return None
    sub = mk[fy1 - y0:fy2 - y0, fx1 - x0:fx2 - x0] > 0
    if not sub.any():
        return None
    region = frame[fy1:fy2, fx1:fx2]
    region[sub] = im[fy1 - y0:fy2 - y0, fx1 - x0:fx2 - x0][sub]
    visible = float(sub.sum()) / float(max(1, (mk > 0).sum()))
    ys, xs = np.nonzero(sub)
    pad = 3
    bbox = (max(0, fx1 + int(xs.min()) - pad), max(0, fy1 + int(ys.min()) - pad),
            min(FRAME_W, fx1 + int(xs.max()) + 1 + pad), min(FRAME_H, fy1 + int(ys.max()) + 1 + pad))
    kpf = kp.copy()
    kpf[:, 0] = kp[:, 0] * scale + x0
    kpf[:, 1] = kp[:, 1] * scale + y0
    outside = (kpf[:, 0] < 0) | (kpf[:, 0] >= FRAME_W) | (kpf[:, 1] < 0) | (kpf[:, 1] >= FRAME_H)
    kpf[outside, 2] = 0.0
    bag_f = None
    if bag is not None:
        bx1, by1 = bag[0] * scale + x0, bag[1] * scale + y0
        bx2, by2 = bag[2] * scale + x0, bag[3] * scale + y0
        bx1, bx2 = max(0.0, bx1), min(float(FRAME_W), bx2)
        if bx2 - bx1 >= 6:
            bag_f = (int(bx1), int(by1), int(bx2), int(by2))
    return {"bbox": bbox, "visible": visible, "kp": kpf, "bag": bag_f}


@dataclass
class Appearance:
    """One person's presence in a frame (ground truth for the fake detector / pose)."""
    name: str
    track_id: int
    bbox: Tuple[int, int, int, int]
    visible: float
    kp: np.ndarray
    bag: Optional[Tuple[int, int, int, int]]


class Scene:
    """Ground-truth timeline shared by the fake detector and the fake pose estimator."""

    def __init__(self):
        self.frames: List[List[Appearance]] = []
        self.archive_at: Dict[int, List[int]] = {}   # frame index -> track ids the tracker archives
        self.hooks: Dict[int, Any] = {}               # frame index -> callable run when that frame is detected
        self.cursor = -1

    @property
    def current(self) -> List[Appearance]:
        return self.frames[self.cursor] if 0 <= self.cursor < len(self.frames) else []


def build_video(path: Path, timeline, n_frames: int, archive_at: Optional[Dict[int, List[int]]] = None,
                seed: int = 11) -> Scene:
    """
    Render `n_frames` frames. `timeline(t)` returns [(spec, track_id, cx, top, scale)] for frame t.
    Writes an MJPG AVI at 25 fps and returns the ground-truth scene.
    """
    rng = np.random.default_rng(seed)
    bg = make_background()
    sprites = {}
    scene = Scene()
    scene.archive_at = dict(archive_at or {})
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (FRAME_W, FRAME_H))
    assert writer.isOpened(), "VideoWriter (MJPG) unavailable"
    for t in range(n_frames):
        frame = bg.copy()
        apps: List[Appearance] = []
        for spec, track_id, cx, top, scale in timeline(t):
            sprite = sprites.get(spec.name)
            if sprite is None:
                sprite = sprites[spec.name] = render_person(spec)
            gt = paste_person(frame, sprite, cx, top, scale)
            if gt is not None:
                apps.append(Appearance(spec.name, track_id, gt["bbox"], gt["visible"], gt["kp"], gt["bag"]))
        noisy = frame.astype(np.float32) + rng.normal(0.0, 2.5, frame.shape)
        writer.write(np.clip(noisy, 0, 255).astype(np.uint8))
        scene.frames.append(apps)
    writer.release()
    return scene


# =====================================================================================  fakes
def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class FakeDetector:
    """Ground-truth 'detector + tracker': reports people that are at least half visible."""

    scene: Scene = None  # set by the test
    MIN_VISIBLE = 0.5

    def __init__(self, *args, **kwargs):
        self.model = object()                 # "weights loaded"
        self.accessory_model_name = "ground-truth"
        self.hits: Dict[int, int] = {}
        self.last_box: Dict[int, Tuple[int, int, int, int]] = {}
        self.roi_calls = 0

    def detect_and_track(self, frame):
        sc = self.scene
        sc.cursor += 1
        hook = sc.hooks.get(sc.cursor)
        if hook is not None:
            hook()
        active, frame_accs = [], []
        for app in sc.current:
            if app.visible < self.MIN_VISIBLE:
                continue
            self.hits[app.track_id] = self.hits.get(app.track_id, 0) + 1
            self.last_box[app.track_id] = app.bbox
            active.append(Tracklet(track_id=app.track_id, bbox=app.bbox, confidence=0.86,
                                   hits=self.hits[app.track_id]))
            if app.bag is not None and sc.cursor % 2 == 0:
                frame_accs.append(("backpack", app.bag, 0.55))
        archived = [Tracklet(track_id=tid, bbox=self.last_box.get(tid, (0, 0, 10, 10)), confidence=0.5,
                             archived=True) for tid in sc.archive_at.get(sc.cursor, [])]
        return active, archived, frame_accs

    def detect_accessories_in_roi(self, frame, roi_bbox, imgsz=None):
        self.roi_calls += 1
        out = []
        x1, y1, x2, y2 = roi_bbox
        for app in self.scene.current:
            if app.bag is None:
                continue
            cx, cy = (app.bag[0] + app.bag[2]) / 2.0, (app.bag[1] + app.bag[3]) / 2.0
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                out.append(("backpack", app.bag, 0.62))
        return out

    assign_accessories = staticmethod(RealDetector.assign_accessories)
    associate_accessories = staticmethod(RealDetector.associate_accessories)


class FakePose:
    """Ground-truth keypoints for each tracked box (same output layout as PoseBiometrics)."""

    scene: Scene = None

    def __init__(self, *args, **kwargs):
        self.fps = 25.0

    def set_fps(self, fps):
        self.fps = float(fps)

    def forget_track(self, track_id):
        pass

    def estimate_pose_batch(self, frame, bboxes, track_ids=None, imgsz=-1):
        out = []
        for bbox in bboxes:
            best = max(self.scene.current, key=lambda a: _iou(a.bbox, bbox), default=None)
            if best is None or _iou(best.bbox, bbox) < 0.3:
                out.append((None, None, {"torso_leg_ratio": 0.0, "shoulder_hip_ratio": 0.0, "cadence_hz": 0.0},
                            "Standing"))
                continue
            kf = best.kp.copy()
            kc = kf.copy()
            kc[:, 0] -= bbox[0]
            kc[:, 1] -= bbox[1]
            out.append((kf, kc, {"torso_leg_ratio": 0.48, "shoulder_hip_ratio": 1.6, "cadence_hz": 1.4}, "Walking"))
        return out

    def estimate_pose(self, frame, bbox, track_id=0):
        return self.estimate_pose_batch(frame, [bbox], [track_id])[0]


class RecordingIdentityManager(IdentityManager):
    """Real IdentityManager that records every frame's assignments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history: List[Tuple[int, Dict[int, Tuple[Optional[str], str, str]]]] = []

    def update(self, observations, frame_idx):
        out = super().update(observations, frame_idx)
        self.history.append((frame_idx, {tid: (a.person_uid, a.status, a.display_label) for tid, a in out.items()}))
        return out


@dataclass
class RunResult:
    worker: Any
    history: List[Tuple[int, Dict[int, Tuple[Optional[str], str, str]]]]
    frames: int
    errors: List[str]
    events: List[Dict[str, Any]]
    signatures: List[Dict[str, Any]]
    finished: bool

    def uids_of(self, track_id: int) -> List[Optional[str]]:
        return [rec[track_id][0] for _, rec in self.history if track_id in rec]

    def final_uid(self, track_id: int) -> Optional[str]:
        uids = [u for u in self.uids_of(track_id) if u]
        return uids[-1] if uids else None

    def payloads_of(self, uid: str) -> List[Dict[str, Any]]:
        return [p for p in self.signatures if p.get("person_uid") == uid]


def run_pipeline(db: ForensicDatabase, video: Path, scene: Scene, camera_id: str = "CAM_03") -> RunResult:
    """Run the real worker loop synchronously (signals are delivered directly in this thread)."""
    FakeDetector.scene = scene
    FakePose.scene = scene
    scene.cursor = -1
    frames, errors, events, sigs, finished = [0], [], [], [], [False]
    with mock.patch.object(worker_mod, "PedestrianDetector", FakeDetector), \
            mock.patch.object(worker_mod, "PoseBiometrics", FakePose), \
            mock.patch.object(worker_mod, "IdentityManager", RecordingIdentityManager):
        worker = worker_mod.ForensicPipelineWorker(db=db, source_type="video", source_path=str(video),
                                                   camera_id=camera_id, device="cpu")
        worker.frame_ready.connect(lambda f: frames.__setitem__(0, frames[0] + 1))
        worker.error_occurred.connect(errors.append)
        worker.identity_event.connect(events.append)
        worker.signature_detected.connect(sigs.append)
        worker.worker_finished.connect(lambda: finished.__setitem__(0, True))
        worker.run()
    history = worker.identity.history if worker.identity is not None else []
    return RunResult(worker, history, frames[0], errors, events, sigs, finished[0])


# =====================================================================================  timelines
SESSION1_FRAMES = 150
A_FIRST_TRACK, B_TRACK, A_RETURN_TRACK = 1, 2, 3


def session1_timeline(t: int):
    """A walks right and leaves (t < 50), ~20+ empty frames, re-enters from the right (t >= 70); B stays."""
    people = [(PERSON_B, B_TRACK, 95.0 + 12.0 * np.sin(t / 15.0), 205.0, 0.80)]
    if t < 50:
        people.append((PERSON_A, A_FIRST_TRACK, 200.0 + 10.0 * t, 190.0, 0.85))
    elif t >= 70:
        people.append((PERSON_A, A_RETURN_TRACK, max(340.0, 700.0 - 7.0 * (t - 70)), 200.0, 0.82))
    return people


def session2_timeline(t: int):
    """New session (app restart): only A, tracker ids start again at 1."""
    return [(PERSON_A, 1, 160.0 + 4.0 * t, 195.0, 0.84)]


# =====================================================================================  tests
class TestReentryPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="reentry_"))
        cls.video1 = cls.tmp / "session1.avi"
        cls.video2 = cls.tmp / "session2.avi"
        cls.db_path = cls.tmp / "forensic_test.db"
        cls.scene1 = build_video(cls.video1, session1_timeline, SESSION1_FRAMES, archive_at={60: [A_FIRST_TRACK]})
        cls.scene2 = build_video(cls.video2, session2_timeline, 60, seed=23)
        for video, scene in ((cls.video1, cls.scene1), (cls.video2, cls.scene2)):
            cap = cv2.VideoCapture(str(video))
            n = 0
            while cap.read()[0]:
                n += 1
            cap.release()
            assert n == len(scene.frames), f"decoded {n} frames, wrote {len(scene.frames)}"

        db = ForensicDatabase(db_path=str(cls.db_path))
        try:
            cls.run1 = run_pipeline(db, cls.video1, cls.scene1)
            cls.stats1 = db.get_stats()
            cls.crops_dir = db.crops_dir
        finally:
            db.close()

        # Application restart: new database connection on the same file, new worker
        db2 = ForensicDatabase(db_path=str(cls.db_path))
        try:
            cls.run2 = run_pipeline(db2, cls.video2, cls.scene2)
            cls.stats2 = db2.get_stats()
        finally:
            db2.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ session 1
    def test_stream_ran_cleanly(self):
        r = self.run1
        self.assertEqual(r.errors, [], f"worker emitted errors: {r.errors}")
        self.assertTrue(r.finished)
        self.assertEqual(r.frames, SESSION1_FRAMES)
        self.assertEqual(r.worker.error_counts, {}, f"per-stage errors: {r.worker.error_counts}")

    def test_reentering_person_gets_the_same_uid(self):
        r = self.run1
        uid_a = r.final_uid(A_FIRST_TRACK)
        uid_b = r.final_uid(B_TRACK)
        uid_a_return = r.final_uid(A_RETURN_TRACK)
        self.assertIsNotNone(uid_a, "A never got an identity before leaving")
        self.assertIsNotNone(uid_b, "B never got an identity")
        self.assertIsNotNone(uid_a_return, "returning A never got an identity")
        self.assertEqual(uid_a_return, uid_a, "re-entering person was given a different UID")
        self.assertNotEqual(uid_b, uid_a)
        # The returning track never showed any other uid (no provisional wrong identity)
        self.assertEqual({u for u in r.uids_of(A_RETURN_TRACK) if u}, {uid_a})
        # Exactly two people exist
        self.assertEqual(self.stats1["total_persons"], 2)
        self.assertEqual(r.worker.identity.session_identity_count, 2)

    def test_no_uid_on_first_frames(self):
        """Root cause #1: a track must not get a UID from its very first (noisy) frame."""
        r = self.run1
        for tid in (A_FIRST_TRACK, B_TRACK, A_RETURN_TRACK):
            first = next(rec[tid] for _, rec in r.history if tid in rec)
            self.assertIsNone(first[0])
            self.assertEqual(first[1], "tentative")
            self.assertEqual(first[2], TENTATIVE_LABEL)

    def test_no_two_tracks_share_a_uid_in_any_frame(self):
        for frame_idx, rec in self.run1.history:
            uids = [u for u, _, _ in rec.values() if u]
            self.assertEqual(len(uids), len(set(uids)), f"duplicate uid in frame {frame_idx}: {rec}")

    def test_identity_events(self):
        r = self.run1
        kinds = [e.get("type") for e in r.events]
        self.assertIn("new", kinds)
        self.assertIn("reacquired", kinds)
        uid_a = r.final_uid(A_FIRST_TRACK)
        reacq = [e for e in r.events if e.get("type") == "reacquired"]
        self.assertTrue(any(e["person_uid"] == uid_a and e["track_id"] == A_RETURN_TRACK for e in reacq))
        for e in r.events:
            self.assertEqual(e.get("camera_id"), "CAM_03")
            self.assertIn("timestamp", e)
        self.assertEqual(sum(1 for k in kinds if k == "new"), 2)

    def test_signature_payloads(self):
        r = self.run1
        self.assertGreater(len(r.signatures), 0)
        for p in r.signatures:
            self.assertTrue(p.get("person_uid"))
            for key in ("status", "display_label", "identity_color", "match_score", "upper_garment",
                        "lower_garment", "accessories", "track_id", "bbox", "camera_id"):
                self.assertIn(key, p)
            self.assertNotEqual(p["status"], "tentative")
            self.assertRegex(p["identity_color"], r"^#[0-9A-Fa-f]{6}$")
            self.assertEqual(p["display_label"], p["person_uid"])
        uid_a = r.final_uid(A_FIRST_TRACK)
        returning = [p for p in r.payloads_of(uid_a) if p.get("track_id") == A_RETURN_TRACK]
        self.assertIn("reacquired", {p["status"] for p in returning})
        # Re-acquired identities carry their gallery match score; freshly minted ones carry none
        for p in returning:
            self.assertIsInstance(p["match_score"], float)
            self.assertGreater(p["match_score"], 0.5)
        for p in r.signatures:
            if p.get("track_id") in (A_FIRST_TRACK, B_TRACK):
                self.assertIsNone(p["match_score"])

    def test_consensus_attributes(self):
        r = self.run1
        uid_a, uid_b = r.final_uid(A_FIRST_TRACK), r.final_uid(B_TRACK)
        last_a = r.payloads_of(uid_a)[-1]
        last_b = r.payloads_of(uid_b)[-1]
        self.assertEqual(color_family(last_a["upper_garment"]["color"]), "red", last_a["upper_garment"])
        self.assertEqual(color_family(last_b["upper_garment"]["color"]), "green", last_b["upper_garment"])
        self.assertEqual(color_family(last_a["lower_garment"]["color"]), "blue", last_a["lower_garment"])
        self.assertIn("backpack", last_a["accessories"])
        self.assertNotIn("backpack", last_b["accessories"])

    def test_best_crop_saved_next_to_database(self):
        r = self.run1
        for tid in (A_FIRST_TRACK, B_TRACK):
            uid = r.final_uid(tid)
            crop = Path(self.crops_dir) / f"{uid}.jpg"
            self.assertTrue(crop.exists(), f"missing thumbnail {crop}")
            img = cv2.imread(str(crop))
            self.assertIsNotNone(img)
            self.assertGreater(img.shape[0], img.shape[1])  # a person crop, not a frame
        with_crop = [p for p in r.signatures if p.get("crop_path")]
        self.assertTrue(with_crop)

    # ------------------------------------------------------------------ session 2 (restart)
    def test_identity_persists_across_sessions(self):
        r1, r2 = self.run1, self.run2
        self.assertEqual(r2.errors, [])
        uid_a = r1.final_uid(A_FIRST_TRACK)
        self.assertEqual(r2.final_uid(1), uid_a, "A got a different UID after an application restart")
        self.assertEqual({u for u in r2.uids_of(1) if u}, {uid_a})
        kinds = [e.get("type") for e in r2.events]
        self.assertIn("reacquired", kinds)
        self.assertNotIn("new", kinds)
        self.assertEqual(self.stats2["total_persons"], 2)


class TestDatabaseClearMidStream(unittest.TestCase):
    """Clearing the database while the worker runs resets identities and never resurrects wiped rows."""

    CLEAR_AT = 25

    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="reentry_clear_"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_clear_while_streaming(self):
        video = self.tmp / "clear.avi"
        scene = build_video(video, lambda t: [(PERSON_B, 7, 300.0 + 2.0 * t, 200.0, 0.82)], 60, seed=31)
        db = ForensicDatabase(db_path=str(self.tmp / "clear.db"))
        try:
            scene.hooks[self.CLEAR_AT] = db.clear_database   # what the UI's "Clear Database" does
            r = run_pipeline(db, video, scene)
            self.assertEqual(r.errors, [])
            self.assertEqual(r.worker.error_counts, {})
            before = [rec[7] for f, rec in r.history if f <= self.CLEAR_AT and 7 in rec]
            after = [rec[7] for f, rec in r.history if f > self.CLEAR_AT and 7 in rec]
            self.assertTrue(any(u for u, _, _ in before), "no identity before the clear")
            self.assertTrue(after and after[0][0] is None and after[0][1] == "tentative",
                            "identity state was not reset after the database clear")
            final_uid = r.final_uid(7)
            self.assertIsNotNone(final_uid)
            profiles = db.load_identity_profiles()
            self.assertEqual([p["person_uid"] for p in profiles], [final_uid])
            self.assertEqual(db.get_stats()["total_persons"], 1)
            self.assertTrue((Path(db.crops_dir) / f"{final_uid}.jpg").exists())
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
