"""
Headless UI Effects, Live Identity Panel & HUD Drawing Tests
Verifies hover/pulse glows, glass panels, the live identity panel lifecycle, the event feed cap,
identity colours, ROI-only HUD blending, and the video player canvas (run with QT_QPA_PLATFORM=offscreen).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:  # Keep the torch-before-Qt import order used by the app on Windows
    import torch  # noqa: F401
except Exception:
    pass

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QEventLoop, QPointF, Qt, QTimer
from PyQt6.QtGui import QEnterEvent, QFocusEvent
from PyQt6.QtWidgets import QApplication, QGraphicsOpacityEffect, QLineEdit, QPushButton, QVBoxLayout, QWidget

from forensic_app.ui import effects
from forensic_app.ui.effects import AuroraBackground, GlassPanel, apply_hover_glow, apply_pulse_glow
from forensic_app.ui import live_panel as live_mod
from forensic_app.ui.live_panel import LiveIdentityPanel, format_identity_event
from forensic_app.ui.player_widget import PlayerWidget, numpy_to_qimage
from forensic_app.ui.trajectory_widget import compress_camera_hops
from forensic_app.utils.drawing import (
    draw_forensic_bbox, draw_skeleton, identity_color_bgr, identity_color_hex, identity_color_rgb
)


def _pump(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _enter(widget: QWidget) -> None:
    p = QPointF(3, 3)
    QApplication.sendEvent(widget, QEnterEvent(p, p, p))


def _leave(widget: QWidget) -> None:
    QApplication.sendEvent(widget, QEvent(QEvent.Type.Leave))


class FakeClock:
    """Controllable monotonic clock for the live panel."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


def _signature(uid: str, **extra):
    payload = {
        "person_uid": uid, "camera_id": "CAM_01", "timestamp": "2026-10-26T12:00:00Z",
        "upper_garment": {"color": "Red", "hex": "#DC2626"},
        "lower_garment": {"color": "Denim Blue", "hex": "#3B5B8C", "type": "Pants/Jeans"},
        "accessories": ["backpack", "cap"], "posture": "Walking", "biometrics": {}, "embedding": [0.0] * 512,
        "track_id": 4, "bbox": [10, 10, 60, 160], "crop_path": None, "confidence": 0.9,
        "status": "confirmed", "match_score": 0.88, "display_label": uid, "identity_color": identity_color_hex(uid),
    }
    payload.update(extra)
    return payload


class TestUIEffects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    # ---- glows ----------------------------------------------------------------------------------
    def test_hover_glow_animates_on_enter_leave(self):
        host = QWidget()
        lay = QVBoxLayout(host)
        btn = QPushButton("Search")
        btn.setProperty("variant", "primary")
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)                       # isolate hover from focus glow
        lay.addWidget(btn)
        host.show()
        glow = apply_hover_glow(btn)
        self.assertIsNotNone(glow)
        self.assertIs(btn.graphicsEffect(), glow.effect)
        self.assertFalse(glow.effect.isEnabled())          # no cost at rest

        _enter(btn)
        _pump(320)
        self.assertGreater(glow.value, 0.9)
        self.assertTrue(glow.effect.isEnabled())
        self.assertGreater(glow.effect.blurRadius(), 10.0)

        _leave(btn)
        _pump(320)
        self.assertLess(glow.value, 0.05)
        self.assertFalse(glow.effect.isEnabled())
        # Applying twice reuses the same controller (max one effect per widget)
        self.assertIs(apply_hover_glow(btn), glow)

        # Focus glow (keyboard navigation) settles at a softer intensity
        edit = QLineEdit()
        lay.addWidget(edit)
        edit_glow = apply_hover_glow(edit)
        QApplication.sendEvent(edit, QFocusEvent(QEvent.Type.FocusIn))
        edit.setFocus()
        _pump(300)
        if edit.hasFocus():
            self.assertGreater(edit_glow.value, 0.4)
        host.close()

    def test_hover_glow_skips_disabled_and_existing_effects(self):
        btn = QPushButton("Disabled")
        glow = apply_hover_glow(btn)
        btn.setEnabled(False)
        _enter(btn)
        _pump(250)
        self.assertLess(glow.value, 0.01)

        other = QPushButton("Has effect")
        other.setGraphicsEffect(QGraphicsOpacityEffect(other))
        self.assertIsNone(apply_hover_glow(other))
        self.assertIsNone(apply_pulse_glow(other))

    def test_pulse_glow_runs_and_stops(self):
        w = QWidget()
        pulse = apply_pulse_glow(w, "#f43f5e", period_ms=200)
        self.assertIsNotNone(pulse)
        pulse.start()
        self.assertTrue(pulse.is_running())
        _pump(120)
        pulse.stop()
        self.assertFalse(pulse.is_running())
        self.assertFalse(pulse.effect.isEnabled())

    # ---- glass ----------------------------------------------------------------------------------
    def test_glass_panel_and_aurora_grab(self):
        aurora = AuroraBackground()
        lay = QVBoxLayout(aurora)
        lay.setContentsMargins(40, 40, 40, 40)
        panel = GlassPanel(radius=16)
        lay.addWidget(panel)
        aurora.resize(420, 300)
        aurora.show()
        _pump(50)
        shot = aurora.grab().toImage()
        self.assertFalse(shot.isNull())
        self.assertEqual(aurora.render_count, 1)
        bg = shot.pixelColor(10, 10)
        inside = shot.pixelColor(210, 150)
        self.assertLess(bg.lightness(), 140)                     # deep navy backdrop
        self.assertNotEqual((bg.red(), bg.green(), bg.blue()), (0, 0, 0))
        self.assertGreater(inside.lightness(), shot.pixelColor(20, 150).lightness() - 30)
        self.assertFalse(panel.grab().isNull())
        aurora.resize(430, 305)                                  # same size bucket: no re-render
        aurora.grab()
        self.assertEqual(aurora.render_count, 1)
        aurora.set_backdrop_mode(True)
        self.assertTrue(aurora.is_backdrop())
        self.assertFalse(aurora.grab().isNull())
        aurora.close()

    def test_backdrop_is_noop_offscreen(self):
        w = QWidget()
        if QApplication.platformName() != "windows":
            self.assertFalse(effects.prepare_window_for_backdrop(w))
            self.assertFalse(effects.enable_window_backdrop(w))
        old = os.environ.get("FORENSIC_GLASS")
        os.environ["FORENSIC_GLASS"] = "0"                                # explicit opt-out always wins
        try:
            self.assertFalse(effects.backdrop_supported())
        finally:
            if old is None:
                os.environ.pop("FORENSIC_GLASS", None)
            else:
                os.environ["FORENSIC_GLASS"] = old
        aurora_rgb, frosted = effects.render_aurora_arrays(200, 120)
        self.assertEqual(aurora_rgb.shape, (120, 200, 3))
        self.assertEqual(frosted.dtype, np.uint8)

    # ---- live identity panel ----------------------------------------------------------------------
    def test_live_panel_add_update_stale_remove(self):
        clock = FakeClock()
        panel = LiveIdentityPanel(clock=clock, auto_tick=False)
        panel.on_signature(_signature("UID-0001", status="new"))
        panel.on_signature(_signature("UID-0002"))
        panel.on_signature({"person_uid": "", "status": "confirmed"})          # ignored
        panel.on_signature(_signature("UID-0009", status="tentative"))         # ignored
        panel.tick()
        self.assertIsNotNone(panel.card_for("UID-0001"))
        self.assertIsNotNone(panel.card_for("UID-0002"))
        self.assertIsNone(panel.card_for("UID-0009"))
        self.assertEqual(panel.in_view_count(), 2)
        self.assertEqual(panel.entry_state("UID-0001")["badge"], "new")
        view = panel.card_for("UID-0001").view
        self.assertEqual(view["badge_text"], "NEW")
        self.assertEqual(view["color"], identity_color_hex("UID-0001"))
        self.assertIn("Backpack", view["accessories"])
        self.assertEqual(view["upper_name"], "Red")

        # Badge expires back to TRACKING; UID-0001 keeps updating, UID-0002 goes quiet
        clock.now += live_mod.BADGE_HOLD_S + 0.5
        panel.on_signature(_signature("UID-0001"))
        panel.tick()
        self.assertEqual(panel.card_for("UID-0001").view["badge_text"], "TRACKING")
        self.assertTrue(panel.entry_state("UID-0002")["stale"])
        self.assertEqual(panel.card_for("UID-0002").view["badge_text"], "RECENTLY SEEN")
        self.assertEqual(panel.in_view_count(), 1)
        self.assertEqual(panel.visible_uids()[0], "UID-0001")              # active cards first

        # Re-entry: the same uid comes back -> same card, flagged as re-identified
        panel.on_signature(_signature("UID-0002", status="reacquired"))
        panel.tick()
        self.assertFalse(panel.entry_state("UID-0002")["stale"])
        self.assertEqual(panel.card_for("UID-0002").view["badge_text"], "RE-IDENTIFIED")

        clock.now += live_mod.REMOVE_AFTER_S + 1.0
        panel.tick()
        self.assertIsNone(panel.card_for("UID-0001"))
        self.assertIsNone(panel.card_for("UID-0002"))
        self.assertEqual(panel.in_view_count(), 0)
        self.assertEqual(panel.session_count(), 2)
        panel.set_session_identities(7)
        self.assertEqual(panel.session_count(), 7)
        panel.clear()
        self.assertEqual(panel.session_count(), 0)

    def test_live_panel_thumbnail_and_merge(self):
        clock = FakeClock()
        panel = LiveIdentityPanel(clock=clock, auto_tick=False)
        tmp = tempfile.mkdtemp()
        crop = os.path.join(tmp, "crop.jpg")
        cv2.imwrite(crop, np.full((120, 60, 3), (40, 40, 200), np.uint8))
        panel.on_signature(_signature("UID-0005", crop_path=crop))
        panel.on_signature(_signature("UID-0006"))
        panel.tick()
        thumb = panel.card_for("UID-0005").view["thumb"]
        self.assertIsNotNone(thumb)
        self.assertFalse(thumb.isNull())
        panel.card_for("UID-0005").grab()                                   # paints without errors

        panel.on_identity_event({"type": "merged", "person_uid": "UID-0005", "merged_from": "UID-0006",
                                 "track_id": 4, "confidence": 0.8, "camera_id": "CAM_01", "timestamp": 1.0})
        panel.tick()
        self.assertIsNone(panel.card_for("UID-0006"))
        self.assertEqual(panel.entry_state("UID-0005")["badge"], "reacquired")

    def test_event_feed_cap_and_order(self):
        panel = LiveIdentityPanel(clock=FakeClock(), auto_tick=False)
        cap = panel.feed.cap()
        for i in range(cap + 50):
            panel.on_identity_event({"type": "reacquired" if i % 2 else "new", "person_uid": f"UID-{i:04d}",
                                     "track_id": i, "confidence": 0.9, "camera_id": "CAM_02",
                                     "timestamp": "2026-10-26T12:00:00Z"})
        self.assertEqual(panel.feed.count(), cap)
        entries = panel.feed.entries()
        self.assertEqual(entries[0]["uid"], f"UID-{cap + 49:04d}")         # most recent first
        self.assertEqual(entries[0]["time"], "12:00:00")
        panel.feed.grab()

    def test_format_identity_event(self):
        e = format_identity_event({"type": "reacquired", "person_uid": "UID-0007", "confidence": 0.91,
                                   "camera_id": "CAM_03", "timestamp": "2026-10-26T08:15:02Z"})
        self.assertIn("91%", e["text"])
        self.assertEqual(e["camera"], "CAM_03")
        self.assertEqual(e["time"], "08:15:02")
        m = format_identity_event({"type": "merged", "person_uid": "UID-0002", "merged_from": "UID-0010"})
        self.assertIn("UID-0010", m["text"])

    # ---- drawing ----------------------------------------------------------------------------------
    def test_identity_colors_deterministic_and_distinct(self):
        self.assertEqual(identity_color_hex("UID-0007"), identity_color_hex("UID-0007"))
        hexes = [identity_color_hex(f"UID-{i:04d}") for i in range(1, 13)]
        self.assertEqual(len(set(hexes)), 12)
        for h in hexes:
            self.assertRegex(h, r"^#[0-9A-F]{6}$")
        rgbs = [np.array(identity_color_rgb(f"UID-{i:04d}"), dtype=float) for i in range(1, 9)]
        for i in range(len(rgbs) - 1):                              # consecutive uids well separated
            self.assertGreater(np.linalg.norm(rgbs[i] - rgbs[i + 1]), 60.0)
        for rgb in rgbs:                                             # vivid, readable on dark backgrounds
            self.assertGreater(rgb.max(), 170)
        b, g, r = identity_color_bgr("UID-0003")
        self.assertEqual(identity_color_hex("UID-0003"), f"#{r:02X}{g:02X}{b:02X}")
        self.assertEqual(identity_color_hex("UID-CAM_01-0036"), identity_color_hex("UID-0036"))
        self.assertRegex(identity_color_hex("provisional-abc"), r"^#[0-9A-F]{6}$")

    def test_draw_forensic_bbox_roi_only(self):
        frame = np.full((720, 1280, 3), 90, np.uint8)
        before = frame.copy()
        draw_forensic_bbox(frame, (500, 200, 700, 600), "UID-0004", 12, "Red", "#DC2626", "Denim Blue", "#3B5B8C",
                           accessories=["backpack", "cap"], status="reacquired", match_score=0.93)
        changed = np.any(frame != before, axis=2)
        self.assertTrue(changed[200, 520:680].any())                     # box edge drawn
        self.assertFalse(changed[400, 560:640].any())                    # box interior untouched
        self.assertFalse(changed[:, :400].any())                         # far left untouched
        self.assertFalse(changed[:, 900:].any())                         # far right untouched
        self.assertFalse(changed[690:, :].any())                         # far below untouched
        # Glow uses the identity colour
        edge = frame[200, 600].astype(int)
        b, g, r = identity_color_bgr("UID-0004")
        self.assertEqual(int(np.argmax(edge)), int(np.argmax([b, g, r])))

    def test_draw_tentative_and_edges(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        draw_forensic_bbox(frame, (100, 100, 220, 380), None, 3, status="tentative")
        self.assertTrue(frame[100, 100:220].any())
        border = frame[100, 100:220].reshape(-1, 3).astype(int)
        lit = border[border.sum(axis=1) > 0]
        self.assertLess(np.mean(np.abs(lit[:, 0] - lit[:, 2])), 60)      # muted grey, not identity coloured
        draw_forensic_bbox(frame, (620, 460, 640, 480), "UID-EDGE-001", 1, accessories=["backpack"])
        draw_forensic_bbox(frame, (-50, -40, 30, 20), "UID-0001", 2, display_label="UID-0001")
        draw_forensic_bbox(frame, (10, 10, 10, 10), "UID-0001", 2)       # degenerate box: no-op
        draw_skeleton(frame, np.full((17, 3), 100.0, np.float32), color="#38bdf8")
        self.assertEqual(frame.shape, (480, 640, 3))

    # ---- player ---------------------------------------------------------------------------------
    def test_player_update_frame_and_camera_id(self):
        player = PlayerWidget()
        player.resize(900, 640)
        player.show()
        self.assertEqual(player.current_camera_id(), "CAM_01")
        player.combo_cam.setCurrentIndex(2)
        self.assertEqual(player.current_camera_id(), "CAM_03")

        frame = np.zeros((360, 640, 3), np.uint8)
        frame[:, :, 2] = 255                                              # pure red (BGR)
        player.update_frame(frame)
        del frame                                                         # buffer must not be referenced
        _pump(20)
        self.assertTrue(player.canvas.has_frame())
        self.assertEqual(player.canvas.frame_size().width(), 640)
        img = player.canvas.grab().toImage()
        c = img.pixelColor(img.width() // 2, img.height() // 2)
        self.assertGreater(c.red(), 200)
        self.assertLess(c.blue(), 60)
        player.update_frame(None)
        player.update_frame(np.zeros((0, 0, 3), np.uint8))
        q = numpy_to_qimage(np.zeros((10, 20), np.uint8))
        self.assertEqual((q.width(), q.height()), (20, 10))

        requested, stopped = [], []
        player.play_requested.connect(lambda t, p: requested.append((t, p)))
        player.stop_requested.connect(lambda: stopped.append(True))
        player._on_start_webcam()
        self.assertEqual(requested, [("webcam", "0")])
        self.assertEqual(player.canvas.state(), "live")
        self.assertFalse(player.combo_cam.isEnabled())
        player.reset_ui_state()
        self.assertEqual(stopped, [])                                     # end-of-stream reset does not re-emit stop
        self.assertFalse(player.canvas.has_frame())
        self.assertTrue(player.btn_load_video.isEnabled())
        player.close()

    def test_camera_hop_compression(self):
        rows = [{"camera_id": "CAM_01", "timestamp": "t1"}, {"camera_id": "CAM_01", "timestamp": "t2"},
                {"camera_id": "CAM_02", "timestamp": "t3"}, {"camera_id": "CAM_01", "timestamp": "t4"}]
        segs = compress_camera_hops(rows)
        self.assertEqual([s["camera"] for s in segs], ["CAM_01", "CAM_02", "CAM_01"])
        self.assertEqual(segs[0]["count"], 2)
        self.assertEqual(segs[0]["end"], "t2")


if __name__ == "__main__":
    unittest.main()
