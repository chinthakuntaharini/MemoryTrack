"""
Forensic Video Player Widget
Renders the annotated live video on a glowing glass canvas (LIVE badge, resolution/FPS chip) with playback controls.
"""

import logging
import os
import re
import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient
)
from PyQt6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from forensic_app.ui.effects import (
    ACCENT_AMBER, ACCENT_CYAN, ACCENT_RED, ACCENT_VIOLET, TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY, GlassPanel,
    LiveIndicator, set_role, set_variant, ui_font
)

logger = logging.getLogger(__name__)

CAMERA_CHANNELS = ("CAM_01 (Entrance)", "CAM_02 (Corridor)", "CAM_03 (Perimeter)", "CAM_04 (Parking)")


def numpy_to_qimage(frame: np.ndarray) -> Optional[QImage]:
    """Convert a BGR / BGRA / grayscale numpy frame into an independent QImage (deep copy, no dangling buffer)."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
        return None
    arr = np.ascontiguousarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        fmt = QImage.Format.Format_Grayscale8
    elif arr.shape[2] == 3:
        fmt = QImage.Format.Format_BGR888
    elif arr.shape[2] == 4:
        arr = np.ascontiguousarray(arr[:, :, [2, 1, 0, 3]])
        fmt = QImage.Format.Format_RGBA8888
    else:
        return None
    # `arr` stays referenced until the deep copy returns, so the QImage never points at freed memory.
    view = QImage(arr.data, w, h, int(arr.strides[0]), fmt)
    if fmt == QImage.Format.Format_BGR888:
        # RGB32 copy: ~3x faster to smooth-scale than BGR888 (matters at 150-200% display scaling)
        return view.convertToFormat(QImage.Format.Format_RGB32)
    return view.copy()


class VideoCanvas(QWidget):
    """Glass video surface: rounded frame, paint-based glowing border (pulses while LIVE), HUD chips, idle placeholder."""

    RADIUS = 16.0
    MARGIN = 9            # room for the outer glow
    PULSE_PERIOD_S = 1.8

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(560, 340)
        self._image: Optional[QImage] = None
        self._scaled: Optional[QPixmap] = None
        self._scaled_dirty = False
        self._state = "idle"                  # idle | live | paused
        self._camera_label = "CAM_01"
        self._source_label = ""
        self._message = ""
        self._fps = 0.0
        self._last_frame_t = 0.0
        self._last_paint_t = 0.0
        self.frames_received = 0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(50)
        self._pulse_timer.timeout.connect(self._on_pulse)
        self._font_badge = ui_font(11, QFont.Weight.Bold)
        self._font_chip = ui_font(11, QFont.Weight.DemiBold)
        self._font_title = ui_font(17, QFont.Weight.DemiBold)
        self._font_body = ui_font(13)

    # ---- state -------------------------------------------------------------------------------
    def has_frame(self) -> bool:
        return self._image is not None

    def frame_size(self) -> QSize:
        return self._image.size() if self._image is not None else QSize(0, 0)

    def fps(self) -> float:
        return self._fps

    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        """idle / live / paused; the border pulses only while live."""
        self._state = state if state in ("idle", "live", "paused") else "idle"
        if self._state == "live":
            self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
        self.update()

    def set_camera_label(self, text: str) -> None:
        self._camera_label = text
        self.update()

    def set_source_label(self, text: str) -> None:
        self._source_label = text
        self.update()

    def set_fps(self, fps: float) -> None:
        """Optional external FPS (pipeline throughput) shown in the chip."""
        self._fps = max(0.0, float(fps))

    def clear_frame(self, message: str = "") -> None:
        """Drop the last frame and show the idle placeholder (optionally with a message)."""
        self._image = None
        self._scaled = None
        self._scaled_dirty = False
        self._message = message
        self._fps = 0.0
        self.update()

    def set_frame(self, frame: np.ndarray) -> bool:
        """Store a copy of the frame; it is scaled lazily on the next paint (no cost while hidden)."""
        img = numpy_to_qimage(frame)
        if img is None:
            return False
        now = time.monotonic()
        if self._last_frame_t > 0:
            dt = now - self._last_frame_t
            if dt > 1e-4:
                inst = 1.0 / dt
                self._fps = inst if self._fps <= 0 else 0.85 * self._fps + 0.15 * inst
        self._last_frame_t = now
        self._image = img
        self._message = ""
        self.frames_received += 1
        self._scaled_dirty = True
        self.update()
        return True

    # ---- geometry ----------------------------------------------------------------------------
    def _content_rect(self) -> QRectF:
        m = self.MARGIN
        return QRectF(self.rect()).adjusted(m, m, -m, -m)

    def _image_rect(self) -> QRectF:
        area = self._content_rect()
        if self._image is None or self._image.width() <= 0 or self._image.height() <= 0:
            return area
        iw, ih = self._image.width(), self._image.height()
        scale = min(area.width() / iw, area.height() / ih)
        w, h = iw * scale, ih * scale
        return QRectF(area.center().x() - w / 2, area.center().y() - h / 2, w, h)

    def _rescale(self) -> None:
        self._scaled_dirty = False
        if self._image is None:
            return
        rect = self._image_rect()
        dpr = max(1.0, float(self.devicePixelRatioF()))
        dw, dh = max(1, int(round(rect.width() * dpr))), max(1, int(round(rect.height() * dpr)))
        scaled = self._image.scaled(dw, dh, Qt.AspectRatioMode.IgnoreAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        rounded = QImage(dw, dh, QImage.Format.Format_ARGB32_Premultiplied)
        rounded.fill(Qt.GlobalColor.transparent)
        p = QPainter(rounded)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(scaled))
        r = self.RADIUS * dpr
        p.drawRoundedRect(QRectF(0, 0, dw, dh), r, r)
        p.end()
        pix = QPixmap.fromImage(rounded)
        pix.setDevicePixelRatio(dpr)
        self._scaled = pix

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._scaled_dirty = self._image is not None

    def _on_pulse(self) -> None:
        # Frames already trigger repaints; only tick when the stream is idle/stalled
        if time.monotonic() - self._last_paint_t > 0.045:
            self.update()

    def _pulse_value(self) -> float:
        phase = (time.monotonic() % self.PULSE_PERIOD_S) / self.PULSE_PERIOD_S
        return 0.5 - 0.5 * np.cos(2 * np.pi * phase)

    # ---- painting ----------------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        self._last_paint_t = time.monotonic()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        area = self._content_rect()
        frame_rect = self._image_rect()
        if self._scaled_dirty:
            self._rescale()

        # Letterbox backdrop
        back = QPainterPath()
        back.addRoundedRect(area, self.RADIUS, self.RADIUS)
        bg = QLinearGradient(area.topLeft(), area.bottomRight())
        bg.setColorAt(0.0, QColor(6, 10, 26, 215))
        bg.setColorAt(1.0, QColor(3, 6, 18, 225))
        p.fillPath(back, bg)

        if self._scaled is not None and self._image is not None:
            p.drawPixmap(frame_rect.topLeft(), self._scaled)
            self._paint_border(p, frame_rect)
            self._paint_badges(p, frame_rect)
        else:
            self._paint_placeholder(p, area)
            self._paint_border(p, area)
        p.end()

    def _paint_border(self, p: QPainter, rect: QRectF) -> None:
        if self._state == "live":
            base = QColor(ACCENT_CYAN)
            k = 0.45 + 0.55 * self._pulse_value()
            layers = 6
        elif self._state == "paused":
            base = QColor(ACCENT_AMBER)
            k = 0.55
            layers = 4
        else:
            base = QColor(255, 255, 255)
            k = 0.0
            layers = 0
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(layers, 0, -1):          # outer soft glow
            c = QColor(base)
            c.setAlphaF(max(0.0, min(1.0, 0.16 * k * (1.0 - i / (layers + 1)) ** 1.6)))
            p.setPen(QPen(c, 2.0))
            grow = i * 1.3
            r = rect.adjusted(-grow, -grow, grow, grow)
            p.drawRoundedRect(r, self.RADIUS + grow, self.RADIUS + grow)
        if layers:
            grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
            c0 = QColor(base)
            c0.setAlphaF(0.55 + 0.4 * k)
            c1 = QColor(ACCENT_VIOLET if self._state == "live" else base.name())
            c1.setAlphaF(0.35 + 0.35 * k)
            grad.setColorAt(0.0, c0)
            grad.setColorAt(1.0, c1)
            p.setPen(QPen(QBrush(grad), 1.4))
        else:
            p.setPen(QPen(QColor(255, 255, 255, 34), 1.0))
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), self.RADIUS, self.RADIUS)

    def _pill(self, p: QPainter, x: float, y: float, text: str, font: QFont, fg: QColor,
              dot: Optional[QColor] = None, align_right: bool = False) -> QRectF:
        p.setFont(font)
        fm = p.fontMetrics()
        pad = 10.0
        dot_w = 14.0 if dot is not None else 0.0
        w = fm.horizontalAdvance(text) + pad * 2 + dot_w
        h = fm.height() + 10.0
        if align_right:
            x -= w
        rect = QRectF(x, y, w, h)
        p.setPen(QPen(QColor(255, 255, 255, 40), 1.0))
        p.setBrush(QColor(6, 10, 24, 190))
        p.drawRoundedRect(rect, h / 2, h / 2)
        tx = rect.left() + pad
        if dot is not None:
            c = rect.left() + pad + 4.0
            center = QPointF(c, rect.center().y())
            if self._state == "live":
                k = self._pulse_value()
                halo = QRadialGradient(center, 4 + 6 * k)
                h0 = QColor(dot)
                h0.setAlphaF(0.6 * (1 - 0.4 * k))
                h1 = QColor(dot)
                h1.setAlphaF(0.0)
                halo.setColorAt(0, h0)
                halo.setColorAt(1, h1)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(halo))
                p.drawEllipse(center, 4 + 6 * k, 4 + 6 * k)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(dot)
            p.drawEllipse(center, 3.6, 3.6)
            tx += dot_w
        p.setPen(fg)
        p.drawText(QRectF(tx, rect.top(), rect.width(), rect.height()),
                   int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), text)
        return rect

    def _paint_badges(self, p: QPainter, rect: QRectF) -> None:
        inset = 12.0
        if self._state == "live":
            label, dot = "LIVE", QColor(ACCENT_RED)
        elif self._state == "paused":
            label, dot = "PAUSED", QColor(ACCENT_AMBER)
        else:
            label, dot = "STOPPED", QColor(TEXT_MUTED)
        badge = self._pill(p, rect.left() + inset, rect.top() + inset, label, self._font_badge,
                           QColor(TEXT_PRIMARY), dot)
        self._pill(p, badge.right() + 6, rect.top() + inset, self._camera_label, self._font_chip,
                   QColor(TEXT_SECONDARY))
        size = self.frame_size()
        chip = f"{size.width()} x {size.height()}"
        if self._fps > 0:
            chip += f"   {self._fps:.1f} FPS"
        self._pill(p, rect.right() - inset, rect.top() + inset, chip, self._font_chip, QColor(TEXT_SECONDARY),
                   align_right=True)

    def _paint_placeholder(self, p: QPainter, area: QRectF) -> None:
        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(area, self.RADIUS, self.RADIUS)
        p.setClipPath(clip)
        # Faint scanning grid
        p.setPen(QPen(QColor(125, 211, 252, 14), 1.0))
        step = 32.0
        x = area.left() + step
        while x < area.right():
            p.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            x += step
        y = area.top() + step
        while y < area.bottom():
            p.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            y += step
        # Soft centre glow
        c = area.center()
        glow = QRadialGradient(c, min(area.width(), area.height()) * 0.55)
        glow.setColorAt(0.0, QColor(56, 189, 248, 38))
        glow.setColorAt(0.6, QColor(167, 139, 250, 14))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(glow))
        p.drawRect(area)
        p.restore()

        # Reticle icon
        p.save()
        icon_c = QPointF(c.x(), c.y() - 38)
        ring = 30.0
        grad = QLinearGradient(icon_c - QPointF(ring, ring), icon_c + QPointF(ring, ring))
        grad.setColorAt(0.0, QColor(ACCENT_CYAN))
        grad.setColorAt(1.0, QColor(ACCENT_VIOLET))
        pen = QPen(QBrush(grad), 2.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(icon_c, ring, ring)
        p.drawEllipse(icon_c, ring * 0.42, ring * 0.42)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            p.drawLine(QPointF(icon_c.x() + dx * ring * 0.7, icon_c.y() + dy * ring * 0.7),
                       QPointF(icon_c.x() + dx * ring * 1.35, icon_c.y() + dy * ring * 1.35))
        p.restore()

        title = "No active feed" if not self._message else self._message
        p.setPen(QColor(TEXT_PRIMARY))
        p.setFont(self._font_title)
        p.drawText(QRectF(area.left(), c.y() + 10, area.width(), 28), int(Qt.AlignmentFlag.AlignCenter), title)
        p.setPen(QColor(TEXT_SECONDARY))
        p.setFont(self._font_body)
        p.drawText(QRectF(area.left() + 24, c.y() + 40, area.width() - 48, 22), int(Qt.AlignmentFlag.AlignCenter),
                   "Load a surveillance video or start the webcam to begin identity tracking.")
        p.setPen(QColor(TEXT_MUTED))
        p.drawText(QRectF(area.left() + 24, c.y() + 64, area.width() - 48, 22), int(Qt.AlignmentFlag.AlignCenter),
                   "Identities persist across re-entries: appearance + clothing colours + accessories.")


class PlayerWidget(QWidget):
    """Forensic surveillance stream display canvas with controls."""

    # UI Events
    play_requested = pyqtSignal(str, str)  # source_type ('video'/'webcam'), source_path
    pause_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_playing = False
        self.is_paused = False
        self.current_video_path: Optional[str] = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Video panel (title row + canvas)
        self.video_panel = GlassPanel(radius=18)
        vp = QVBoxLayout(self.video_panel)
        vp.setContentsMargins(14, 12, 14, 10)
        vp.setSpacing(6)
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self.lbl_feed_title = set_role(QLabel("LIVE SURVEILLANCE FEED"), "section")
        self.lbl_feed_source = set_role(QLabel("No source loaded"), "muted")
        title_row.addWidget(self.lbl_feed_title)
        title_row.addStretch()
        title_row.addWidget(self.lbl_feed_source)
        vp.addLayout(title_row)

        self.canvas = VideoCanvas()
        vp.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.video_panel, stretch=1)

        # Control bar
        self.control_bar = GlassPanel(radius=14)
        ctrl_layout = QHBoxLayout(self.control_bar)
        ctrl_layout.setContentsMargins(12, 10, 12, 10)
        ctrl_layout.setSpacing(8)

        self.btn_load_video = set_variant(QPushButton("Load Video File"), "primary")
        self.btn_load_video.setToolTip("Open a recorded surveillance video")
        self.btn_load_video.clicked.connect(self._on_load_video)

        self.btn_webcam = set_variant(QPushButton("Start Webcam"), "ghost")
        self.btn_webcam.setToolTip("Stream from the default camera device")
        self.btn_webcam.clicked.connect(self._on_start_webcam)

        self.btn_pause = set_variant(QPushButton("Pause"), "ghost")
        self.btn_pause.clicked.connect(self._on_pause_toggle)
        self.btn_pause.setEnabled(False)

        self.btn_stop = set_variant(QPushButton("Stop"), "danger")
        self.btn_stop.setObjectName("btn_danger")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)

        # Camera selection dropdown
        self.combo_cam = QComboBox()
        self.combo_cam.addItems(list(CAMERA_CHANNELS))
        self.combo_cam.setMinimumWidth(176)
        self.combo_cam.setToolTip("Camera node recorded with every sighting")
        self.combo_cam.currentIndexChanged.connect(self._on_camera_changed)

        self.live_indicator = LiveIndicator()

        ctrl_layout.addWidget(self.btn_load_video)
        ctrl_layout.addWidget(self.btn_webcam)
        ctrl_layout.addWidget(self.btn_pause)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addSpacing(14)
        ctrl_layout.addWidget(set_role(QLabel("Channel"), "field"))
        ctrl_layout.addWidget(self.combo_cam)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.live_indicator)

        layout.addWidget(self.control_bar)
        self._on_camera_changed()

    # ---- public API ----------------------------------------------------------------------------
    def current_camera_id(self) -> str:
        """Camera id parsed from the channel combo, e.g. 'CAM_01 (Entrance)' -> 'CAM_01'."""
        text = self.combo_cam.currentText().strip()
        m = re.search(r"CAM[_\-]?\d+", text, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper().replace("-", "_")
        return text.split(" ")[0] if text else "CAM_01"

    def update_frame(self, cv_frame: np.ndarray) -> None:
        """Render processed OpenCV BGR frame onto the glass canvas (the frame buffer is copied)."""
        if cv_frame is None or not isinstance(cv_frame, np.ndarray) or cv_frame.size == 0:
            return
        if self.canvas.set_frame(cv_frame) and self.canvas.state() == "idle" and self.is_playing:
            self.canvas.set_state("paused" if self.is_paused else "live")

    def set_pipeline_fps(self, fps: float) -> None:
        """Show pipeline throughput (from the worker's stats) in the canvas chip."""
        self.canvas.set_fps(fps)

    def reset_ui_state(self) -> None:
        """Reset buttons and canvas when the stream ends (does not emit stop_requested)."""
        self._set_idle_controls()
        self.canvas.set_state("idle")
        self.canvas.clear_frame("Stream ended")
        self.live_indicator.set_state("idle")

    # ---- internals -----------------------------------------------------------------------------
    def _on_camera_changed(self, *_args) -> None:
        self.canvas.set_camera_label(self.combo_cam.currentText())

    def _set_streaming_controls(self) -> None:
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(True)
        self.btn_load_video.setEnabled(False)
        self.btn_webcam.setEnabled(False)
        self.combo_cam.setEnabled(False)
        self.is_playing = True
        self.is_paused = False

    def _set_idle_controls(self) -> None:
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(False)
        self.btn_load_video.setEnabled(True)
        self.btn_webcam.setEnabled(True)
        self.combo_cam.setEnabled(True)
        self.is_playing = False
        self.is_paused = False

    def _begin_stream(self, source_type: str, source_path: str, label: str) -> None:
        self._set_streaming_controls()
        self.lbl_feed_source.setText(label)
        self.canvas.set_source_label(label)
        self.canvas.clear_frame("Connecting to stream...")
        self.canvas.set_state("live")
        self.live_indicator.set_state("live", self.current_camera_id())
        self.play_requested.emit(source_type, source_path)

    def _on_load_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Surveillance Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if file_path:
            self.current_video_path = file_path
            self._begin_stream("video", file_path, os.path.basename(file_path))

    def _on_start_webcam(self) -> None:
        self._begin_stream("webcam", "0", "Webcam 0")

    def _on_pause_toggle(self) -> None:
        if self.is_paused:
            self.btn_pause.setText("Pause")
            self.is_paused = False
            self.canvas.set_state("live")
            self.live_indicator.set_state("live", self.current_camera_id())
            self.resume_requested.emit()
        else:
            self.btn_pause.setText("Resume")
            self.is_paused = True
            self.canvas.set_state("paused")
            self.live_indicator.set_state("paused")
            self.pause_requested.emit()

    def _on_stop(self) -> None:
        self._set_idle_controls()
        self.canvas.set_state("idle")
        self.canvas.clear_frame("Pipeline stopped")
        self.live_indicator.set_state("idle")
        self.stop_requested.emit()
