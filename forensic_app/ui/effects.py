"""
Glassmorphism, Blur & Glow Effects
Windows acrylic backdrop, aurora background, frosted glass panels, and animated hover / pulse glow helpers.
"""

import ctypes
import logging
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import (
    QEasingCurve, QEvent, QObject, QPoint, QPointF, QRectF, QSize, Qt, QVariantAnimation, pyqtSignal
)
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QFontDatabase, QFontMetrics, QGuiApplication, QIcon, QImage, QLinearGradient, QPainter,
    QPainterPath, QPalette, QPen, QPixmap, QRadialGradient, QTransform
)
from PyQt6.QtWidgets import (
    QAbstractScrollArea, QAbstractSpinBox, QApplication, QComboBox, QFrame, QGraphicsDropShadowEffect,
    QLabel, QLineEdit, QPushButton, QWidget
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------
# Theme tokens (keep in sync with styles.qss)
# ---------------------------------------------------------------------------------------------
ACCENT_CYAN = "#38bdf8"
ACCENT_SKY = "#7dd3fc"
ACCENT_VIOLET = "#a78bfa"
ACCENT_INDIGO = "#818cf8"
ACCENT_EMERALD = "#34d399"
ACCENT_ROSE = "#fb7185"
ACCENT_RED = "#f43f5e"
ACCENT_AMBER = "#fbbf24"
TEXT_PRIMARY = "#e6edf7"
TEXT_SECONDARY = "#9fb0c6"
TEXT_MUTED = "#64748b"
BASE_DARK = "#070b1a"
POPUP_DARK = "#0f1529"

UI_DIR = Path(__file__).resolve().parent
ASSETS_DIR = UI_DIR / "assets"
STYLESHEET_PATH = UI_DIR / "styles.qss"

# ---------------------------------------------------------------------------------------------
# Fonts, palette, stylesheet
# ---------------------------------------------------------------------------------------------
_UI_FONT_CANDIDATES = ("Segoe UI", "Segoe UI Variable Text", "Segoe UI Variable", "Inter", "Noto Sans",
                       "DejaVu Sans", "Helvetica Neue", "Arial")
_MONO_FONT_CANDIDATES = ("Cascadia Mono", "Consolas", "JetBrains Mono", "DejaVu Sans Mono", "Menlo",
                         "Courier New")


def _first_available_family(candidates: Iterable[str]) -> str:
    """Return the first installed font family among candidates (or the app default)."""
    try:
        installed = set(QFontDatabase.families())
    except Exception:
        installed = set()
    for fam in candidates:
        if fam in installed:
            return fam
    app = QGuiApplication.instance()
    return app.font().family() if app is not None else ""


def pick_ui_font_family() -> str:
    """Preferred UI font family available on this machine."""
    return _first_available_family(_UI_FONT_CANDIDATES)


def pick_mono_font_family() -> str:
    """Preferred monospace font family available on this machine."""
    return _first_available_family(_MONO_FONT_CANDIDATES)


def ui_font(pixel_size: int = 13, weight: QFont.Weight = QFont.Weight.Normal, mono: bool = False) -> QFont:
    """Build a pixel-sized UI (or monospace) font."""
    font = QFont(pick_mono_font_family() if mono else pick_ui_font_family())
    font.setPixelSize(max(1, int(pixel_size)))
    font.setWeight(weight)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return font


def dark_palette() -> QPalette:
    """Dark navy palette so native widgets never flash white before the stylesheet applies."""
    pal = QPalette()
    window = QColor("#0b1022")
    base = QColor("#0d1326")
    text = QColor(TEXT_PRIMARY)
    muted = QColor(TEXT_MUTED)
    pal.setColor(QPalette.ColorRole.Window, window)
    pal.setColor(QPalette.ColorRole.WindowText, text)
    pal.setColor(QPalette.ColorRole.Base, base)
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#111a33"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(POPUP_DARK))
    pal.setColor(QPalette.ColorRole.ToolTipText, text)
    pal.setColor(QPalette.ColorRole.PlaceholderText, muted)
    pal.setColor(QPalette.ColorRole.Text, text)
    pal.setColor(QPalette.ColorRole.Button, QColor("#141c36"))
    pal.setColor(QPalette.ColorRole.ButtonText, text)
    pal.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#0ea5e9"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, QColor(ACCENT_CYAN))
    pal.setColor(QPalette.ColorRole.Light, QColor("#26304f"))
    pal.setColor(QPalette.ColorRole.Midlight, QColor("#1b2442"))
    pal.setColor(QPalette.ColorRole.Mid, QColor("#141b33"))
    pal.setColor(QPalette.ColorRole.Dark, QColor("#070b18"))
    pal.setColor(QPalette.ColorRole.Shadow, QColor("#000000"))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor("#56627a"))
    return pal


def apply_app_theme(app: QApplication) -> None:
    """Fusion style + dark palette + UI font for the whole application."""
    try:
        app.setStyle("Fusion")
    except Exception as exc:  # pragma: no cover - style plugin missing
        logger.debug(f"Fusion style unavailable: {exc}")
    app.setPalette(dark_palette())
    app.setFont(ui_font(13))


def load_glass_stylesheet(path: Optional[Path] = None) -> str:
    """Read styles.qss and resolve the @ASSETS@ token to the absolute assets directory."""
    qss_path = Path(path) if path else STYLESHEET_PATH
    if not qss_path.exists():
        return ""
    try:
        text = qss_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning(f"Could not read stylesheet {qss_path}: {exc}")
        return ""
    return text.replace("@ASSETS@", ASSETS_DIR.as_posix())


def set_role(widget: QWidget, role: str) -> QWidget:
    """Tag a widget with a QSS `role` property (title, subtitle, caption, chip, ...)."""
    widget.setProperty("role", role)
    return widget


def set_variant(button: QPushButton, variant: str) -> QPushButton:
    """Tag a button with a QSS `variant` property (primary, danger, ghost)."""
    button.setProperty("variant", variant)
    return button


def repolish(widget: QWidget) -> None:
    """Re-apply the stylesheet after a dynamic property change."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def make_transparent_scroll(scroll: QAbstractScrollArea) -> None:
    """Make a scroll area, its viewport and its content see-through so glass shows behind."""
    scroll.setObjectName("glassScroll")
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    viewport = scroll.viewport()
    viewport.setObjectName("glassScrollViewport")
    viewport.setAutoFillBackground(False)
    content = getattr(scroll, "widget", None)
    if callable(content):
        inner = content()
        if inner is not None:
            inner.setObjectName(inner.objectName() or "glassScrollContent")
            inner.setAutoFillBackground(False)
            inner.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


# ---------------------------------------------------------------------------------------------
# Windows 11 acrylic / mica backdrop (DWM)
# ---------------------------------------------------------------------------------------------
_MIN_BACKDROP_BUILD = 22621
DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_CAPTION_COLOR = 35
DWMWA_SYSTEMBACKDROP_TYPE = 38
_DWMSBT = {"none": 1, "mica": 2, "acrylic": 3, "tabbed": 4}


class _MARGINS(ctypes.Structure):
    """Win32 MARGINS structure for DwmExtendFrameIntoClientArea."""
    _fields_ = [("cxLeftWidth", ctypes.c_int), ("cxRightWidth", ctypes.c_int),
                ("cyTopHeight", ctypes.c_int), ("cyBottomHeight", ctypes.c_int)]


def windows_build() -> int:
    """Windows build number (0 on other platforms)."""
    if sys.platform != "win32":
        return 0
    try:
        return int(sys.getwindowsversion().build)
    except Exception:
        return 0


def _native_windows_platform() -> bool:
    """True when Qt runs on the real Windows QPA plugin (not offscreen/minimal)."""
    if sys.platform != "win32" or QGuiApplication.instance() is None:
        return False
    return QGuiApplication.platformName() == "windows"


def backdrop_supported() -> bool:
    """Whether a DWM acrylic/mica backdrop can be attempted (Win11 22H2+, opt-out FORENSIC_GLASS=0)."""
    if os.environ.get("FORENSIC_GLASS", "1").strip().lower() in ("0", "false", "off", "no"):
        return False
    return _native_windows_platform() and windows_build() >= _MIN_BACKDROP_BUILD


def prepare_window_for_backdrop(window: QWidget) -> bool:
    """Set WA_TranslucentBackground BEFORE the first show when a backdrop can be attempted."""
    if not backdrop_supported():
        return False
    if window.isVisible():
        return window.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    return True


def _dwm_set_int(hwnd: int, attribute: int, value: int) -> bool:
    """DwmSetWindowAttribute with a 4-byte integer payload; True on S_OK."""
    data = ctypes.c_int(int(value))
    hr = ctypes.windll.dwmapi.DwmSetWindowAttribute(
        ctypes.c_void_p(hwnd), ctypes.c_uint(attribute), ctypes.byref(data), ctypes.sizeof(data))
    return int(hr) == 0


def set_dark_title_bar(window: QWidget, caption_hex: Optional[str] = None) -> bool:
    """Dark immersive title bar (and optional caption colour) on Windows 10/11."""
    if not _native_windows_platform():
        return False
    try:
        hwnd = int(window.winId())
        ok = _dwm_set_int(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1)
        if not ok:
            ok = _dwm_set_int(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_OLD, 1)
        if caption_hex and windows_build() >= 22000:
            c = QColor(caption_hex)
            colorref = (c.blue() << 16) | (c.green() << 8) | c.red()
            _dwm_set_int(hwnd, DWMWA_CAPTION_COLOR, colorref)
        return ok
    except Exception as exc:
        logger.debug(f"Dark title bar unavailable: {exc}")
        return False


def enable_window_backdrop(window: QWidget, mode: str = "acrylic") -> bool:
    """
    Enable a true translucent + blurred (acrylic / mica) window backdrop via DWM.
    Requires WA_TranslucentBackground set before the first show (see prepare_window_for_backdrop).
    Always applies the dark title bar. Returns False (UI stays opaque) on any failure.
    """
    if not backdrop_supported() or not window.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground):
        set_dark_title_bar(window, caption_hex="#0a1024")
        return False
    set_dark_title_bar(window)
    try:
        hwnd = int(window.winId())
        if not _dwm_set_int(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, _DWMSBT.get(mode, 3)):
            logger.info("DWM system backdrop rejected; using opaque glass theme")
            return False
        margins = _MARGINS(-1, -1, -1, -1)
        hr = ctypes.windll.dwmapi.DwmExtendFrameIntoClientArea(ctypes.c_void_p(hwnd), ctypes.byref(margins))
        if int(hr) != 0:
            _dwm_set_int(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, _DWMSBT["none"])
            logger.info("DwmExtendFrameIntoClientArea failed; using opaque glass theme")
            return False
        window.setProperty("glassBackdrop", True)
        logger.info(f"Windows {mode} backdrop enabled")
        return True
    except Exception as exc:
        logger.warning(f"Window backdrop unavailable: {exc}")
        return False


# ---------------------------------------------------------------------------------------------
# Aurora background (gradient + heavily blurred colour orbs, cached pixmaps)
# ---------------------------------------------------------------------------------------------
# (cx, cy, radius as fraction of the diagonal, RGB 0..1, strength)
_AURORA_ORBS = (
    (0.06, 0.02, 0.34, (0.22, 0.74, 0.97), 0.70),   # cyan, top-left
    (0.92, 0.08, 0.32, (0.65, 0.55, 0.98), 0.62),   # violet, top-right
    (0.72, 0.98, 0.38, (0.10, 0.80, 0.66), 0.46),   # teal / emerald, bottom
    (0.32, 0.66, 0.26, (0.39, 0.40, 0.95), 0.36),   # indigo, centre-left
    (0.00, 0.98, 0.22, (0.93, 0.36, 0.65), 0.18),   # faint rose, bottom-left
)
_BASE_TOP = np.array([0.045, 0.060, 0.135], dtype=np.float32)      # #0b0f22
_BASE_BOTTOM = np.array([0.016, 0.024, 0.066], dtype=np.float32)   # #040611


@lru_cache(maxsize=1)
def _dither_tile() -> np.ndarray:
    """Fixed 128x128 noise tile used to dither gradients (prevents 8-bit banding)."""
    rng = np.random.RandomState(7)
    return ((rng.rand(128, 128, 1).astype(np.float32) - 0.5) * (2.2 / 255.0))


def render_aurora_arrays(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render the aurora backdrop at (width, height) device pixels.
    Returns (aurora_rgb, frosted_rgb) uint8 arrays; frosted is a heavier blur used under glass panels.
    """
    width, height = max(8, int(width)), max(8, int(height))
    sw, sh = max(48, width // 6), max(32, height // 6)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    u = xx / max(sw - 1, 1)
    v = yy / max(sh - 1, 1)

    t = np.clip(0.65 * v + 0.35 * u, 0.0, 1.0)[..., None]
    img = _BASE_TOP * (1.0 - t) + _BASE_BOTTOM * t
    diag = math.hypot(sw, sh)
    for cx, cy, rad, rgb, strength in _AURORA_ORBS:
        d2 = ((xx - cx * sw) ** 2 + (yy - cy * sh) ** 2) / max((rad * diag) ** 2, 1e-6)
        alpha = (strength * np.exp(-d2 * 2.0))[..., None] * 0.55
        img = img + (np.asarray(rgb, dtype=np.float32) - img) * alpha

    vignette = 1.0 - 0.32 * np.clip(((u - 0.5) ** 2 + (v - 0.5) ** 2) * 2.0, 0.0, 1.0)
    img = img * vignette[..., None]

    frosted_small = cv2.GaussianBlur(img, (0, 0), sigmaX=max(2.0, sw * 0.035))
    frosted_small = frosted_small * 0.94 + 0.012

    def _finish(small: np.ndarray) -> np.ndarray:
        big = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
        tile = _dither_tile()
        reps_y = (height + tile.shape[0] - 1) // tile.shape[0]
        reps_x = (width + tile.shape[1] - 1) // tile.shape[1]
        noise = np.tile(tile, (reps_y, reps_x, 1))[:height, :width]
        return np.clip((big + noise) * 255.0, 0, 255).astype(np.uint8)

    return _finish(img), _finish(frosted_small)


def numpy_rgb_to_qimage(arr: np.ndarray) -> QImage:
    """Deep-copy an RGB uint8 numpy image into a QImage (no dangling buffer)."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, int(arr.strides[0]), QImage.Format.Format_RGB888).copy()


class AuroraBackground(QWidget):
    """Central backdrop painting a deep-navy gradient with blurred colour orbs from a cached pixmap."""

    BACKDROP_OPACITY = 0.58   # aurora alpha over the OS acrylic blur
    _BUCKET = 96              # re-render only when the size crosses a 96 px bucket

    backdrop_mode_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("auroraBackground")
        self._backdrop = False
        self._key: Optional[Tuple[int, int, float]] = None
        self._aurora_px: Optional[QPixmap] = None
        self._frosted_px: Optional[QPixmap] = None   # device-pixel sized, dpr 1 (brush sampling)
        self._dpr = 1.0
        self.render_count = 0

    def is_backdrop(self) -> bool:
        """True while the OS acrylic backdrop shows through (partial alpha painting)."""
        return self._backdrop

    def set_backdrop_mode(self, enabled: bool) -> None:
        """Switch between opaque aurora and translucent aurora over the OS backdrop."""
        enabled = bool(enabled)
        if enabled != self._backdrop:
            self._backdrop = enabled
            self.backdrop_mode_changed.emit(enabled)
            self.update()

    def _ensure_cache(self) -> None:
        dpr = max(1.0, float(self.devicePixelRatioF()))
        b = self._BUCKET
        bw = ((max(1, self.width()) + b - 1) // b) * b
        bh = ((max(1, self.height()) + b - 1) // b) * b
        key = (bw, bh, round(dpr, 3))
        if key == self._key and self._aurora_px is not None:
            return
        dw, dh = int(math.ceil(bw * dpr)), int(math.ceil(bh * dpr))
        aurora, frosted = render_aurora_arrays(dw, dh)

        img = numpy_rgb_to_qimage(aurora)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 15))
        step = 28.0 * dpr
        radius = 0.9 * dpr
        y = step * 0.5
        while y < dh:
            x = step * 0.5
            while x < dw:
                painter.drawEllipse(QPointF(x, y), radius, radius)
                x += step
            y += step
        painter.end()

        pixmap = QPixmap.fromImage(img)
        pixmap.setDevicePixelRatio(dpr)
        self._aurora_px = pixmap
        self._frosted_px = QPixmap.fromImage(numpy_rgb_to_qimage(frosted))
        self._dpr = dpr
        self._key = key
        self.render_count += 1

    def frosted_source(self) -> Tuple[Optional[QPixmap], float]:
        """Pre-blurred backdrop (device pixels, dpr 1) and the device pixel ratio for sampling."""
        self._ensure_cache()
        return self._frosted_px, self._dpr

    def paintEvent(self, event) -> None:
        self._ensure_cache()
        painter = QPainter(self)
        if self._backdrop:
            painter.setOpacity(self.BACKDROP_OPACITY)
        r = event.rect()
        d = self._dpr
        src = QRectF(r.x() * d, r.y() * d, r.width() * d, r.height() * d)
        painter.drawPixmap(QRectF(r), self._aurora_px, src)
        painter.end()


def find_aurora(widget: Optional[QWidget]) -> Optional[AuroraBackground]:
    """Nearest AuroraBackground ancestor of a widget (or None)."""
    w = widget.parentWidget() if widget is not None else None
    while w is not None:
        if isinstance(w, AuroraBackground):
            return w
        w = w.parentWidget()
    return None


# ---------------------------------------------------------------------------------------------
# Frosted glass panel
# ---------------------------------------------------------------------------------------------
def paint_glass(painter: QPainter, widget: QWidget, rect: QRectF, radius: float, *, tint: float = 0.055,
                border_alpha: float = 0.14, highlight: float = 0.26, accent: Optional[QColor] = None,
                border_color: Optional[QColor] = None, sample: bool = True) -> None:
    """
    Paint a frosted glass surface into `rect` of `widget`: blurred backdrop sample (when not in OS
    backdrop mode) + white tint gradient + 1px border + top highlight rim. Cheap: no per-frame blur.
    """
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    aurora = find_aurora(widget) if sample else None
    backdrop = aurora is not None and aurora.is_backdrop()

    if aurora is not None and not backdrop:
        frosted, dpr = aurora.frosted_source()
        if frosted is not None:
            off = widget.mapTo(aurora, QPoint(0, 0))
            brush = QBrush(frosted)
            brush.setTransform(QTransform().translate(-off.x(), -off.y()).scale(1.0 / dpr, 1.0 / dpr))
            painter.fillPath(path, brush)
    elif sample:
        painter.fillPath(path, QColor(10, 14, 32, 105 if backdrop else 215))

    grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    grad.setColorAt(0.0, QColor(255, 255, 255, int(255 * min(1.0, tint + 0.035))))
    grad.setColorAt(1.0, QColor(255, 255, 255, int(255 * tint)))
    painter.fillPath(path, grad)

    if accent is not None:
        glow = QRadialGradient(rect.topLeft() + QPointF(rect.width() * 0.12, 0.0), max(rect.width(), 1.0) * 0.7)
        c0 = QColor(accent)
        c0.setAlphaF(0.10)
        c1 = QColor(accent)
        c1.setAlphaF(0.0)
        glow.setColorAt(0.0, c0)
        glow.setColorAt(1.0, c1)
        painter.fillPath(path, glow)

    pen_color = QColor(border_color) if border_color is not None else QColor(255, 255, 255, int(255 * border_alpha))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(pen_color, 1.0))
    painter.drawPath(path)

    if highlight > 0:
        rim = QLinearGradient(rect.topLeft(), rect.topLeft() + QPointF(0.0, min(rect.height(), radius * 2.6)))
        rim.setColorAt(0.0, QColor(255, 255, 255, int(255 * highlight)))
        rim.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.setPen(QPen(QBrush(rim), 1.0))
        painter.drawPath(path)


class GlassPanel(QFrame):
    """Frosted glass panel sampling the matching region of the pre-blurred aurora beneath it."""

    def __init__(self, parent: Optional[QWidget] = None, *, radius: float = 16.0, tint: float = 0.055,
                 border_alpha: float = 0.14, accent: Optional[str] = None, highlight: float = 0.26):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName(self.objectName() or "glassPanel")
        self._radius = float(radius)
        self._tint = float(tint)
        self._border_alpha = float(border_alpha)
        self._highlight = float(highlight)
        self._accent = QColor(accent) if accent else None

    def radius(self) -> float:
        """Corner radius in logical pixels."""
        return self._radius

    def set_accent(self, color: Optional[str]) -> None:
        """Optional accent tint glowing from the top-left corner."""
        self._accent = QColor(color) if color else None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        paint_glass(painter, self, rect, self._radius, tint=self._tint, border_alpha=self._border_alpha,
                    highlight=self._highlight, accent=self._accent)
        painter.end()


# ---------------------------------------------------------------------------------------------
# Animated hover / focus glow (QGraphicsDropShadowEffect, offset 0)
# ---------------------------------------------------------------------------------------------
def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class HoverGlow(QObject):
    """Animates a zero-offset drop-shadow 'glow' on Enter/Leave/FocusIn/FocusOut (disabled at rest)."""

    def __init__(self, widget: QWidget, color: str = ACCENT_CYAN, *, radius: float = 26.0, alpha: float = 0.85,
                 rest_radius: float = 0.0, rest_alpha: float = 0.0, focus: bool = True, duration_ms: int = 190):
        super().__init__(widget)
        self._widget = widget
        self._color = QColor(color)
        self._radius, self._alpha = float(radius), float(alpha)
        self._rest_radius, self._rest_alpha = float(rest_radius), float(rest_alpha)
        self._focus = focus
        self._hovered = False
        self._value = 0.0
        self.effect = QGraphicsDropShadowEffect(widget)
        self.effect.setOffset(0.0, 0.0)
        widget.setGraphicsEffect(self.effect)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(int(duration_ms))
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._apply)
        widget.installEventFilter(self)
        self._apply(0.0)

    @property
    def value(self) -> float:
        """Current glow intensity 0..1."""
        return self._value

    @property
    def hovered(self) -> bool:
        return self._hovered

    def set_color(self, color: str) -> None:
        """Change the glow colour (e.g. identity colour)."""
        self._color = QColor(color)
        self._apply(self._value)

    def target(self) -> float:
        """Intensity the glow is heading to given hover/focus/enabled state."""
        w = self._widget
        if not w.isEnabled():
            return 0.0
        if self._hovered:
            return 1.0
        if self._focus and w.hasFocus():
            return 0.6
        return 0.0

    def refresh(self) -> None:
        """Animate towards the current target intensity."""
        goal = self.target()
        self._anim.stop()
        if abs(goal - self._value) < 1e-3:
            self._apply(goal)
            return
        self._anim.setStartValue(float(self._value))
        self._anim.setEndValue(float(goal))
        self._anim.start()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._widget:
            et = event.type()
            if et in (QEvent.Type.Enter, QEvent.Type.HoverEnter):
                self._hovered = True
                self.refresh()
            elif et in (QEvent.Type.Leave, QEvent.Type.HoverLeave, QEvent.Type.Hide):
                self._hovered = False
                self.refresh()
            elif et in (QEvent.Type.FocusIn, QEvent.Type.FocusOut, QEvent.Type.EnabledChange):
                if et == QEvent.Type.EnabledChange and not self._widget.isEnabled():
                    self._hovered = False
                self.refresh()
        return False

    def _apply(self, v) -> None:
        self._value = float(v)
        alpha = max(0.0, min(1.0, _lerp(self._rest_alpha, self._alpha, self._value)))
        radius = max(0.0, _lerp(self._rest_radius, self._radius, self._value))
        c = QColor(self._color)
        c.setAlphaF(alpha)
        self.effect.setColor(c)
        self.effect.setBlurRadius(radius)
        self.effect.setEnabled(alpha > 0.004 and radius > 0.5)


_VARIANT_GLOW = {
    "primary": (ACCENT_CYAN, 30.0, 0.95),
    "danger": (ACCENT_ROSE, 30.0, 0.95),
    "ghost": (ACCENT_SKY, 22.0, 0.9),
    "input": (ACCENT_CYAN, 18.0, 0.75),
}


def apply_hover_glow(widget: QWidget, color: Optional[str] = None, *, radius: Optional[float] = None,
                     alpha: Optional[float] = None, focus: bool = True) -> Optional[HoverGlow]:
    """
    Attach an animated hover/focus glow to a widget. Returns None when the widget already has a
    graphics effect (max one per widget) or is opted out via the `noGlow` property.
    """
    if widget is None or widget.property("noGlow"):
        return None
    existing = getattr(widget, "_hover_glow", None)
    if isinstance(existing, HoverGlow):
        if color:
            existing.set_color(color)
        return existing
    if widget.graphicsEffect() is not None:
        return None
    variant = str(widget.property("variant") or ("input" if not isinstance(widget, QPushButton) else "ghost"))
    if isinstance(widget, QPushButton) and (widget.objectName() == "btn_danger"):
        variant = "danger"
    v_color, v_radius, v_alpha = _VARIANT_GLOW.get(variant, _VARIANT_GLOW["ghost"])
    glow = HoverGlow(widget, color or v_color, radius=radius if radius is not None else v_radius,
                     alpha=alpha if alpha is not None else v_alpha, focus=focus)
    widget._hover_glow = glow
    return glow


def install_hover_glows(root: QWidget) -> List[HoverGlow]:
    """Attach hover glows to every button and input below `root` (skips widgets with effects)."""
    glows: List[HoverGlow] = []
    types = (QPushButton, QComboBox, QLineEdit, QAbstractSpinBox)
    for widget in root.findChildren(QWidget):
        if not isinstance(widget, types):
            continue
        # Line edits embedded in spin boxes / combos glow through their parent instead
        if isinstance(widget, QLineEdit) and isinstance(widget.parentWidget(), (QAbstractSpinBox, QComboBox)):
            continue
        glow = apply_hover_glow(widget)
        if glow is not None:
            glows.append(glow)
    return glows


class HoverProgress(QObject):
    """Animated 0..1 hover progress for custom-painted widgets (calls widget.update() while animating)."""

    changed = pyqtSignal(float)

    def __init__(self, widget: QWidget, duration_ms: int = 200):
        super().__init__(widget)
        self._widget = widget
        self._value = 0.0
        self._hovered = False
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(int(duration_ms))
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_value)
        widget.installEventFilter(self)

    @property
    def value(self) -> float:
        return self._value

    @property
    def hovered(self) -> bool:
        return self._hovered

    def _animate(self, goal: float) -> None:
        self._anim.stop()
        self._anim.setStartValue(float(self._value))
        self._anim.setEndValue(float(goal))
        self._anim.start()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._widget:
            et = event.type()
            if et == QEvent.Type.Enter:
                self._hovered = True
                self._animate(1.0)
            elif et in (QEvent.Type.Leave, QEvent.Type.Hide):
                self._hovered = False
                self._animate(0.0)
        return False

    def _on_value(self, v) -> None:
        self._value = float(v)
        self.changed.emit(self._value)
        self._widget.update()


# ---------------------------------------------------------------------------------------------
# Pulsing glow (LIVE indicator)
# ---------------------------------------------------------------------------------------------
class PulseAnimator(QObject):
    """Looping 0 -> 1 -> 0 value for paint-based pulsing glows."""

    valueChanged = pyqtSignal(float)

    def __init__(self, parent: Optional[QObject] = None, period_ms: int = 1700):
        super().__init__(parent)
        self._value = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setKeyValueAt(0.5, 1.0)
        self._anim.setEndValue(0.0)
        self._anim.setDuration(int(period_ms))
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._anim.valueChanged.connect(self._on_value)

    @property
    def value(self) -> float:
        return self._value

    def is_running(self) -> bool:
        return self._anim.state() == QVariantAnimation.State.Running

    def start(self) -> None:
        if not self.is_running():
            self._anim.start()

    def stop(self) -> None:
        self._anim.stop()
        self._on_value(0.0)

    def _on_value(self, v) -> None:
        self._value = float(v)
        self.valueChanged.emit(self._value)


class PulseGlow(QObject):
    """Continuously pulsing zero-offset drop-shadow glow for small indicator widgets."""

    def __init__(self, widget: QWidget, color: str = ACCENT_RED, *, min_radius: float = 6.0,
                 max_radius: float = 24.0, min_alpha: float = 0.35, max_alpha: float = 0.95,
                 period_ms: int = 1700):
        super().__init__(widget)
        self._widget = widget
        self._color = QColor(color)
        self._min_radius, self._max_radius = float(min_radius), float(max_radius)
        self._min_alpha, self._max_alpha = float(min_alpha), float(max_alpha)
        self.effect = QGraphicsDropShadowEffect(widget)
        self.effect.setOffset(0.0, 0.0)
        widget.setGraphicsEffect(self.effect)
        self._pulse = PulseAnimator(self, period_ms)
        self._pulse.valueChanged.connect(self._apply)
        self._apply(0.0)
        self.effect.setEnabled(False)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self._apply(self._pulse.value)

    def is_running(self) -> bool:
        return self._pulse.is_running()

    def start(self) -> None:
        self.effect.setEnabled(True)
        self._pulse.start()

    def stop(self) -> None:
        self._pulse.stop()
        self.effect.setEnabled(False)

    def _apply(self, v: float) -> None:
        c = QColor(self._color)
        c.setAlphaF(max(0.0, min(1.0, _lerp(self._min_alpha, self._max_alpha, v))))
        self.effect.setColor(c)
        self.effect.setBlurRadius(_lerp(self._min_radius, self._max_radius, v))


def apply_pulse_glow(widget: QWidget, color: str = ACCENT_RED, **kwargs) -> Optional[PulseGlow]:
    """Attach a pulsing glow to a small widget (None if it already has a graphics effect)."""
    if widget is None or widget.graphicsEffect() is not None:
        return None
    glow = PulseGlow(widget, color, **kwargs)
    widget._pulse_glow = glow
    return glow


class LiveIndicator(QWidget):
    """Pill showing IDLE / LIVE / PAUSED with a paint-based pulsing dot halo while live."""

    STATES = {
        "idle": ("IDLE", TEXT_MUTED),
        "live": ("LIVE", ACCENT_RED),
        "paused": ("PAUSED", ACCENT_AMBER),
        "error": ("ERROR", ACCENT_ROSE),
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._state = "idle"
        self._detail = ""
        self._pulse = PulseAnimator(self, 1500)
        self._pulse.valueChanged.connect(lambda _v: self.update())
        self.setFixedHeight(30)
        self.setMinimumWidth(88)
        self.setFont(ui_font(11, QFont.Weight.Bold))

    def state(self) -> str:
        return self._state

    def set_state(self, state: str, detail: str = "") -> None:
        """Switch indicator state; pulses only while live."""
        self._state = state if state in self.STATES else "idle"
        self._detail = detail
        if self._state == "live":
            self._pulse.start()
        else:
            self._pulse.stop()
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        label = self.STATES[self._state][0] + (f"  {self._detail}" if self._detail else "")
        return QSize(fm.horizontalAdvance(label) + 44, 30)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        label, color_hex = self.STATES[self._state]
        color = QColor(color_hex)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        bg = QColor(color)
        bg.setAlphaF(0.12 if self._state != "idle" else 0.06)
        border = QColor(color)
        border.setAlphaF(0.45 if self._state != "idle" else 0.18)
        p.setPen(QPen(border, 1.0))
        p.setBrush(bg)
        p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)

        center = QPointF(rect.left() + 16, rect.center().y())
        if self._state == "live":
            k = self._pulse.value
            halo = QRadialGradient(center, 5 + 6 * k)
            c0 = QColor(color)
            c0.setAlphaF(0.55 * (1.0 - 0.5 * k))
            c1 = QColor(color)
            c1.setAlphaF(0.0)
            halo.setColorAt(0.0, c0)
            halo.setColorAt(1.0, c1)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(halo))
            p.drawEllipse(center, 5 + 6 * k, 5 + 6 * k)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawEllipse(center, 4.0, 4.0)

        p.setPen(QColor(TEXT_PRIMARY) if self._state != "idle" else QColor(TEXT_SECONDARY))
        p.setFont(self.font())
        text_rect = rect.adjusted(28, 0, -10, 0)
        text = label + (f"  {self._detail}" if self._detail else "")
        p.drawText(text_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), text)
        p.end()


# ---------------------------------------------------------------------------------------------
# Small painted assets: colour swatches, rounded thumbnails, brand mark
# ---------------------------------------------------------------------------------------------
_SWATCH_CACHE: Dict[Tuple[str, int, float], QPixmap] = {}


def _valid_hex(hex_color: Optional[str]) -> bool:
    return bool(hex_color) and QColor(str(hex_color)).isValid()


def swatch_pixmap(hex_color: Optional[str], size: int = 14, radius: float = 4.0) -> QPixmap:
    """Rounded colour swatch with a subtle light ring (dashed ring when the colour is unknown)."""
    app = QGuiApplication.instance()
    dpr = float(app.devicePixelRatio()) if app is not None else 1.0
    key = (str(hex_color or ""), int(size), dpr)
    cached = _SWATCH_CACHE.get(key)
    if cached is not None:
        return cached
    px = QPixmap(int(math.ceil(size * dpr)), int(math.ceil(size * dpr)))
    px.setDevicePixelRatio(dpr)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    rect = QRectF(0.75, 0.75, size - 1.5, size - 1.5)
    if _valid_hex(hex_color):
        p.setBrush(QColor(str(hex_color)))
        p.setPen(QPen(QColor(255, 255, 255, 120), 1.0))
    else:
        p.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(QColor(255, 255, 255, 110), 1.0)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
    p.drawRoundedRect(rect, radius, radius)
    p.end()
    if len(_SWATCH_CACHE) > 512:
        _SWATCH_CACHE.clear()
    _SWATCH_CACHE[key] = px
    return px


def swatch_icon(hex_color: Optional[str], size: int = 14) -> QIcon:
    """QIcon wrapper around swatch_pixmap (for combo boxes and table cells)."""
    return QIcon(swatch_pixmap(hex_color, size))


def rounded_cover_pixmap(image: QImage, width: int, height: int, radius: float = 8.0,
                         dpr: float = 1.0) -> QPixmap:
    """Scale an image to cover (width x height), centre-crop and round its corners (antialiased)."""
    dw, dh = max(1, int(round(width * dpr))), max(1, int(round(height * dpr)))
    scaled = image.scaled(dw, dh, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                          Qt.TransformationMode.SmoothTransformation)
    ox = (scaled.width() - dw) // 2
    oy = (scaled.height() - dh) // 2
    out = QImage(dw, dh, QImage.Format.Format_ARGB32_Premultiplied)
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    brush = QBrush(scaled)
    brush.setTransform(QTransform().translate(-ox, -oy))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(brush)
    p.drawRoundedRect(QRectF(0, 0, dw, dh), radius * dpr, radius * dpr)
    p.end()
    px = QPixmap.fromImage(out)
    px.setDevicePixelRatio(dpr)
    return px


def load_rounded_thumbnail(path: Optional[str], width: int, height: int, radius: float = 8.0,
                           dpr: float = 1.0) -> Optional[QPixmap]:
    """Load an image file as a rounded cover thumbnail (None if missing/unreadable)."""
    if not path or not os.path.exists(path):
        return None
    img = QImage(path)
    if img.isNull():
        return None
    return rounded_cover_pixmap(img, width, height, radius, dpr)


def paint_brand_mark(painter: QPainter, rect: QRectF) -> None:
    """Paint the app's reticle brand mark (gradient ring + crosshair) into rect."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = rect.center()
    r = min(rect.width(), rect.height()) / 2.0
    glow = QRadialGradient(c, r)
    glow.setColorAt(0.0, QColor(56, 189, 248, 90))
    glow.setColorAt(1.0, QColor(56, 189, 248, 0))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(glow))
    painter.drawEllipse(c, r, r)
    grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
    grad.setColorAt(0.0, QColor(ACCENT_CYAN))
    grad.setColorAt(1.0, QColor(ACCENT_VIOLET))
    pen = QPen(QBrush(grad), max(1.5, r * 0.13))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    ring = r * 0.62
    painter.drawEllipse(c, ring, ring)
    tick = r * 0.30
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        a = QPointF(c.x() + dx * (ring - tick * 0.2), c.y() + dy * (ring - tick * 0.2))
        b = QPointF(c.x() + dx * (ring + tick), c.y() + dy * (ring + tick))
        painter.drawLine(a, b)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#e0f2fe"))
    painter.drawEllipse(c, r * 0.14, r * 0.14)
    painter.restore()


def brand_icon(size: int = 64) -> QIcon:
    """Application icon rendered from the brand mark."""
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    bg = QPainterPath()
    bg.addRoundedRect(QRectF(2, 2, size - 4, size - 4), size * 0.24, size * 0.24)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.fillPath(bg, QColor("#0b1124"))
    paint_brand_mark(p, QRectF(6, 6, size - 12, size - 12))
    p.end()
    return QIcon(px)


class BrandMark(QWidget):
    """Header logo widget painting the reticle brand mark."""

    def __init__(self, size: int = 42, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        paint_brand_mark(p, QRectF(self.rect()))
        p.end()


def chip_label(text: str, color: Optional[str] = None, parent: Optional[QWidget] = None,
               muted: bool = False) -> QLabel:
    """Small rounded chip label tinted with an accent colour."""
    lbl = QLabel(text, parent)
    c = QColor(color or (TEXT_MUTED if muted else ACCENT_SKY))
    bg = QColor(c)
    bg.setAlphaF(0.14)
    border = QColor(c)
    border.setAlphaF(0.40)
    fg = QColor(c).lighter(135) if not muted else QColor(TEXT_SECONDARY)
    lbl.setStyleSheet(
        f"QLabel {{ background-color: rgba({bg.red()},{bg.green()},{bg.blue()},{bg.alphaF():.2f});"
        f" border: 1px solid rgba({border.red()},{border.green()},{border.blue()},{border.alphaF():.2f});"
        f" border-radius: 9px; padding: 1px 8px; color: {fg.name()}; font-size: 11px; font-weight: 600; }}")
    return lbl


def rgba_css(color: str, alpha: float) -> str:
    """CSS rgba() string for a hex colour and alpha."""
    c = QColor(color)
    return f"rgba({c.red()},{c.green()},{c.blue()},{max(0.0, min(1.0, alpha)):.3f})"


def elide(text: str, font: QFont, width: int) -> str:
    """Elide text on the right to fit width pixels."""
    return QFontMetrics(font).elidedText(text, Qt.TextElideMode.ElideRight, max(0, int(width)))


__all__ = [
    "ACCENT_CYAN", "ACCENT_SKY", "ACCENT_VIOLET", "ACCENT_INDIGO", "ACCENT_EMERALD", "ACCENT_ROSE", "ACCENT_RED",
    "ACCENT_AMBER", "TEXT_PRIMARY", "TEXT_SECONDARY", "TEXT_MUTED", "BASE_DARK", "POPUP_DARK",
    "pick_ui_font_family", "pick_mono_font_family", "ui_font", "dark_palette", "apply_app_theme",
    "load_glass_stylesheet", "set_role", "set_variant", "repolish", "make_transparent_scroll",
    "backdrop_supported", "prepare_window_for_backdrop", "set_dark_title_bar", "enable_window_backdrop",
    "render_aurora_arrays", "numpy_rgb_to_qimage", "AuroraBackground", "find_aurora", "paint_glass",
    "GlassPanel", "HoverGlow", "apply_hover_glow", "install_hover_glows", "HoverProgress", "PulseAnimator",
    "PulseGlow", "apply_pulse_glow", "LiveIndicator", "swatch_pixmap", "swatch_icon", "rounded_cover_pixmap",
    "load_rounded_thumbnail", "paint_brand_mark", "brand_icon", "BrandMark", "chip_label", "rgba_css", "elide",
]
