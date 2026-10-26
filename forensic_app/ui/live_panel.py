"""
Live Identity Panel
Real-time "identities in view" glass cards (thumbnail, identity colour, clothing swatches, accessories, status)
plus a most-recent-first identity event feed. Updates are buffered and repainted by a ~5 Hz timer.
"""

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import QModelIndex, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QScrollArea, QSizePolicy, QStyle, QStyledItemDelegate,
    QStyleOptionViewItem, QVBoxLayout, QWidget
)

from forensic_app.ui.effects import (
    ACCENT_AMBER, ACCENT_CYAN, ACCENT_EMERALD, ACCENT_SKY, ACCENT_VIOLET, TEXT_MUTED, TEXT_PRIMARY,
    TEXT_SECONDARY, GlassPanel, HoverProgress, load_rounded_thumbnail, make_transparent_scroll, paint_glass,
    set_role, ui_font
)
from forensic_app.utils.drawing import accessory_short_label, identity_color_hex

logger = logging.getLogger(__name__)

STALE_AFTER_S = 3.0        # card dims to "recently seen" after this long without updates
REMOVE_AFTER_S = 20.0      # card is removed after this long without updates
THUMB_REFRESH_S = 1.0      # crop thumbnails are re-checked at most once per second per identity
BADGE_HOLD_S = 5.0         # NEW / RE-IDENTIFIED badges stay visible this long
FEED_CAP = 100             # max identity events kept in the feed
TICK_MS = 200              # repaint throttle (~5 Hz)

# status -> (badge text, colour)
STATUS_BADGES = {
    "new": ("NEW", ACCENT_EMERALD),
    "reacquired": ("RE-IDENTIFIED", ACCENT_VIOLET),
    "confirmed": ("TRACKING", ACCENT_CYAN),
    "stale": ("RECENTLY SEEN", TEXT_MUTED),
}

EVENT_STYLES = {
    "new": ("New identity registered", ACCENT_EMERALD),
    "reacquired": ("Re-identified", ACCENT_VIOLET),
    "merged": ("Identity merged", ACCENT_AMBER),
}


def format_event_time(ts: Any) -> str:
    """HH:MM:SS from an epoch number, ISO string, or datetime (falls back to now)."""
    try:
        if isinstance(ts, datetime):
            return ts.strftime("%H:%M:%S")
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
        if isinstance(ts, str) and ts.strip():
            text = ts.strip()
            if "T" in text or " " in text:
                part = text.replace("T", " ").split(" ")[1]
                return part[:8]
            return text[:8]
    except Exception:
        pass
    return datetime.now().strftime("%H:%M:%S")


def format_identity_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Turn an identity_event payload into a feed entry dict (pure; unit-testable)."""
    kind = str(payload.get("type") or "event").lower()
    uid = str(payload.get("person_uid") or "UID-?")
    camera = str(payload.get("camera_id") or "")
    conf = payload.get("confidence")
    text, color = EVENT_STYLES.get(kind, (kind.replace("_", " ").title(), ACCENT_SKY))
    if kind == "reacquired" and conf is not None:
        try:
            text = f"Re-identified  {float(conf) * 100:.0f}% match"
        except (TypeError, ValueError):
            pass
    elif kind == "merged":
        src = payload.get("merged_from")
        text = f"Merged {src} into this identity" if src else "Identity merged"
    track = payload.get("track_id")
    return {
        "kind": kind,
        "uid": uid,
        "camera": camera,
        "time": format_event_time(payload.get("timestamp")),
        "text": text,
        "track_id": track,
        "uid_color": str(payload.get("identity_color") or identity_color_hex(uid)),
        "kind_color": color,
    }


@dataclass
class _IdentityEntry:
    """Buffered live state for one identity in view."""
    uid: str
    first_seen: float
    last_update: float
    payload: Dict[str, Any] = field(default_factory=dict)
    badge: str = "confirmed"
    badge_until: float = 0.0
    last_status: str = ""
    stale: bool = False
    dirty: bool = True
    thumb_checked: float = -1e9
    thumb_key: Optional[Tuple[str, float]] = None
    thumb: Optional[QPixmap] = None


class IdentityCard(QWidget):
    """Custom-painted glass card for one identity (glow border in the identity colour on hover)."""

    CARD_HEIGHT = 124
    THUMB_W, THUMB_H = 60, 94

    clicked = pyqtSignal(str)

    def __init__(self, uid: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.uid = uid
        self.view: Dict[str, Any] = {"uid": uid, "color": identity_color_hex(uid)}
        self.setFixedHeight(self.CARD_HEIGHT)
        self.setMinimumWidth(280)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{uid}: click to open the camera trajectory")
        self._hover = HoverProgress(self, 180)
        self._f_uid = ui_font(15, QFont.Weight.Bold)
        self._f_small = ui_font(11, QFont.Weight.DemiBold)
        self._f_body = ui_font(12)
        self._f_badge = ui_font(10, QFont.Weight.Bold)

    def set_view(self, view: Dict[str, Any]) -> None:
        self.view = view
        self.update()

    def hover_value(self) -> float:
        return self._hover.value

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.uid)
        super().mouseReleaseEvent(event)

    def sizeHint(self) -> QSize:
        return QSize(340, self.CARD_HEIGHT)

    # ---- painting ------------------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        v = self.view
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        stale = bool(v.get("stale"))
        if stale:
            p.setOpacity(0.55)
        color = QColor(v.get("color") or ACCENT_CYAN)
        hover = self._hover.value
        rect = QRectF(self.rect()).adjusted(4.5, 3.5, -4.5, -3.5)
        radius = 14.0

        # Hover glow (outer strokes in the identity colour)
        if hover > 0.01:
            for i in range(4, 0, -1):
                c = QColor(color)
                c.setAlphaF(0.22 * hover * (1.0 - i / 5.0))
                p.setPen(QPen(c, 2.0))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(rect.adjusted(-i, -i, i, i), radius + i, radius + i)

        border = QColor(color)
        border.setAlphaF(0.20 + 0.55 * hover)
        paint_glass(p, self, rect, radius, tint=0.05 + 0.03 * hover, highlight=0.22, accent=color.name(),
                    border_color=border)

        # Identity accent bar
        bar = QRectF(rect.left() + 1.5, rect.top() + 14, 3.0, rect.height() - 28)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(bar, 1.5, 1.5)

        # Thumbnail
        thumb_rect = QRectF(rect.left() + 12, rect.top() + (rect.height() - self.THUMB_H) / 2,
                            self.THUMB_W, self.THUMB_H)
        thumb = v.get("thumb")
        if isinstance(thumb, QPixmap) and not thumb.isNull():
            p.drawPixmap(thumb_rect.topLeft(), thumb)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 40), 1.0))
            p.drawRoundedRect(thumb_rect.adjusted(0.5, 0.5, -0.5, -0.5), 9, 9)
        else:
            self._paint_silhouette(p, thumb_rect, color)

        x0 = thumb_rect.right() + 12
        right = rect.right() - 12
        top = rect.top() + 8

        # Row 1: uid + track id + status badge
        badge_text = str(v.get("badge_text") or "TRACKING")
        badge_color = QColor(v.get("badge_color") or ACCENT_CYAN)
        p.setFont(self._f_badge)
        bfm = p.fontMetrics()
        bw = bfm.horizontalAdvance(badge_text) + 16
        badge_rect = QRectF(right - bw, top + 2, bw, 19)
        bg = QColor(badge_color)
        bg.setAlphaF(0.16)
        bd = QColor(badge_color)
        bd.setAlphaF(0.55)
        p.setPen(QPen(bd, 1.0))
        p.setBrush(bg)
        p.drawRoundedRect(badge_rect, 9.5, 9.5)
        p.setPen(badge_color.lighter(130))
        p.drawText(badge_rect, int(Qt.AlignmentFlag.AlignCenter), badge_text)

        p.setFont(self._f_uid)
        uid_fm = p.fontMetrics()
        uid_text = uid_fm.elidedText(str(v.get("label") or self.uid), Qt.TextElideMode.ElideRight,
                                     int(badge_rect.left() - x0 - 50))
        p.setPen(color.lighter(112))
        p.drawText(QRectF(x0, top, badge_rect.left() - x0, 24),
                   int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), uid_text)
        track = v.get("track_id")
        if track is not None:
            tx = x0 + uid_fm.horizontalAdvance(uid_text) + 8
            p.setFont(self._f_small)
            p.setPen(QColor(TEXT_MUTED))
            p.drawText(QRectF(tx, top + 1, 60, 24), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                       f"#{track}")

        # Row 2: clothing swatches
        y2 = top + 29
        col_w = (right - x0) / 2.0
        self._paint_swatch_field(p, QRectF(x0, y2, col_w - 6, 20), "TOP", v.get("upper_name", "Unknown"),
                                 v.get("upper_hex"))
        self._paint_swatch_field(p, QRectF(x0 + col_w, y2, col_w, 20), "BOTTOM", v.get("lower_name", "Unknown"),
                                 v.get("lower_hex"))

        # Row 3: accessory chips
        y3 = y2 + 26
        self._paint_chips(p, QRectF(x0, y3, right - x0, 20), list(v.get("accessories") or []))

        # Row 4: posture / camera / match / age
        y4 = y3 + 26
        p.setFont(self._f_body)
        p.setPen(QColor(TEXT_SECONDARY))
        meta = "  |  ".join(t for t in (v.get("posture"), v.get("camera"), v.get("match_text")) if t)
        age = str(v.get("age_text") or "")
        fm = p.fontMetrics()
        age_w = fm.horizontalAdvance(age) + 4 if age else 0
        p.drawText(QRectF(x0, y4, right - x0 - age_w - 8, 18),
                   int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                   fm.elidedText(meta, Qt.TextElideMode.ElideRight, int(right - x0 - age_w - 8)))
        if age:
            p.setPen(QColor(TEXT_MUTED))
            p.drawText(QRectF(right - age_w, y4, age_w, 18),
                       int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), age)
        p.end()

    def _paint_silhouette(self, p: QPainter, rect: QRectF, color: QColor) -> None:
        path = QPainterPath()
        path.addRoundedRect(rect, 9, 9)
        grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        top = QColor(color)
        top.setAlphaF(0.18)
        grad.setColorAt(0.0, top)
        grad.setColorAt(1.0, QColor(6, 10, 26, 200))
        p.fillPath(path, grad)
        p.setPen(QPen(QColor(255, 255, 255, 36), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        fig = QColor(color)
        fig.setAlphaF(0.55)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fig)
        cx = rect.center().x()
        p.drawEllipse(QPointF(cx, rect.top() + rect.height() * 0.30), rect.width() * 0.16, rect.width() * 0.16)
        body = QRectF(cx - rect.width() * 0.28, rect.top() + rect.height() * 0.48, rect.width() * 0.56,
                      rect.height() * 0.42)
        p.drawRoundedRect(body, 10, 10)

    def _paint_swatch_field(self, p: QPainter, rect: QRectF, caption: str, name: str, hex_color: Optional[str]) -> None:
        p.setFont(self._f_badge)
        p.setPen(QColor(TEXT_MUTED))
        cap_w = p.fontMetrics().horizontalAdvance(caption) + 6
        p.drawText(QRectF(rect.left(), rect.top(), cap_w, rect.height()),
                   int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), caption)
        sw = QRectF(rect.left() + cap_w, rect.center().y() - 6.5, 13, 13)
        c = QColor(hex_color) if hex_color and QColor(hex_color).isValid() else None
        if c is not None:
            p.setBrush(c)
            p.setPen(QPen(QColor(255, 255, 255, 120), 1.0))
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(QColor(255, 255, 255, 110), 1.0)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
        p.drawRoundedRect(sw, 4, 4)
        p.setFont(self._f_body)
        p.setPen(QColor(TEXT_PRIMARY))
        tx = sw.right() + 6
        p.drawText(QRectF(tx, rect.top(), rect.right() - tx, rect.height()),
                   int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                   p.fontMetrics().elidedText(str(name), Qt.TextElideMode.ElideRight, int(rect.right() - tx)))

    def _paint_chips(self, p: QPainter, rect: QRectF, accessories: List[str]) -> None:
        p.setFont(self._f_small)
        fm = p.fontMetrics()
        x = rect.left()
        if not accessories:
            p.setPen(QColor(TEXT_MUTED))
            p.drawText(rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), "No accessories detected")
            return
        amber = QColor(ACCENT_AMBER)
        for i, name in enumerate(accessories):
            w = fm.horizontalAdvance(name) + 16
            remaining = len(accessories) - i
            if x + w > rect.right() and i > 0:
                more = f"+{remaining}"
                p.setPen(QColor(TEXT_SECONDARY))
                p.drawText(QRectF(x, rect.top(), 40, rect.height()),
                           int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), more)
                break
            chip = QRectF(x, rect.top(), w, rect.height())
            bg = QColor(amber)
            bg.setAlphaF(0.13)
            bd = QColor(amber)
            bd.setAlphaF(0.45)
            p.setPen(QPen(bd, 1.0))
            p.setBrush(bg)
            p.drawRoundedRect(chip, rect.height() / 2, rect.height() / 2)
            p.setPen(amber.lighter(125))
            p.drawText(chip, int(Qt.AlignmentFlag.AlignCenter), name)
            x += w + 6


class _FeedDelegate(QStyledItemDelegate):
    """Paints identity event rows: kind dot, uid (identity colour), text, time and camera."""

    ROW_H = 46

    def __init__(self, parent=None):
        super().__init__(parent)
        self._f_uid = ui_font(12, QFont.Weight.Bold)
        self._f_text = ui_font(12)
        self._f_meta = ui_font(11, mono=True)

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(200, self.ROW_H)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        entry = index.data(Qt.ItemDataRole.UserRole) or {}
        r = QRectF(option.rect).adjusted(2, 2, -2, -2)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if option.state & QStyle.StateFlag.State_MouseOver:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 12))
            painter.drawRoundedRect(r, 9, 9)
        kind_color = QColor(entry.get("kind_color") or ACCENT_SKY)
        center = QPointF(r.left() + 12, r.top() + 15)
        halo = QColor(kind_color)
        halo.setAlphaF(0.22)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(halo)
        painter.drawEllipse(center, 7, 7)
        painter.setBrush(kind_color)
        painter.drawEllipse(center, 3.5, 3.5)

        x = r.left() + 26
        painter.setFont(self._f_uid)
        uid = str(entry.get("uid", ""))
        uid_color = QColor(entry.get("uid_color") or ACCENT_CYAN)
        painter.setPen(uid_color.lighter(110))
        fm = painter.fontMetrics()
        uid_w = fm.horizontalAdvance(uid)
        painter.drawText(QRectF(x, r.top() + 4, uid_w + 2, 20),
                         int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), uid)
        painter.setFont(self._f_text)
        painter.setPen(QColor(TEXT_PRIMARY))
        tfm = painter.fontMetrics()
        tx = x + uid_w + 8
        painter.drawText(QRectF(tx, r.top() + 4, r.right() - tx - 4, 20),
                         int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                         tfm.elidedText(str(entry.get("text", "")), Qt.TextElideMode.ElideRight,
                                        int(r.right() - tx - 4)))
        painter.setFont(self._f_meta)
        painter.setPen(QColor(TEXT_MUTED))
        meta = str(entry.get("time", ""))
        if entry.get("camera"):
            meta += f"   {entry['camera']}"
        if entry.get("track_id") is not None:
            meta += f"   track #{entry['track_id']}"
        painter.drawText(QRectF(x, r.top() + 23, r.right() - x, 18),
                         int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), meta)
        painter.setPen(QPen(QColor(255, 255, 255, 12), 1.0))
        painter.drawLine(QPointF(x, r.bottom() + 1), QPointF(r.right() - 4, r.bottom() + 1))
        painter.restore()


class EventFeed(QListWidget):
    """Most-recent-first identity event list capped at FEED_CAP entries."""

    def __init__(self, parent: Optional[QWidget] = None, cap: int = FEED_CAP):
        super().__init__(parent)
        self._cap = int(cap)
        self.setObjectName("eventFeed")
        self.setItemDelegate(_FeedDelegate(self))
        self.setUniformItemSizes(True)
        self.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.setMouseTracking(True)
        self.setFrameShape(QListWidget.Shape.NoFrame)
        self.viewport().setAutoFillBackground(False)

    def cap(self) -> int:
        return self._cap

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.count() == 0:
            p = QPainter(self.viewport())
            p.setPen(QColor(TEXT_MUTED))
            p.setFont(ui_font(12))
            p.drawText(QRectF(self.viewport().rect()).adjusted(8, 8, -8, -8),
                       int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                       "New identities, re-identifications and merges are logged here.")
            p.end()

    def add_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Insert an event at the top; trims the oldest entries beyond the cap."""
        entry = format_identity_event(payload)
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, entry)
        item.setToolTip(f"{entry['time']}  {entry['camera']}  {entry['uid']}: {entry['text']}")
        self.insertItem(0, item)
        while self.count() > self._cap:
            self.takeItem(self.count() - 1)
        return entry

    def entries(self) -> List[Dict[str, Any]]:
        """Feed entries, most recent first."""
        return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]


class _CountPill(QWidget):
    """Small glass counter pill: caption + value."""

    def __init__(self, caption: str, color: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._caption = caption
        self._value = "0"
        self._color = QColor(color)
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._f_cap = ui_font(10, QFont.Weight.Bold)
        self._f_val = ui_font(13, QFont.Weight.Bold)

    def set_value(self, value: Any) -> None:
        self._value = str(value)
        self.setFixedWidth(self.sizeHint().width())
        self.update()

    def value(self) -> str:
        return self._value

    def sizeHint(self) -> QSize:
        w = QFontMetrics(self._f_cap).horizontalAdvance(self._caption) + \
            QFontMetrics(self._f_val).horizontalAdvance(self._value) + 30
        return QSize(max(70, w), 28)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        bg = QColor(self._color)
        bg.setAlphaF(0.10)
        bd = QColor(self._color)
        bd.setAlphaF(0.35)
        p.setPen(QPen(bd, 1.0))
        p.setBrush(bg)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.setFont(self._f_cap)
        p.setPen(QColor(TEXT_SECONDARY))
        cw = p.fontMetrics().horizontalAdvance(self._caption)
        p.drawText(QRectF(r.left() + 11, r.top(), cw + 2, r.height()), int(Qt.AlignmentFlag.AlignVCenter), self._caption)
        p.setFont(self._f_val)
        p.setPen(self._color.lighter(120))
        p.drawText(QRectF(r.left() + 11 + cw + 7, r.top(), r.width(), r.height()), int(Qt.AlignmentFlag.AlignVCenter),
                   self._value)
        p.end()


class LiveIdentityPanel(QWidget):
    """'Identities in view' cards keyed by person_uid + identity event feed (buffered, ~5 Hz repaint)."""

    identity_selected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None, clock: Callable[[], float] = time.monotonic,
                 auto_tick: bool = True):
        super().__init__(parent)
        self._clock = clock
        self._auto_tick = auto_tick
        self._entries: Dict[str, _IdentityEntry] = {}
        self._cards: Dict[str, IdentityCard] = {}
        self._session_uids: set = set()
        self._external_session_count: Optional[int] = None
        self._pending_badges: Dict[str, Tuple[str, float]] = {}
        self._order: List[str] = []
        self._order_dirty = False
        self._thumb_cache: "OrderedDict[Tuple[str, float], QPixmap]" = OrderedDict()
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self.tick)
        self._init_ui()

    # ---- UI --------------------------------------------------------------------------------------
    def _init_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.panel = GlassPanel(radius=18)
        outer.addWidget(self.panel)
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(14, 12, 10, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(6)
        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        title_col.addWidget(set_role(QLabel("IDENTITIES IN VIEW"), "section"))
        title_col.addWidget(set_role(QLabel("Same person, same UID on re-entry"), "hint"))
        header.addLayout(title_col)
        header.addStretch()
        self.pill_in_view = _CountPill("IN VIEW", ACCENT_CYAN)
        self.pill_session = _CountPill("SESSION", ACCENT_VIOLET)
        header.addWidget(self.pill_in_view)
        header.addWidget(self.pill_session)
        layout.addLayout(header)

        # Cards
        self.cards_scroll = QScrollArea()
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cards_container = QWidget()
        self.cards_container.setObjectName("liveCardsContainer")
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 2, 4, 2)
        self.cards_layout.setSpacing(4)
        self.lbl_empty = set_role(QLabel("No identities in view yet.\nStart a stream: people appear here once "
                                         "their identity is established."), "placeholder")
        self.lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_empty.setWordWrap(True)
        self.lbl_empty.setMinimumHeight(120)
        self.cards_layout.addWidget(self.lbl_empty)
        self.cards_layout.addStretch()
        self.cards_scroll.setWidget(self.cards_container)
        make_transparent_scroll(self.cards_scroll)
        self.cards_container.setObjectName("liveCardsContainer")
        layout.addWidget(self.cards_scroll, stretch=3)

        # Divider + feed
        feed_header = QHBoxLayout()
        feed_header.addWidget(set_role(QLabel("IDENTITY EVENTS"), "section"))
        feed_header.addStretch()
        self.lbl_feed_count = set_role(QLabel("0 events"), "hint")
        feed_header.addWidget(self.lbl_feed_count)
        layout.addSpacing(2)
        layout.addLayout(feed_header)
        self.feed = EventFeed()
        layout.addWidget(self.feed, stretch=2)
        self._update_counters()

    # ---- public API ------------------------------------------------------------------------------
    def on_signature(self, payload: Dict[str, Any]) -> None:
        """Buffer a signature_detected payload (repaint happens on the next tick)."""
        if not isinstance(payload, dict):
            return
        uid = str(payload.get("person_uid") or "").strip()
        status = str(payload.get("status") or "confirmed").lower()
        if not uid or status == "tentative":
            return
        now = self._clock()
        entry = self._entries.get(uid)
        if entry is None:
            entry = _IdentityEntry(uid=uid, first_seen=now, last_update=now)
            self._entries[uid] = entry
            self._order_dirty = True
            pending = self._pending_badges.pop(uid, None)
            if pending and now - pending[1] <= BADGE_HOLD_S:
                entry.badge, entry.badge_until = pending[0], pending[1] + BADGE_HOLD_S
        entry.payload = {k: v for k, v in payload.items() if k != "embedding"}
        entry.last_update = now
        if status in ("new", "reacquired") and status != entry.last_status:
            entry.badge, entry.badge_until = status, now + BADGE_HOLD_S
        entry.last_status = status
        entry.dirty = True
        self._session_uids.add(uid)
        self._ensure_timer()

    def on_identity_event(self, payload: Dict[str, Any]) -> None:
        """Append an identity_event to the feed and flag the matching card."""
        if not isinstance(payload, dict):
            return
        self.feed.add_event(payload)
        kind = str(payload.get("type") or "").lower()
        uid = str(payload.get("person_uid") or "").strip()
        now = self._clock()
        if kind == "merged":
            src = str(payload.get("merged_from") or "").strip()
            if src and src != uid:
                self._remove(src)
                self._session_uids.discard(src)
        if uid and kind in ("new", "reacquired", "merged"):
            badge = "new" if kind == "new" else "reacquired"
            entry = self._entries.get(uid)
            if entry is not None:
                entry.badge, entry.badge_until = badge, now + BADGE_HOLD_S
                entry.dirty = True
            else:
                self._pending_badges[uid] = (badge, now)
            self._session_uids.add(uid)
        self._update_counters()
        self._ensure_timer()

    def set_session_identities(self, count: int) -> None:
        """Authoritative session identity count from the worker (stats_updated)."""
        self._external_session_count = max(0, int(count))
        self._update_counters()

    def clear(self) -> None:
        """Remove all cards, feed entries and counters (e.g. after a database reset)."""
        for uid in list(self._entries.keys()):
            self._remove(uid)
        self._entries.clear()
        self._pending_badges.clear()
        self._session_uids.clear()
        self._external_session_count = None
        self.feed.clear()
        self._update_counters()

    def tick(self, now: Optional[float] = None) -> None:
        """Apply buffered updates: create/update/age/remove cards (called by the ~5 Hz timer)."""
        now = self._clock() if now is None else float(now)
        for uid in [u for u, e in self._entries.items() if now - e.last_update > REMOVE_AFTER_S]:
            self._remove(uid)

        for uid, entry in self._entries.items():
            stale = (now - entry.last_update) > STALE_AFTER_S
            if stale != entry.stale:
                entry.stale = stale
                entry.dirty = True
                self._order_dirty = True
            if entry.badge in ("new", "reacquired") and now > entry.badge_until:
                entry.badge = "confirmed"
                entry.dirty = True
            if now - entry.thumb_checked >= THUMB_REFRESH_S:
                entry.thumb_checked = now
                if self._refresh_thumbnail(entry):
                    entry.dirty = True
            card = self._cards.get(uid)
            if card is None:
                card = IdentityCard(uid)
                card.clicked.connect(self.identity_selected.emit)
                self._cards[uid] = card
                self._order_dirty = True
                entry.dirty = True
            if entry.dirty or entry.stale:
                card.set_view(self._view_for(entry, now))
                entry.dirty = False

        if self._order_dirty:
            self._reorder()
        self._update_counters()
        if not self._entries and self._timer.isActive():
            self._timer.stop()

    def card_for(self, uid: str) -> Optional[IdentityCard]:
        return self._cards.get(uid)

    def visible_uids(self) -> List[str]:
        """Card order as displayed (active newest first, then recently seen)."""
        return list(self._order)

    def in_view_count(self) -> int:
        return sum(1 for e in self._entries.values() if not e.stale)

    def session_count(self) -> int:
        if self._external_session_count is not None:
            return self._external_session_count
        return len(self._session_uids)

    def entry_state(self, uid: str) -> Optional[Dict[str, Any]]:
        """Introspection helper (tests): badge / stale flags for an identity."""
        e = self._entries.get(uid)
        if e is None:
            return None
        return {"badge": e.badge, "stale": e.stale, "last_update": e.last_update, "first_seen": e.first_seen}

    # ---- internals -------------------------------------------------------------------------------
    def _ensure_timer(self) -> None:
        if self._auto_tick and not self._timer.isActive():
            self._timer.start()

    def _remove(self, uid: str) -> None:
        self._entries.pop(uid, None)
        card = self._cards.pop(uid, None)
        if card is not None:
            self.cards_layout.removeWidget(card)
            card.hide()
            card.deleteLater()
        if uid in self._order:
            self._order.remove(uid)
            self._order_dirty = True

    def _reorder(self) -> None:
        active = sorted((e for e in self._entries.values() if not e.stale), key=lambda e: -e.first_seen)
        stale = sorted((e for e in self._entries.values() if e.stale), key=lambda e: -e.last_update)
        order = [e.uid for e in active + stale if e.uid in self._cards]
        for uid in order:
            self.cards_layout.removeWidget(self._cards[uid])
        for i, uid in enumerate(order):
            self.cards_layout.insertWidget(1 + i, self._cards[uid])
            self._cards[uid].show()
        self._order = order
        self.lbl_empty.setVisible(not order)
        self._order_dirty = False

    def _refresh_thumbnail(self, entry: _IdentityEntry) -> bool:
        path = entry.payload.get("crop_path")
        if not path or not os.path.exists(str(path)):
            return False
        try:
            mtime = os.path.getmtime(str(path))
        except OSError:
            return False
        key = (str(path), mtime)
        if key == entry.thumb_key:
            return False
        pix = self._thumb_cache.get(key)
        if pix is None:
            dpr = max(1.0, float(self.devicePixelRatioF()))
            pix = load_rounded_thumbnail(str(path), IdentityCard.THUMB_W, IdentityCard.THUMB_H, 9.0, dpr)
            if pix is None:
                return False
            self._thumb_cache[key] = pix
            while len(self._thumb_cache) > 96:
                self._thumb_cache.popitem(last=False)
        entry.thumb_key, entry.thumb = key, pix
        return True

    def _view_for(self, entry: _IdentityEntry, now: float) -> Dict[str, Any]:
        pl = entry.payload
        upper = pl.get("upper_garment") or {}
        lower = pl.get("lower_garment") or {}
        lower_name = str(lower.get("color") or "Unknown")
        lower_type = str(lower.get("type") or "")
        if lower_type and lower_type != lower_name and lower_type.lower().startswith("bare"):
            lower_name = f"{lower_name} (shorts)"
        if entry.stale:
            badge = "stale"
        else:
            badge = entry.badge if entry.badge in STATUS_BADGES else "confirmed"
        badge_text, badge_color = STATUS_BADGES[badge]
        age = now - entry.last_update
        if entry.stale:
            age_text = f"seen {int(age)}s ago"
        else:
            age_text = "live"
        match = pl.get("match_score")
        match_text = ""
        if match is not None:
            try:
                match_text = f"match {float(match) * 100:.0f}%"
            except (TypeError, ValueError):
                match_text = ""
        accessories = [accessory_short_label(a) for a in (pl.get("accessories") or []) if a]
        return {
            "uid": entry.uid,
            "label": pl.get("display_label") or entry.uid,
            "color": str(pl.get("identity_color") or identity_color_hex(entry.uid)),
            "track_id": pl.get("track_id"),
            "badge_text": badge_text,
            "badge_color": badge_color,
            "upper_name": str(upper.get("color") or "Unknown"),
            "upper_hex": upper.get("hex"),
            "lower_name": lower_name,
            "lower_hex": lower.get("hex"),
            "accessories": accessories,
            "posture": str(pl.get("posture") or ""),
            "camera": str(pl.get("camera_id") or ""),
            "match_text": match_text,
            "age_text": age_text,
            "stale": entry.stale,
            "thumb": entry.thumb,
        }

    def _update_counters(self) -> None:
        self.pill_in_view.set_value(self.in_view_count())
        self.pill_session.set_value(self.session_count())
        n = self.feed.count()
        self.lbl_feed_count.setText(f"{n} event{'s' if n != 1 else ''}")
