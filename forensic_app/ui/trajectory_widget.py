"""
Multi-Camera Trajectory & Timeline Tracking Widget
Visualizes the spatio-temporal route of a person as camera-hop chips plus a detailed sightings table.
"""

import contextlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QPointF, QRectF, QSize, QStringListModel, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QCompleter, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QScrollArea, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget
)

from forensic_app.ui.effects import (
    ACCENT_AMBER, ACCENT_CYAN, ACCENT_EMERALD, ACCENT_ROSE, ACCENT_SKY, ACCENT_VIOLET, TEXT_MUTED, TEXT_PRIMARY,
    TEXT_SECONDARY, GlassPanel, make_transparent_scroll, paint_glass, set_role, set_variant, swatch_icon, ui_font
)
from forensic_app.utils.drawing import accessory_short_label, identity_color_hex

logger = logging.getLogger(__name__)

_CAMERA_COLORS = (ACCENT_CYAN, ACCENT_VIOLET, ACCENT_EMERALD, ACCENT_AMBER, ACCENT_ROSE, ACCENT_SKY)


def camera_color(camera_id: str) -> str:
    """Stable accent colour per camera node (by its number)."""
    m = re.search(r"(\d+)", camera_id or "")
    idx = (int(m.group(1)) - 1) if m else sum(map(ord, camera_id or "x"))
    return _CAMERA_COLORS[idx % len(_CAMERA_COLORS)]


def short_time(ts: Any) -> str:
    """'2026-10-26T12:01:03Z' -> '12:01:03' (passes other strings through)."""
    text = str(ts or "")
    m = re.search(r"(\d{2}:\d{2}:\d{2})", text)
    return m.group(1) if m else text


def compress_camera_hops(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse consecutive sightings on the same camera into hop segments (pure; unit-testable)."""
    segments: List[Dict[str, Any]] = []
    for r in rows:
        cam = str(r.get("camera_id") or "?")
        ts = r.get("timestamp")
        if segments and segments[-1]["camera"] == cam:
            segments[-1]["end"] = ts
            segments[-1]["count"] += 1
        else:
            segments.append({"camera": cam, "start": ts, "end": ts, "count": 1})
    return segments


class CameraHopTimeline(QWidget):
    """Compact horizontal camera-hop timeline: one glass chip per consecutive camera segment."""

    CHIP_W, CHIP_H, GAP = 170, 70, 34

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.segments: List[Dict[str, Any]] = []
        self._f_cam = ui_font(13, QFont.Weight.Bold)
        self._f_time = ui_font(11, mono=True)
        self._f_meta = ui_font(11)
        self.setMinimumHeight(self.CHIP_H + 12)

    def set_segments(self, segments: List[Dict[str, Any]]) -> None:
        self.segments = list(segments)
        self.updateGeometry()
        self.setMinimumWidth(self._content_width())
        self.update()

    def _content_width(self) -> int:
        n = len(self.segments)
        return 0 if n == 0 else int(n * self.CHIP_W + (n - 1) * self.GAP + 12)

    def sizeHint(self) -> QSize:
        return QSize(max(300, self._content_width()), self.CHIP_H + 12)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.segments:
            p.setPen(QColor(TEXT_MUTED))
            p.setFont(self._f_meta)
            p.drawText(QRectF(self.rect()), int(Qt.AlignmentFlag.AlignCenter),
                       "Camera hops appear here once a route is traced.")
            p.end()
            return
        y = (self.height() - self.CHIP_H) / 2.0
        x = 6.0
        for i, seg in enumerate(self.segments):
            color = QColor(camera_color(seg["camera"]))
            rect = QRectF(x, y, self.CHIP_W, self.CHIP_H)
            border = QColor(color)
            border.setAlphaF(0.55)
            paint_glass(p, self, rect, 12.0, tint=0.06, accent=color.name(), border_color=border, highlight=0.2)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawEllipse(QPointF(rect.left() + 14, rect.top() + 17), 4, 4)
            p.setFont(self._f_cam)
            p.setPen(color.lighter(120))
            p.drawText(QRectF(rect.left() + 24, rect.top() + 6, rect.width() - 30, 22),
                       int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), seg["camera"])
            p.setFont(self._f_meta)
            p.setPen(QColor(TEXT_MUTED))
            p.drawText(QRectF(rect.left() + 10, rect.top() + 6, rect.width() - 20, 22),
                       int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), f"hop {i + 1}")
            p.setFont(self._f_time)
            p.setPen(QColor(TEXT_PRIMARY))
            start, end = short_time(seg["start"]), short_time(seg["end"])
            t_text = start if start == end else f"{start} - {end}"
            p.drawText(QRectF(rect.left() + 12, rect.top() + 28, rect.width() - 20, 18),
                       int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), t_text)
            p.setFont(self._f_meta)
            p.setPen(QColor(TEXT_SECONDARY))
            n = seg["count"]
            p.drawText(QRectF(rect.left() + 12, rect.top() + 46, rect.width() - 20, 18),
                       int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                       f"{n} sighting{'s' if n != 1 else ''}")
            if i < len(self.segments) - 1:
                ax1, ax2 = rect.right() + 6, rect.right() + self.GAP - 6
                ay = rect.center().y()
                nxt = QColor(camera_color(self.segments[i + 1]["camera"]))
                pen = QPen(nxt, 1.6)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawLine(QPointF(ax1, ay), QPointF(ax2, ay))
                head = QPainterPath()
                head.moveTo(ax2 - 6, ay - 4)
                head.lineTo(ax2, ay)
                head.lineTo(ax2 - 6, ay + 4)
                p.drawPath(head)
            x += self.CHIP_W + self.GAP
        p.end()


class TrajectoryWidget(QWidget):
    """Spatio-temporal camera timeline and route history table."""

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self._init_ui()

    def _db_lock(self):
        lock = getattr(self.db, "_lock", None)
        return lock if lock is not None else contextlib.nullcontext()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Top query bar
        bar = GlassPanel(radius=16)
        top_layout = QHBoxLayout(bar)
        top_layout.setContentsMargins(16, 12, 16, 12)
        top_layout.setSpacing(10)
        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title_col.addWidget(set_role(QLabel("TRAJECTORY TIMELINE"), "section"))
        title_col.addWidget(set_role(QLabel("Trace a person's route across camera nodes"), "hint"))
        top_layout.addLayout(title_col)
        top_layout.addSpacing(18)
        top_layout.addWidget(set_role(QLabel("Target Person UID"), "field"))
        self.input_uid = QLineEdit()
        self.input_uid.setPlaceholderText("e.g. UID-0001")
        self.input_uid.setMinimumWidth(220)
        self.input_uid.setClearButtonEnabled(True)
        self.input_uid.returnPressed.connect(self.load_trajectory)
        self._completer_model = QStringListModel(self)
        completer = QCompleter(self._completer_model, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.input_uid.setCompleter(completer)
        top_layout.addWidget(self.input_uid)

        self.btn_load_trajectory = set_variant(QPushButton("Track Multi-Camera Route"), "primary")
        self.btn_load_trajectory.clicked.connect(self.load_trajectory)
        top_layout.addWidget(self.btn_load_trajectory)
        top_layout.addStretch()
        layout.addWidget(bar)

        # Hop timeline + table
        body = GlassPanel(radius=18)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(16, 14, 16, 14)
        body_layout.setSpacing(10)
        self.lbl_summary = set_role(QLabel("No route loaded"), "muted")
        body_layout.addWidget(self.lbl_summary)

        self.timeline = CameraHopTimeline()
        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.timeline_scroll.setFixedHeight(self.timeline.CHIP_H + 26)
        self.timeline_scroll.setWidget(self.timeline)
        make_transparent_scroll(self.timeline_scroll)
        self.timeline.setObjectName("timelineContainer")
        body_layout.addWidget(self.timeline_scroll)

        # Timeline Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Step", "Camera Node", "Timestamp", "Upper Clothing", "Lower Clothing", "Carried Items", "Posture"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        body_layout.addWidget(self.table, stretch=1)
        layout.addWidget(body, stretch=1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_uid_suggestions()

    def refresh_uid_suggestions(self) -> None:
        """Populate the UID completer from the persons table."""
        if not self.db:
            return
        try:
            with self._db_lock():
                cur = self.db.conn.cursor()
                cur.execute("SELECT person_uid FROM persons ORDER BY last_seen DESC LIMIT 500;")
                uids = [str(r[0]) for r in cur.fetchall() if r[0]]
            self._completer_model.setStringList(uids)
        except Exception as e:
            logger.debug(f"UID suggestions unavailable: {e}")

    def set_uid(self, uid: str) -> None:
        """Programmatically trace a uid (e.g. from a live identity card)."""
        self.input_uid.setText(uid)
        self.load_trajectory()

    def _fetch_rows(self, uid: str) -> List[Dict[str, Any]]:
        with self._db_lock():
            cursor = self.db.conn.cursor()
            try:
                cursor.execute("""
                    SELECT camera_id, timestamp, upper_color, upper_hex, lower_color, lower_hex,
                           accessories_json, posture
                    FROM signatures
                    WHERE person_uid = ?
                    ORDER BY sig_id ASC
                """, (uid,))
            except Exception:
                cursor.execute("""
                    SELECT camera_id, timestamp, upper_color, lower_color, accessories_json, posture
                    FROM signatures
                    WHERE person_uid = ?
                    ORDER BY sig_id ASC
                """, (uid,))
            return [dict(r) for r in cursor.fetchall()]

    def load_trajectory(self) -> None:
        """Fetch all temporal sightings for the queried UID."""
        target_uid = self.input_uid.text().strip()
        if not target_uid or not self.db:
            return

        try:
            rows = self._fetch_rows(target_uid)
            if not rows and target_uid.upper() != target_uid:
                rows = self._fetch_rows(target_uid.upper())
                target_uid = target_uid.upper() if rows else target_uid
        except Exception as e:
            logger.error(f"Trajectory query failed for {target_uid}: {e}")
            rows = []

        self.table.setRowCount(len(rows))
        for idx, r in enumerate(rows):
            accs = []
            try:
                accs = json.loads(r.get("accessories_json") or "[]")
            except Exception:
                accs = []
            self.table.setItem(idx, 0, QTableWidgetItem(f"#{idx+1}"))
            cam = str(r.get("camera_id") or "")
            cam_item = QTableWidgetItem(cam)
            cam_item.setForeground(QColor(camera_color(cam)).lighter(115))
            self.table.setItem(idx, 1, cam_item)
            self.table.setItem(idx, 2, QTableWidgetItem(str(r.get("timestamp") or "")))
            up_item = QTableWidgetItem(str(r.get("upper_color") or "Unknown"))
            up_item.setIcon(swatch_icon(r.get("upper_hex")))
            self.table.setItem(idx, 3, up_item)
            lo_item = QTableWidgetItem(str(r.get("lower_color") or "None"))
            lo_item.setIcon(swatch_icon(r.get("lower_hex")))
            self.table.setItem(idx, 4, lo_item)
            acc_text = ", ".join(accessory_short_label(a) for a in accs) if accs else "None"
            self.table.setItem(idx, 5, QTableWidgetItem(acc_text))
            self.table.setItem(idx, 6, QTableWidgetItem(str(r.get("posture") or "Standing")))

        segments = compress_camera_hops(rows)
        self.timeline.set_segments(segments)
        if rows:
            cams = len({s["camera"] for s in segments})
            color = identity_color_hex(target_uid)
            self.lbl_summary.setText(
                f"<span style='color:{color}; font-weight:700;'>{target_uid}</span>"
                f"  |  {len(rows)} sighting{'s' if len(rows) != 1 else ''}"
                f"  |  {cams} camera{'s' if cams != 1 else ''}  |  {len(segments)} hop{'s' if len(segments) != 1 else ''}"
                f"  |  first {short_time(rows[0].get('timestamp'))}  |  last {short_time(rows[-1].get('timestamp'))}")
        else:
            self.lbl_summary.setText(f"No sightings recorded for {target_uid}")
