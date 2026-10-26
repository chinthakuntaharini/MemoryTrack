"""
Forensic Gallery & Match Cards Widget
Renders ranked candidate glass cards (identity accent, colour swatches, accessory chips, score bar) and inspection dialogs.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QStackedLayout, QTextEdit, QVBoxLayout, QWidget
)

from forensic_app.ui.effects import (
    ACCENT_AMBER, ACCENT_CYAN, ACCENT_VIOLET, GlassPanel, HoverProgress, apply_hover_glow, chip_label,
    load_rounded_thumbnail, make_transparent_scroll, paint_glass, set_role, set_variant, swatch_pixmap, ui_font
)
from forensic_app.utils.drawing import accessory_short_label, identity_color_hex

logger = logging.getLogger(__name__)


class ScoreBar(QWidget):
    """Slim gradient match-score bar with a soft glow on the filled part."""

    def __init__(self, value: float = 0.0, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._value = max(0.0, min(1.0, float(value)))
        self.setFixedHeight(10)
        self.setMinimumWidth(96)

    def value(self) -> float:
        return self._value

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 2, -1, -2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 18))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        if self._value > 0:
            fill = QRectF(r.left(), r.top(), max(r.height(), r.width() * self._value), r.height())
            glow = QColor(ACCENT_CYAN)
            glow.setAlphaF(0.22)
            p.setBrush(glow)
            p.drawRoundedRect(fill.adjusted(-1, -2, 1, 2), (r.height() + 4) / 2, (r.height() + 4) / 2)
            grad = QLinearGradient(fill.topLeft(), fill.topRight())
            grad.setColorAt(0.0, QColor(ACCENT_CYAN))
            grad.setColorAt(1.0, QColor(ACCENT_VIOLET))
            p.setBrush(grad)
            p.drawRoundedRect(fill, r.height() / 2, r.height() / 2)
        p.end()


class CandidateCard(QFrame):
    """Forensic candidate match card component (glass, identity accent bar, hover glow)."""

    THUMB_W, THUMB_H = 72, 108

    def __init__(self, data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.data = data
        uid = str(self.data.get("person_uid") or "UID-UNKNOWN")
        self.identity_color = str(self.data.get("identity_color") or identity_color_hex(uid))
        self._hover = HoverProgress(self, 200)
        self._init_ui()

    def hover_value(self) -> float:
        return self._hover.value

    def _init_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumHeight(self.THUMB_H + 24)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 16, 12)
        layout.setSpacing(14)

        # 1. Thumbnail Crop
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(self.THUMB_W, self.THUMB_H)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crop_path = self.data.get("crop_path")
        pix = None
        if crop_path and os.path.exists(str(crop_path)):
            dpr = max(1.0, float(self.devicePixelRatioF()))
            pix = load_rounded_thumbnail(str(crop_path), self.THUMB_W, self.THUMB_H, 10.0, dpr)
        if pix is not None:
            self.thumb_label.setPixmap(pix)
        else:
            self.thumb_label.setText("No Crop")
            self.thumb_label.setStyleSheet(
                "QLabel { background-color: rgba(6,10,26,0.65); border: 1px dashed rgba(255,255,255,0.16);"
                " border-radius: 10px; color: #64748b; font-size: 11px; }")
        layout.addWidget(self.thumb_label)

        # 2. Metadata & Attribute Badges
        info_layout = QVBoxLayout()
        info_layout.setSpacing(6)

        uid = self.data.get("person_uid", "UID-UNKNOWN")
        cam = self.data.get("camera_id", "CAM_01")
        time_str = str(self.data.get("timestamp", "N/A") or "N/A").replace("T", "  ").replace("Z", "")

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_lbl = QLabel(str(uid))
        title_lbl.setFont(ui_font(15, QFont.Weight.Bold))
        title_lbl.setStyleSheet(f"color: {QColor(self.identity_color).lighter(110).name()};")
        title_row.addWidget(title_lbl)
        title_row.addWidget(chip_label(str(cam), ACCENT_CYAN))
        title_row.addStretch()
        info_layout.addLayout(title_row)

        time_lbl = set_role(QLabel(f"Last sighted  {time_str}"), "muted")
        info_layout.addWidget(time_lbl)

        upper = self.data.get("upper_color", "Unknown")
        lower = self.data.get("lower_color", "Not Detected")
        swatch_row = QHBoxLayout()
        swatch_row.setSpacing(6)
        for caption, name, hex_c in (("TOP", upper, self.data.get("upper_hex")),
                                     ("BOTTOM", lower, self.data.get("lower_hex"))):
            swatch_row.addWidget(set_role(QLabel(caption), "caption"))
            sw = QLabel()
            sw.setPixmap(swatch_pixmap(hex_c, 14))
            sw.setFixedSize(16, 16)
            swatch_row.addWidget(sw)
            name_lbl = QLabel(str(name))
            name_lbl.setStyleSheet("color: #e6edf7; font-size: 12px; font-weight: 600;")
            swatch_row.addWidget(name_lbl)
            swatch_row.addSpacing(10)
        swatch_row.addStretch()
        info_layout.addLayout(swatch_row)

        chips_row = QHBoxLayout()
        chips_row.setSpacing(5)
        accs = [a for a in (self.data.get("accessories") or []) if a]
        if accs:
            for a in accs[:5]:
                chips_row.addWidget(chip_label(accessory_short_label(a), ACCENT_AMBER))
            if len(accs) > 5:
                chips_row.addWidget(chip_label(f"+{len(accs) - 5}", muted=True))
        else:
            chips_row.addWidget(chip_label("No accessories", muted=True))
        posture = self.data.get("posture", "Standing")
        if posture:
            chips_row.addWidget(chip_label(str(posture), ACCENT_VIOLET))
        chips_row.addStretch()
        info_layout.addLayout(chips_row)
        layout.addLayout(info_layout, stretch=1)

        # 3. Match Confidence & Inspection Button
        score_layout = QVBoxLayout()
        score_layout.setSpacing(6)
        score = float(self.data.get("similarity_score", 0.5) or 0.0)
        score_lbl = QLabel(f"{score * 100.0:.1f}%")
        score_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        score_lbl.setFont(ui_font(20, QFont.Weight.Bold))
        score_lbl.setStyleSheet("color: #f8fafc;")
        cap = set_role(QLabel("MATCH SCORE"), "caption")
        cap.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.score_bar = ScoreBar(score)

        btn_inspect = set_variant(QPushButton("Inspect Traits"), "ghost")
        btn_inspect.setProperty("size", "small")
        btn_inspect.clicked.connect(self._inspect_details)
        apply_hover_glow(btn_inspect)

        score_layout.addStretch()
        score_layout.addWidget(cap)
        score_layout.addWidget(score_lbl)
        score_layout.addWidget(self.score_bar)
        score_layout.addSpacing(4)
        score_layout.addWidget(btn_inspect)
        score_layout.addStretch()
        layout.addLayout(score_layout)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(self.identity_color)
        hover = self._hover.value
        rect = QRectF(self.rect()).adjusted(4.5, 3.5, -4.5, -3.5)
        radius = 14.0
        if hover > 0.01:
            for i in range(4, 0, -1):
                c = QColor(color)
                c.setAlphaF(0.22 * hover * (1.0 - i / 5.0))
                p.setPen(QPen(c, 2.0))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(rect.adjusted(-i, -i, i, i), radius + i, radius + i)
        border = QColor(color) if hover > 0.01 else QColor(255, 255, 255)
        border.setAlphaF(0.55 * hover + 0.14 * (1 - hover))
        paint_glass(p, self, rect, radius, tint=0.05 + 0.03 * hover, accent=color.name(), border_color=border)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(QRectF(rect.left() + 1.5, rect.top() + 16, 3.0, rect.height() - 32), 1.5, 1.5)
        p.end()

    def _inspect_details(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Forensic Signature Analysis - {self.data.get('person_uid')}")
        dialog.setMinimumSize(560, 440)
        d_layout = QVBoxLayout(dialog)
        d_layout.setContentsMargins(16, 16, 16, 16)
        d_layout.setSpacing(10)
        head = QLabel(str(self.data.get("person_uid", "")))
        head.setFont(ui_font(18, QFont.Weight.Bold))
        head.setStyleSheet(f"color: {self.identity_color};")
        d_layout.addWidget(head)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setFont(ui_font(12, mono=True))
        shown = {}
        for k, v in self.data.items():
            if k == "embedding" and isinstance(v, (list, tuple)):
                shown[k] = f"<{len(v)}-D appearance vector>"
            else:
                shown[k] = v
        txt.setText(json.dumps(shown, indent=2, default=str))
        d_layout.addWidget(txt)

        btn_close = set_variant(QPushButton("Close"), "primary")
        btn_close.clicked.connect(dialog.accept)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(btn_close)
        d_layout.addLayout(row)
        dialog.exec()


class GalleryWidget(QWidget):
    """Ranked forensic search results gallery."""

    PLACEHOLDER_TEXT = ("Build a query on the left and run the search.\n"
                        "Ranked candidates with clothing colours and accessories appear here.")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.panel = GlassPanel(radius=18)
        outer.addWidget(self.panel)
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(16, 14, 12, 12)
        layout.setSpacing(10)

        # Header bar
        header_layout = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title_col.addWidget(set_role(QLabel("CANDIDATE MATCHES"), "section"))
        self.lbl_title = set_role(QLabel("Candidate Matches: Ready for Query"), "muted")
        title_col.addWidget(self.lbl_title)
        header_layout.addLayout(title_col)
        header_layout.addStretch()

        self.btn_clear_view = set_variant(QPushButton("Clear Results"), "ghost")
        self.btn_clear_view.setProperty("size", "small")
        self.btn_clear_view.clicked.connect(self.clear_cards)
        header_layout.addWidget(self.btn_clear_view)
        layout.addLayout(header_layout)

        # Stack: placeholder vs scrollable cards
        self.stack_host = QWidget()
        self.stack = QStackedLayout(self.stack_host)
        self.placeholder = set_role(QLabel(self.PLACEHOLDER_TEXT), "placeholder")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.stack.addWidget(self.placeholder)

        # Scrollable Cards Container
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 6, 0)
        self.cards_layout.setSpacing(6)
        self.cards_layout.addStretch()
        self.scroll_area.setWidget(self.cards_container)
        make_transparent_scroll(self.scroll_area)
        self.cards_container.setObjectName("cardsContainer")
        self.stack.addWidget(self.scroll_area)
        layout.addWidget(self.stack_host, stretch=1)

    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """Populate gallery with ranked candidate match cards."""
        self.clear_cards()
        n = len(results)
        self.lbl_title.setText(f"Candidate Matches: {n} Person{'s' if n != 1 else ''} Found")

        for item in results:
            card = CandidateCard(item)
            # Insert before trailing stretch
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
        if n:
            self.stack.setCurrentWidget(self.scroll_area)
        else:
            self.placeholder.setText("No candidates matched these attributes.\nTry fewer accessories or 'Any' colour.")
            self.stack.setCurrentWidget(self.placeholder)

    def clear_cards(self) -> None:
        """Remove all cards from the gallery."""
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.lbl_title.setText("Candidate Matches: 0")
        self.placeholder.setText(self.PLACEHOLDER_TEXT)
        self.stack.setCurrentWidget(self.placeholder)
