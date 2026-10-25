"""
Forensic Gallery & Match Cards Widget
Renders ranked candidate match cards with similarity scores, trait badges, and inspection dialogs.
"""

import os
from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QPushButton, QLabel, QFrame, QDialog, QTextEdit,
    QProgressBar, QGridLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage


class CandidateCard(QFrame):
    """Forensic candidate match card component."""

    def __init__(self, data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.data = data
        self._init_ui()

    def _init_ui(self) -> None:
        self.setStyleSheet("""
            CandidateCard {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 8px;
                padding: 6px;
            }
            CandidateCard:hover {
                border: 1px solid #38bdf8;
                background-color: #162032;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # 1. Thumbnail Crop
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(70, 105)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setStyleSheet("background-color: #090d16; border-radius: 4px; border: 1px solid #334155;")

        crop_path = self.data.get("crop_path")
        if crop_path and os.path.exists(crop_path):
            pix = QPixmap(crop_path).scaled(70, 105, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.thumb_label.setPixmap(pix)
        else:
            self.thumb_label.setText("No Crop")

        layout.addWidget(self.thumb_label)

        # 2. Metadata & Attribute Badges
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        uid = self.data.get("person_uid", "UID-UNKNOWN")
        cam = self.data.get("camera_id", "CAM_01")
        time_str = self.data.get("timestamp", "N/A")

        title_lbl = QLabel(f"<b>{uid}</b> | {cam}")
        title_lbl.setStyleSheet("color: #38bdf8; font-size: 13px;")
        time_lbl = QLabel(f"Sighted: {time_str}")
        time_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")

        upper = self.data.get("upper_color", "Unknown")
        lower = self.data.get("lower_color", "Not Detected")
        accs = self.data.get("accessories", [])
        acc_str = ", ".join(accs) if accs else "None"
        posture = self.data.get("posture", "Standing")

        badge_text = f"Top: <font color='#38bdf8'><b>{upper}</b></font> | Bottom: <font color='#a78bfa'><b>{lower}</b></font><br>Accessories: <b>{acc_str}</b> | Posture: <b>{posture}</b>"
        badge_lbl = QLabel(badge_text)
        badge_lbl.setStyleSheet("font-size: 11px; color: #cbd5e1;")

        info_layout.addWidget(title_lbl)
        info_layout.addWidget(time_lbl)
        info_layout.addWidget(badge_lbl)
        layout.addLayout(info_layout, stretch=1)

        # 3. Match Confidence & Inspection Button
        score_layout = QVBoxLayout()
        score = float(self.data.get("similarity_score", 0.5)) * 100.0

        score_lbl = QLabel(f"<b>{score:.1f}% Match</b>")
        score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_lbl.setStyleSheet("color: #38bdf8; font-size: 14px;")

        pbar = QProgressBar()
        pbar.setFixedSize(90, 8)
        pbar.setTextVisible(False)
        pbar.setRange(0, 100)
        pbar.setValue(int(score))
        pbar.setStyleSheet("""
            QProgressBar { background-color: #1e293b; border-radius: 4px; }
            QProgressBar::chunk { background-color: #0284c7; border-radius: 4px; }
        """)

        btn_inspect = QPushButton("Inspect Traits")
        btn_inspect.setStyleSheet("padding: 4px 8px; font-size: 11px; background-color: #334155;")
        btn_inspect.clicked.connect(self._inspect_details)

        score_layout.addWidget(score_lbl)
        score_layout.addWidget(pbar)
        score_layout.addWidget(btn_inspect)
        score_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addLayout(score_layout)

    def _inspect_details(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Forensic Signature Analysis - {self.data.get('person_uid')}")
        dialog.setMinimumSize(540, 420)
        d_layout = QVBoxLayout(dialog)

        txt = QTextEdit()
        txt.setReadOnly(True)
        import json
        txt.setText(json.dumps(self.data, indent=2))
        d_layout.addWidget(txt)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        d_layout.addWidget(btn_close)

        dialog.exec()


class GalleryWidget(QWidget):
    """Ranked forensic search results gallery."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header bar
        header_layout = QHBoxLayout()
        self.lbl_title = QLabel("Candidate Matches: Ready for Query")
        self.lbl_title.setStyleSheet("font-weight: 600; color: #38bdf8; font-size: 14px;")
        header_layout.addWidget(self.lbl_title)
        header_layout.addStretch()

        self.btn_clear_view = QPushButton("Clear Results")
        self.btn_clear_view.setStyleSheet("background-color: #1e293b; padding: 4px 10px; font-size: 12px;")
        self.btn_clear_view.clicked.connect(self.clear_cards)
        header_layout.addWidget(self.btn_clear_view)
        layout.addLayout(header_layout)

        # Scrollable Cards Container
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: 1px solid #1e293b; background-color: #090d16; border-radius: 6px; }")

        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(8, 8, 8, 8)
        self.cards_layout.setSpacing(8)
        self.cards_layout.addStretch()

        self.scroll_area.setWidget(self.cards_container)
        layout.addWidget(self.scroll_area)

    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """Populate gallery with ranked candidate match cards."""
        self.clear_cards()
        self.lbl_title.setText(f"Candidate Matches: {len(results)} Persons Found")

        for item in results:
            card = CandidateCard(item)
            # Insert before trailing stretch
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

    def clear_cards(self) -> None:
        """Remove all cards from the gallery."""
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.lbl_title.setText("Candidate Matches: 0")
