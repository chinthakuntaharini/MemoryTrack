"""
Multi-Camera Trajectory & Timeline Tracking Widget
Visualizes the spatio-temporal route and sightings of suspects across camera nodes.
"""

from typing import List, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QLineEdit
)
from PyQt6.QtCore import Qt


class TrajectoryWidget(QWidget):
    """Spatio-temporal camera timeline and route history table."""

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Top query bar
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Target Person UID:"))
        self.input_uid = QLineEdit()
        self.input_uid.setPlaceholderText("e.g. UID-CAM_01-0001")
        top_layout.addWidget(self.input_uid)

        self.btn_load_trajectory = QPushButton("Track Multi-Camera Route")
        self.btn_load_trajectory.clicked.connect(self.load_trajectory)
        top_layout.addWidget(self.btn_load_trajectory)
        top_layout.addStretch()

        layout.addLayout(top_layout)

        # Timeline Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Step", "Camera Node", "Timestamp", "Upper Clothing", "Lower Clothing", "Carried Items", "Posture"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)

        layout.addWidget(self.table)

    def load_trajectory(self) -> None:
        """Fetch all temporal sightings for the queried UID."""
        target_uid = self.input_uid.text().strip()
        if not target_uid or not self.db:
            return

        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT camera_id, timestamp, upper_color, lower_color, accessories_json, posture
            FROM signatures
            WHERE person_uid = ?
            ORDER BY sig_id ASC
        """, (target_uid,))
        rows = cursor.fetchall()

        self.table.setRowCount(len(rows))
        import json
        for idx, r in enumerate(rows):
            accs = []
            try:
                accs = json.loads(r["accessories_json"] or "[]")
            except Exception:
                accs = []
            self.table.setItem(idx, 0, QTableWidgetItem(f"#{idx+1}"))
            self.table.setItem(idx, 1, QTableWidgetItem(str(r["camera_id"] or "")))
            self.table.setItem(idx, 2, QTableWidgetItem(str(r["timestamp"] or "")))
            self.table.setItem(idx, 3, QTableWidgetItem(str(r["upper_color"] or "Unknown")))
            self.table.setItem(idx, 4, QTableWidgetItem(str(r["lower_color"] or "None")))
            self.table.setItem(idx, 5, QTableWidgetItem(", ".join(accs) if accs else "None"))
            self.table.setItem(idx, 6, QTableWidgetItem(str(r["posture"] or "Standing")))
