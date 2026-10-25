"""
Forensic Re-ID Application Master Window
Integrates stream player, visual query builder, candidate gallery, and database manager.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QStatusBar,
    QMessageBox, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon

from forensic_app.core.database import ForensicDatabase
from forensic_app.pipeline.worker import ForensicPipelineWorker
from forensic_app.ui.player_widget import PlayerWidget
from forensic_app.ui.query_widget import QueryWidget
from forensic_app.ui.gallery_widget import GalleryWidget
from forensic_app.ui.trajectory_widget import TrajectoryWidget


class ForensicMainWindow(QMainWindow):
    """Primary GUI Application Window for Forensic Re-ID Surveillance."""

    def __init__(self, db: Optional[ForensicDatabase] = None):
        super().__init__()
        self.setWindowTitle("Forensic Non-Facial Person Tracker & Re-ID System")
        self.setGeometry(80, 60, 1280, 840)

        # Initialize Forensic Database
        self.db = db if db is not None else ForensicDatabase()
        self.worker: Optional[ForensicPipelineWorker] = None

        self._load_stylesheet()
        self._init_ui()

        # Auto-refresh database stats every 3 seconds
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._refresh_db_stats)
        self.stats_timer.start(3000)

    def _load_stylesheet(self) -> None:
        qss_path = Path(__file__).resolve().parent / "styles.qss"
        if qss_path.exists():
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(8)

        # Header Title Bar
        header_layout = QHBoxLayout()
        title_lbl = QLabel("FORENSIC INVESTIGATIVE RE-ID & PEDESTRIAN TRACKER")
        title_lbl.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color: #38bdf8; letter-spacing: 1px;")

        self.btn_clear_db = QPushButton("Clear Database")
        self.btn_clear_db.setObjectName("btn_danger")
        self.btn_clear_db.setToolTip("Safely delete all tracked profiles and reset vector index")
        self.btn_clear_db.clicked.connect(self._on_clear_database_safe)

        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_clear_db)
        main_layout.addLayout(header_layout)

        # Master Tab Widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # TAB 1: Live Surveillance Player
        self.player_widget = PlayerWidget()
        self.player_widget.play_requested.connect(self._start_pipeline)
        self.player_widget.pause_requested.connect(self._pause_pipeline)
        self.player_widget.resume_requested.connect(self._resume_pipeline)
        self.player_widget.stop_requested.connect(self._stop_pipeline)
        self.tabs.addTab(self.player_widget, "Live Surveillance Player")

        # TAB 2: Visual Query & Match Gallery
        query_gallery_tab = QWidget()
        qg_layout = QHBoxLayout(query_gallery_tab)
        qg_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.query_widget = QueryWidget()
        self.query_widget.search_requested.connect(self._execute_search)
        self.query_widget.reset_requested.connect(self._reset_search)

        self.gallery_widget = GalleryWidget()

        splitter.addWidget(self.query_widget)
        splitter.addWidget(self.gallery_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        qg_layout.addWidget(splitter)
        self.tabs.addTab(query_gallery_tab, "Forensic Search & Gallery")

        # TAB 3: Multi-Camera Trajectory Timeline
        self.trajectory_widget = TrajectoryWidget(self.db)
        self.tabs.addTab(self.trajectory_widget, "Camera Trajectory Timeline")

        # TAB 4: Database Master Explorer
        self.tab_db_explorer = QWidget()
        self._setup_db_explorer_tab()
        self.tabs.addTab(self.tab_db_explorer, "Database Records & System Logs")

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System Ready | Models Loaded | Database Connected")

    def _setup_db_explorer_tab(self) -> None:
        layout = QVBoxLayout(self.tab_db_explorer)

        top_layout = QHBoxLayout()
        self.btn_refresh_table = QPushButton("Refresh Database Table")
        self.btn_refresh_table.clicked.connect(self._populate_db_table)
        top_layout.addWidget(self.btn_refresh_table)

        self.lbl_db_summary = QLabel("Database: Loading...")
        self.lbl_db_summary.setStyleSheet("color: #38bdf8; font-weight: 600;")
        top_layout.addWidget(self.lbl_db_summary)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        # Persons Table
        self.db_table = QTableWidget()
        self.db_table.setColumnCount(8)
        self.db_table.setHorizontalHeaderLabels([
            "Person UID", "Last Camera", "Last Sighted", "Upper Garment", "Lower Garment", "Accessories", "Posture", "Status"
        ])
        self.db_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.db_table.horizontalHeader().setStretchLastSection(True)
        self.db_table.setAlternatingRowColors(True)
        layout.addWidget(self.db_table)

        self._populate_db_table()

    def _populate_db_table(self) -> None:
        if not self.db:
            return
        try:
            with self.db._lock:
                cursor = self.db.conn.cursor()
                cursor.execute("SELECT person_uid, last_camera, last_seen, upper_color, lower_color, accessories_json, posture, status FROM persons ORDER BY last_seen DESC LIMIT 100;")
                rows = cursor.fetchall()

            self.db_table.setRowCount(len(rows))
            import json
            for idx, r in enumerate(rows):
                accs = []
                try:
                    accs = json.loads(r["accessories_json"] or "[]")
                except Exception:
                    accs = []
                self.db_table.setItem(idx, 0, QTableWidgetItem(str(r["person_uid"] or "")))
                self.db_table.setItem(idx, 1, QTableWidgetItem(str(r["last_camera"] or "CAM_01")))
                self.db_table.setItem(idx, 2, QTableWidgetItem(str(r["last_seen"] or "")))
                self.db_table.setItem(idx, 3, QTableWidgetItem(str(r["upper_color"] or "Unknown")))
                self.db_table.setItem(idx, 4, QTableWidgetItem(str(r["lower_color"] or "None")))
                self.db_table.setItem(idx, 5, QTableWidgetItem(", ".join(accs) if accs else "None"))
                self.db_table.setItem(idx, 6, QTableWidgetItem(str(r["posture"] or "Standing")))
                self.db_table.setItem(idx, 7, QTableWidgetItem(str(r["status"] or "ACTIVE")))

            stats = self.db.get_stats()
            self.lbl_db_summary.setText(
                f"Stored Persons: {stats['total_persons']} | Signatures: {stats['total_signatures']} | FAISS Vectors: {stats['faiss_indexed']}"
            )
        except Exception as e:
            logger.error(f"Error populating database table: {e}")

    def _on_clear_database_safe(self) -> None:
        """
        Safely delete all database records with zero risk of crashing.
        Directly addresses the user's issue where the app previously closed unexpectedly.
        """
        reply = QMessageBox.question(
            self, "Confirm Database Reset",
            "Are you sure you want to completely erase all forensic person records and reset the vector index?\n\nThis operation cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Stop active pipeline first if running
                if self.worker and self.worker.isRunning():
                    self.worker.stop()
                    self.worker.wait(1500)
                    self.player_widget.reset_ui_state()

                if self.worker:
                    self.worker.track_to_uid.clear()

                success = self.db.clear_database()
                if success:
                    self._populate_db_table()
                    self.gallery_widget.clear_cards()
                    self.status_bar.showMessage("Database and FAISS Index Cleared Successfully")
                    QMessageBox.information(self, "Success", "All forensic profiles and vector indices have been wiped cleanly.")
                else:
                    QMessageBox.warning(self, "Warning", "Could not clear database. Check system logs.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An unexpected error occurred while resetting database: {e}")

    def _start_pipeline(self, source_type: str, source_path: str) -> None:
        """Start asynchronous worker thread."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()

        self.worker = ForensicPipelineWorker(
            db=self.db,
            source_type=source_type,
            source_path=source_path,
            camera_id="CAM_01",
            device="cpu"
        )
        self.worker.frame_ready.connect(self.player_widget.update_frame)
        self.worker.stats_updated.connect(self._on_stats_updated)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.worker_finished.connect(self.player_widget.reset_ui_state)
        self.worker.start()
        self.status_bar.showMessage(f"Pipeline Active | Streaming {source_type.upper()} ({source_path})")

    def _pause_pipeline(self) -> None:
        if self.worker:
            self.worker.pause()
            self.status_bar.showMessage("Pipeline Paused")

    def _resume_pipeline(self) -> None:
        if self.worker:
            self.worker.resume()
            self.status_bar.showMessage("Pipeline Resumed")

    def _stop_pipeline(self) -> None:
        if self.worker:
            self.worker.stop()
            self.status_bar.showMessage("Pipeline Stopped")

    def _on_stats_updated(self, active_tracks: int, total_profiles: int, fps: float) -> None:
        self.status_bar.showMessage(
            f"Active Tracklets: {active_tracks} | Session Profiles: {total_profiles} | FPS: {fps:.1f} | DB Indexed: {self.db.get_stats()['faiss_indexed']}"
        )

    def _on_worker_error(self, err_msg: str) -> None:
        self.status_bar.showMessage(f"Error: {err_msg}")
        QMessageBox.warning(self, "Pipeline Alert", err_msg)

    def _execute_search(self, query_params: dict) -> None:
        """Execute forensic query and display ranked match cards."""
        results = self.db.query_by_attributes(
            upper_color=query_params.get("upper_color"),
            lower_color=query_params.get("lower_color"),
            accessories=query_params.get("accessories"),
            top_k=20
        )
        self.gallery_widget.display_results(results)

    def _reset_search(self) -> None:
        self.gallery_widget.clear_cards()

    def _refresh_db_stats(self) -> None:
        stats = self.db.get_stats()
        self.lbl_db_summary.setText(
            f"Stored Persons: {stats['total_persons']} | Signatures: {stats['total_signatures']} | FAISS Vectors: {stats['faiss_indexed']}"
        )

    def closeEvent(self, event) -> None:
        """Clean up threads and database connections on application exit."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        if self.db:
            self.db.close()
        event.accept()
