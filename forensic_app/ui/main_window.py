"""
Forensic Re-ID Application Master Window
Glass console integrating the live stream + identity panel, visual query gallery, trajectories and database logs.
"""

import contextlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QEasingCurve, QRectF, Qt, QTimer, QVariantAnimation
from PyQt6.QtGui import QColor, QGuiApplication, QLinearGradient, QPainter
from PyQt6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton, QSplitter, QStatusBar, QTabBar,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget
)

from forensic_app.core.database import ForensicDatabase
from forensic_app.ui.effects import (
    ACCENT_AMBER, ACCENT_CYAN, ACCENT_EMERALD, ACCENT_ROSE, ACCENT_VIOLET, TEXT_SECONDARY, AuroraBackground,
    BrandMark, GlassPanel, brand_icon, enable_window_backdrop, install_hover_glows,
    load_glass_stylesheet, prepare_window_for_backdrop, set_role, set_variant, swatch_icon, ui_font
)
from forensic_app.ui.gallery_widget import GalleryWidget
from forensic_app.ui.live_panel import LiveIdentityPanel
from forensic_app.ui.player_widget import PlayerWidget
from forensic_app.ui.query_widget import QueryWidget
from forensic_app.ui.trajectory_widget import TrajectoryWidget
from forensic_app.utils.drawing import accessory_short_label, identity_color_hex

logger = logging.getLogger(__name__)

try:
    from forensic_app.pipeline.worker import ForensicPipelineWorker
    _WORKER_IMPORT_ERROR: Optional[str] = None
except Exception as _imp_err:  # keep the console usable when the ML stack cannot load
    ForensicPipelineWorker = None  # type: ignore[assignment]
    _WORKER_IMPORT_ERROR = str(_imp_err)
    logger.error(f"Pipeline worker unavailable: {_imp_err}")

try:
    from forensic_app.utils.color_naming import FORENSIC_COLOR_PALETTE
except Exception:  # pragma: no cover
    FORENSIC_COLOR_PALETTE = {}


def _pick_device() -> str:
    """'cuda' when a CUDA device is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _palette_hex(name: Optional[str]) -> Optional[str]:
    """Hex for a palette colour name (None when unknown)."""
    data = FORENSIC_COLOR_PALETTE.get(str(name or ""))
    return data.get("hex") if isinstance(data, dict) else None


class StatTile(GlassPanel):
    """Header stat tile: caption, large value, hint line and an accent glow dot."""

    def __init__(self, caption: str, accent: str, hint: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent, radius=14, accent=accent, tint=0.05)
        self._accent = QColor(accent)
        self.setMinimumWidth(128)
        self.setFixedHeight(66)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(0)
        self.lbl_caption = set_role(QLabel(caption.upper()), "caption")
        self.lbl_value = set_role(QLabel("0"), "value")
        self.lbl_value.setStyleSheet(f"color: {self._accent.lighter(118).name()};")
        self.lbl_hint = set_role(QLabel(hint), "hint")
        layout.addWidget(self.lbl_caption)
        layout.addWidget(self.lbl_value)
        layout.addWidget(self.lbl_hint)

    def set_value(self, value: Any, hint: Optional[str] = None) -> None:
        self.lbl_value.setText(str(value))
        if hint is not None:
            self.lbl_hint.setText(hint)

    def value(self) -> str:
        return self.lbl_value.text()


class GlowTabBar(QTabBar):
    """Glass tab bar with an animated glowing underline under the active tab."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setDrawBase(False)
        self.setExpanding(False)
        self._rect: Optional[QRectF] = None
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(260)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self.currentChanged.connect(self._animate_to)

    def _target(self, index: int) -> QRectF:
        r = self.tabRect(index)
        if r.isNull():
            return QRectF()
        return QRectF(r.left() + 16, r.bottom() - 6, max(8, r.width() - 38), 3)

    def _animate_to(self, index: int) -> None:
        target = self._target(index)
        if self._rect is None or self._rect.isNull() or target.isNull():
            self._rect = target
            self.update()
            return
        self._anim.stop()
        self._anim.setStartValue(QRectF(self._rect))
        self._anim.setEndValue(target)
        self._anim.start()

    def _on_anim(self, value) -> None:
        self._rect = QRectF(value)
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.count() == 0 or self.currentIndex() < 0:
            return
        if self._anim.state() != QVariantAnimation.State.Running:
            self._rect = self._target(self.currentIndex())
        rect = self._rect
        if rect is None or rect.isNull():
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        for grow, alpha in ((7.0, 0.10), (4.0, 0.20), (2.0, 0.38)):
            c = QColor(ACCENT_CYAN)
            c.setAlphaF(alpha)
            p.setBrush(c)
            g = rect.adjusted(-grow, -grow / 2, grow, grow / 2)
            p.drawRoundedRect(g, g.height() / 2, g.height() / 2)
        grad = QLinearGradient(rect.topLeft(), rect.topRight())
        grad.setColorAt(0.0, QColor(ACCENT_CYAN))
        grad.setColorAt(1.0, QColor(ACCENT_VIOLET))
        p.setBrush(grad)
        p.drawRoundedRect(rect, 1.5, 1.5)
        p.end()


class GlassTabWidget(QTabWidget):
    """QTabWidget using the glowing glass tab bar."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setTabBar(GlowTabBar(self))
        self.setDocumentMode(True)


class ForensicMainWindow(QMainWindow):
    """Primary GUI Application Window for Forensic Re-ID Surveillance."""

    def __init__(self, db: Optional[ForensicDatabase] = None):
        super().__init__()
        self.setWindowTitle("Forensic Non-Facial Person Tracker & Re-ID System")
        self.setWindowIcon(brand_icon())
        self.setMinimumSize(1100, 680)
        self._fit_to_screen(1440, 900)
        self.setFont(ui_font(13))

        # Initialize Forensic Database
        self.db = db if db is not None else ForensicDatabase()
        self.worker = None
        self._retired_workers: List[Any] = []
        self._session_identities = 0
        self._faiss_count = 0
        self._backdrop_checked = False
        self.backdrop_enabled = False
        # Transparency must be requested before the native window is first shown
        self._backdrop_prepared = prepare_window_for_backdrop(self)

        self._load_stylesheet()
        self._init_ui()
        install_hover_glows(self.centralWidget())

        # Auto-refresh database stats every 3 seconds
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._refresh_db_stats)
        self.stats_timer.start(3000)
        self._refresh_db_stats()

    # ---- construction --------------------------------------------------------------------------
    def _fit_to_screen(self, width: int, height: int) -> None:
        """Preferred size, shrunk to fit the available screen area (title bar included) and centred."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(width, height)
            return
        avail = screen.availableGeometry()
        frame_h = 48  # native title bar + borders
        w = max(self.minimumWidth(), min(width, avail.width() - 24))
        h = max(self.minimumHeight(), min(height, avail.height() - frame_h - 12))
        self.resize(w, h)
        self.move(avail.x() + max(0, (avail.width() - w) // 2), avail.y() + max(0, (avail.height() - h - frame_h) // 2))

    def _load_stylesheet(self) -> None:
        qss = load_glass_stylesheet()
        if qss:
            self.setStyleSheet(qss)

    def _db_lock(self):
        lock = getattr(self.db, "_lock", None)
        return lock if lock is not None else contextlib.nullcontext()

    def _init_ui(self) -> None:
        self.aurora = AuroraBackground()
        self.setCentralWidget(self.aurora)
        main_layout = QVBoxLayout(self.aurora)
        main_layout.setContentsMargins(18, 14, 18, 12)
        main_layout.setSpacing(10)

        main_layout.addWidget(self._build_header())

        # Master Tab Widget
        self.tabs = GlassTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        # TAB 1: Live Surveillance (player + live identity panel)
        live_tab = QWidget()
        live_layout = QHBoxLayout(live_tab)
        live_layout.setContentsMargins(0, 0, 0, 0)
        self.live_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.live_splitter.setChildrenCollapsible(False)
        self.player_widget = PlayerWidget()
        self.player_widget.play_requested.connect(self._start_pipeline)
        self.player_widget.pause_requested.connect(self._pause_pipeline)
        self.player_widget.resume_requested.connect(self._resume_pipeline)
        self.player_widget.stop_requested.connect(self._stop_pipeline)
        self.live_panel = LiveIdentityPanel()
        self.live_panel.setMinimumWidth(340)
        self.live_panel.identity_selected.connect(self._open_trajectory_for)
        self.live_splitter.addWidget(self.player_widget)
        self.live_splitter.addWidget(self.live_panel)
        self.live_splitter.setStretchFactor(0, 3)
        self.live_splitter.setStretchFactor(1, 1)
        self.live_splitter.setSizes([960, 400])
        live_layout.addWidget(self.live_splitter)
        self.tabs.addTab(live_tab, "Live Surveillance")

        # TAB 2: Visual Query & Match Gallery
        query_gallery_tab = QWidget()
        qg_layout = QHBoxLayout(query_gallery_tab)
        qg_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.query_widget = QueryWidget()
        self.query_widget.setMinimumWidth(400)
        self.query_widget.search_requested.connect(self._execute_search)
        self.query_widget.reset_requested.connect(self._reset_search)
        self.gallery_widget = GalleryWidget()
        splitter.addWidget(self.query_widget)
        splitter.addWidget(self.gallery_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([440, 920])
        qg_layout.addWidget(splitter)
        self.tabs.addTab(query_gallery_tab, "Forensic Search")

        # TAB 3: Multi-Camera Trajectory Timeline
        self.trajectory_widget = TrajectoryWidget(self.db)
        self.tabs.addTab(self.trajectory_widget, "Trajectory Timeline")

        # TAB 4: Database Master Explorer
        self.tab_db_explorer = QWidget()
        self._setup_db_explorer_tab()
        self.tabs.addTab(self.tab_db_explorer, "Database && Logs")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Glass status bar (inside the aurora so the backdrop continues behind it)
        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("glassStatus")
        self.status_bar.setSizeGripEnabled(False)
        self.lbl_status_stats = QLabel("Pipeline idle")
        self.status_bar.addPermanentWidget(self.lbl_status_stats)
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("System Ready | Models Loaded | Database Connected")

    def _build_header(self) -> QWidget:
        header = GlassPanel(radius=18, accent=ACCENT_CYAN)
        header.setFixedHeight(88)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(12)

        layout.addWidget(BrandMark(46))
        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title_lbl = set_role(QLabel("MemoryTrack Forensic Console"), "title")
        subtitle = set_role(QLabel("Non-facial person re-identification  |  appearance + clothing colours + "
                                   "accessories  |  multi-camera"), "subtitle")
        title_col.addStretch()
        title_col.addWidget(title_lbl)
        title_col.addWidget(subtitle)
        title_col.addStretch()
        layout.addLayout(title_col)
        layout.addStretch()

        self.tile_tracks = StatTile("Active Tracks", ACCENT_CYAN, "in current frame")
        self.tile_identities = StatTile("Identities", ACCENT_VIOLET, "this session")
        self.tile_fps = StatTile("FPS", ACCENT_EMERALD, "pipeline idle")
        self.tile_fps.set_value("0.0")
        self.tile_vectors = StatTile("DB Vectors", ACCENT_AMBER, "indexed")
        for tile in (self.tile_tracks, self.tile_identities, self.tile_fps, self.tile_vectors):
            layout.addWidget(tile)

        layout.addSpacing(6)
        self.btn_clear_db = set_variant(QPushButton("Clear Database"), "danger")
        self.btn_clear_db.setObjectName("btn_danger")
        self.btn_clear_db.setToolTip("Safely delete all tracked profiles and reset vector index")
        self.btn_clear_db.setMinimumHeight(40)
        self.btn_clear_db.clicked.connect(self._on_clear_database_safe)
        layout.addWidget(self.btn_clear_db)
        return header

    def _setup_db_explorer_tab(self) -> None:
        layout = QVBoxLayout(self.tab_db_explorer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        bar = GlassPanel(radius=16)
        top_layout = QHBoxLayout(bar)
        top_layout.setContentsMargins(16, 10, 16, 10)
        top_layout.setSpacing(12)
        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title_col.addWidget(set_role(QLabel("DATABASE RECORDS"), "section"))
        title_col.addWidget(set_role(QLabel("Persistent identities, sightings and system events"), "hint"))
        top_layout.addLayout(title_col)
        top_layout.addSpacing(10)
        self.lbl_db_summary = set_role(QLabel("Database: Loading..."), "summary")
        top_layout.addWidget(self.lbl_db_summary)
        top_layout.addStretch()
        self.btn_refresh_table = set_variant(QPushButton("Refresh Database Table"), "ghost")
        self.btn_refresh_table.clicked.connect(self._populate_db_table)
        top_layout.addWidget(self.btn_refresh_table)
        layout.addWidget(bar)

        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)

        persons_panel = GlassPanel(radius=18)
        pl = QVBoxLayout(persons_panel)
        pl.setContentsMargins(14, 12, 14, 12)
        pl.setSpacing(8)
        pl.addWidget(set_role(QLabel("PERSON IDENTITIES"), "section"))
        # Persons Table
        self.db_table = QTableWidget()
        self.db_table.setColumnCount(9)
        self.db_table.setHorizontalHeaderLabels([
            "Person UID", "Last Camera", "Last Sighted", "Upper Garment", "Lower Garment", "Accessories",
            "Posture", "Sightings", "Status"
        ])
        self._style_table(self.db_table)
        self.db_table.cellDoubleClicked.connect(self._on_person_double_clicked)
        pl.addWidget(self.db_table)
        split.addWidget(persons_panel)

        logs_panel = GlassPanel(radius=18)
        ll = QVBoxLayout(logs_panel)
        ll.setContentsMargins(14, 12, 14, 12)
        ll.setSpacing(8)
        ll.addWidget(set_role(QLabel("SYSTEM LOGS"), "section"))
        self.logs_table = QTableWidget()
        self.logs_table.setColumnCount(4)
        self.logs_table.setHorizontalHeaderLabels(["Time", "Event", "Severity", "Details"])
        self._style_table(self.logs_table)
        ll.addWidget(self.logs_table)
        split.addWidget(logs_panel)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, stretch=1)

        self._populate_db_table()

    @staticmethod
    def _style_table(table: QTableWidget) -> None:
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setHighlightSections(False)
        table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(34)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setWordWrap(False)

    # ---- window lifecycle ----------------------------------------------------------------------
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._backdrop_checked:
            self._backdrop_checked = True
            # Always applies the dark title bar; acrylic (or FORENSIC_GLASS_MODE=mica) only when prepared
            mode = os.environ.get("FORENSIC_GLASS_MODE", "acrylic").strip().lower() or "acrylic"
            ok = enable_window_backdrop(self, mode)
            self.backdrop_enabled = bool(ok)
            self.aurora.set_backdrop_mode(self.backdrop_enabled)

    # ---- database views ------------------------------------------------------------------------
    def _populate_db_table(self) -> None:
        if not self.db:
            return
        try:
            rows = self._query_persons()
            self.db_table.setRowCount(len(rows))
            for idx, r in enumerate(rows):
                accs = []
                try:
                    accs = json.loads(r.get("accessories_json") or "[]")
                except Exception:
                    accs = []
                uid = str(r.get("person_uid") or "")
                uid_item = QTableWidgetItem(uid)
                uid_item.setForeground(QColor(identity_color_hex(uid)).lighter(110))
                f = uid_item.font()
                f.setBold(True)
                uid_item.setFont(f)
                self.db_table.setItem(idx, 0, uid_item)
                self.db_table.setItem(idx, 1, QTableWidgetItem(str(r.get("last_camera") or "CAM_01")))
                self.db_table.setItem(idx, 2, QTableWidgetItem(str(r.get("last_seen") or "")))
                upper = str(r.get("upper_color") or "Unknown")
                up_item = QTableWidgetItem(upper)
                up_item.setIcon(swatch_icon(r.get("upper_hex") or _palette_hex(upper)))
                self.db_table.setItem(idx, 3, up_item)
                lower = str(r.get("lower_color") or "None")
                lo_item = QTableWidgetItem(lower)
                lo_item.setIcon(swatch_icon(r.get("lower_hex") or _palette_hex(lower)))
                self.db_table.setItem(idx, 4, lo_item)
                acc_text = ", ".join(accessory_short_label(a) for a in accs) if accs else "None"
                self.db_table.setItem(idx, 5, QTableWidgetItem(acc_text))
                self.db_table.setItem(idx, 6, QTableWidgetItem(str(r.get("posture") or "Standing")))
                self.db_table.setItem(idx, 7, QTableWidgetItem(str(r.get("n_sightings") or "")))
                status = str(r.get("status") or "ACTIVE")
                st_item = QTableWidgetItem(status)
                st_item.setForeground(QColor(ACCENT_EMERALD if status.upper() == "ACTIVE" else TEXT_SECONDARY))
                self.db_table.setItem(idx, 8, st_item)

            self._populate_logs()
            self._refresh_db_stats()
        except Exception as e:
            logger.error(f"Error populating database table: {e}")

    def _query_persons(self) -> List[Dict[str, Any]]:
        """Persons with their latest garment hex colours and sighting counts (db lock held)."""
        with self._db_lock():
            cursor = self.db.conn.cursor()
            try:
                cursor.execute("""
                    SELECT p.person_uid, p.last_camera, p.last_seen, p.upper_color, p.lower_color,
                           p.accessories_json, p.posture, p.status,
                           (SELECT s.upper_hex FROM signatures s WHERE s.person_uid = p.person_uid
                             ORDER BY s.sig_id DESC LIMIT 1) AS upper_hex,
                           (SELECT s.lower_hex FROM signatures s WHERE s.person_uid = p.person_uid
                             ORDER BY s.sig_id DESC LIMIT 1) AS lower_hex,
                           (SELECT COUNT(*) FROM signatures s WHERE s.person_uid = p.person_uid) AS n_sightings
                    FROM persons p ORDER BY p.last_seen DESC LIMIT 100;
                """)
            except Exception:
                cursor.execute("SELECT person_uid, last_camera, last_seen, upper_color, lower_color, "
                               "accessories_json, posture, status FROM persons ORDER BY last_seen DESC LIMIT 100;")
            return [dict(r) for r in cursor.fetchall()]

    def _populate_logs(self) -> None:
        try:
            with self._db_lock():
                cursor = self.db.conn.cursor()
                cursor.execute("SELECT log_time, event_type, severity, details FROM system_logs "
                               "ORDER BY log_id DESC LIMIT 200;")
                rows = [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            logger.debug(f"System logs unavailable: {e}")
            rows = []
        sev_colors = {"INFO": ACCENT_CYAN, "WARNING": ACCENT_AMBER, "WARN": ACCENT_AMBER,
                      "ERROR": ACCENT_ROSE, "CRITICAL": ACCENT_ROSE}
        self.logs_table.setRowCount(len(rows))
        for idx, r in enumerate(rows):
            self.logs_table.setItem(idx, 0, QTableWidgetItem(str(r.get("log_time") or "")))
            self.logs_table.setItem(idx, 1, QTableWidgetItem(str(r.get("event_type") or "")))
            sev = str(r.get("severity") or "INFO").upper()
            sev_item = QTableWidgetItem(sev)
            sev_item.setForeground(QColor(sev_colors.get(sev, TEXT_SECONDARY)))
            self.logs_table.setItem(idx, 2, sev_item)
            self.logs_table.setItem(idx, 3, QTableWidgetItem(str(r.get("details") or "")))

    def _on_tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.tab_db_explorer:
            self._populate_db_table()

    def _on_person_double_clicked(self, row: int, _col: int) -> None:
        item = self.db_table.item(row, 0)
        if item is not None and item.text():
            self._open_trajectory_for(item.text())

    def _open_trajectory_for(self, uid: str) -> None:
        """Jump to the trajectory tab and trace a uid."""
        self.tabs.setCurrentWidget(self.trajectory_widget)
        self.trajectory_widget.set_uid(uid)

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
                if self.worker is not None and self.worker.isRunning():
                    self.worker.stop()
                    self.worker.wait(1500)
                    self.player_widget.reset_ui_state()

                if self.worker is not None:
                    reset = getattr(self.worker, "reset_identities", None)
                    if callable(reset):
                        reset()
                    # The stopped worker is retired so its already-queued late signals (frames,
                    # signatures of wiped identities) are ignored by _is_current()
                    self._retire_worker()

                success = self.db.clear_database()
                if success:
                    self.live_panel.clear()
                    self.gallery_widget.clear_cards()
                    self._populate_db_table()
                    self.status_bar.showMessage("Database and FAISS Index Cleared Successfully", 6000)
                    QMessageBox.information(self, "Success", "All forensic profiles and vector indices have been wiped cleanly.")
                else:
                    QMessageBox.warning(self, "Warning", "Could not clear database. Check system logs.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An unexpected error occurred while resetting database: {e}")

    # ---- pipeline control ----------------------------------------------------------------------
    def _retire_worker(self) -> None:
        """Stop the current worker; keep a reference until its thread really ends."""
        worker = self.worker
        self.worker = None
        if worker is None:
            return
        try:
            if worker.isRunning():
                worker.stop()
            if worker.isRunning():
                self._retired_workers.append(worker)
                worker.finished.connect(lambda w=worker: self._forget_worker(w))
        except Exception as e:
            logger.debug(f"Worker shutdown issue: {e}")

    def _forget_worker(self, worker) -> None:
        if worker in self._retired_workers:
            self._retired_workers.remove(worker)

    def _is_current(self) -> bool:
        """True when the signal sender is the active worker (ignores late signals of stopped workers)."""
        sender = self.sender()
        return sender is None or sender is self.worker

    def _start_pipeline(self, source_type: str, source_path: str) -> None:
        """Start asynchronous worker thread."""
        self._retire_worker()
        if ForensicPipelineWorker is None:
            self.player_widget.reset_ui_state()
            QMessageBox.critical(self, "Pipeline Unavailable",
                                 f"The inference pipeline could not be loaded:\n\n{_WORKER_IMPORT_ERROR}")
            return

        camera_id = self.player_widget.current_camera_id()
        device = _pick_device()
        self.worker = ForensicPipelineWorker(
            db=self.db,
            source_type=source_type,
            source_path=source_path,
            camera_id=camera_id,
            device=device
        )
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.stats_updated.connect(self._on_stats_updated)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.worker_finished.connect(self._on_worker_finished)
        sig_signal = getattr(self.worker, "signature_detected", None)
        if sig_signal is not None:
            sig_signal.connect(self._on_signature)
        event_signal = getattr(self.worker, "identity_event", None)
        if event_signal is not None:
            event_signal.connect(self._on_identity_event)
        self.worker.start()
        self.tile_fps.set_value("0.0", f"{device.upper()}  |  {camera_id}")
        self.status_bar.showMessage(f"Pipeline Active | Streaming {source_type.upper()} ({source_path}) | "
                                    f"{camera_id} | device {device}")

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
            self._retire_worker()
            self.status_bar.showMessage("Pipeline Stopped")
            self.tile_fps.set_value("0.0", "pipeline idle")
            self.lbl_status_stats.setText("Pipeline idle")

    def _on_frame_ready(self, frame) -> None:
        if self._is_current():
            self.player_widget.update_frame(frame)

    def _on_signature(self, payload: dict) -> None:
        if self._is_current():
            self.live_panel.on_signature(payload)

    def _on_identity_event(self, payload: dict) -> None:
        if not self._is_current():
            return
        self.live_panel.on_identity_event(payload)
        kind = str(payload.get("type") or "")
        uid = payload.get("person_uid", "")
        cam = payload.get("camera_id", "")
        if kind == "reacquired":
            conf = payload.get("confidence")
            pct = f" ({float(conf) * 100:.0f}% match)" if isinstance(conf, (int, float)) else ""
            msg = f"{uid} re-identified on {cam}{pct}"
        elif kind == "new":
            msg = f"New identity {uid} registered on {cam}"
        elif kind == "merged":
            msg = f"{payload.get('merged_from')} merged into {uid}"
        else:
            msg = f"{kind}: {uid}"
        self.status_bar.showMessage(msg, 5000)

    def _on_worker_finished(self) -> None:
        if self._is_current():
            self.player_widget.reset_ui_state()
            self.tile_fps.set_value("0.0", "pipeline idle")
            self.lbl_status_stats.setText("Pipeline idle")

    def _on_stats_updated(self, active_tracks: int, total_profiles: int, fps: float) -> None:
        if not self._is_current():
            return
        self._session_identities = int(total_profiles)
        self.tile_tracks.set_value(int(active_tracks))
        self.tile_identities.set_value(int(total_profiles))
        self.tile_fps.set_value(f"{float(fps):.1f}", "frames / sec")
        self.player_widget.set_pipeline_fps(float(fps))
        self.live_panel.set_session_identities(int(total_profiles))
        camera = self.player_widget.current_camera_id()
        self.lbl_status_stats.setText(
            f"Active Tracklets: {active_tracks} | Session Profiles: {total_profiles} | FPS: {fps:.1f} | "
            f"DB Indexed: {self._faiss_count} | {camera}"
        )

    def _on_worker_error(self, err_msg: str) -> None:
        if not self._is_current():
            return
        self.status_bar.showMessage(f"Error: {err_msg}")
        QMessageBox.warning(self, "Pipeline Alert", err_msg)

    # ---- search --------------------------------------------------------------------------------
    def _execute_search(self, query_params: dict) -> None:
        """Execute forensic query and display ranked match cards."""
        results = self.db.query_by_attributes(
            upper_color=query_params.get("upper_color"),
            lower_color=query_params.get("lower_color"),
            accessories=query_params.get("accessories"),
            top_k=20
        )
        self.gallery_widget.display_results(results)
        self.status_bar.showMessage(f"Forensic search returned {len(results)} candidate(s)", 5000)

    def _reset_search(self) -> None:
        self.gallery_widget.clear_cards()

    def _refresh_db_stats(self) -> None:
        try:
            stats = self.db.get_stats()
        except Exception as e:
            logger.debug(f"DB stats unavailable: {e}")
            return
        self._faiss_count = int(stats.get("faiss_indexed", 0))
        self.lbl_db_summary.setText(
            f"Stored Persons: {stats['total_persons']} | Signatures: {stats['total_signatures']} | FAISS Vectors: {stats['faiss_indexed']}"
        )
        self.tile_vectors.set_value(self._faiss_count, f"{stats['total_signatures']} signatures")
        running = self.worker is not None
        if not running:
            self.tile_identities.set_value(stats["total_persons"], "stored identities")
        else:
            self.tile_identities.lbl_hint.setText(f"session  |  {stats['total_persons']} stored")

    def closeEvent(self, event) -> None:
        """Clean up threads and database connections on application exit."""
        self.stats_timer.stop()
        workers = [w for w in [self.worker] + self._retired_workers if w is not None]
        for w in workers:
            try:
                if w.isRunning():
                    w.stop()
                    w.wait(3000)
            except Exception:
                pass
        if self.db:
            self.db.close()
        event.accept()
