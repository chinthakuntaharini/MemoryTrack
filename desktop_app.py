import os
import sys

# Critical configuration for PyTorch / OpenMP / PyInstaller on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

# Handle PyInstaller frozen environment & DLL loading
if getattr(sys, 'frozen', False):
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    
    if sys.platform == 'win32':
        # Add all possible torch & dependency DLL directories
        dll_dirs = [
            os.path.join(base_dir, 'torch', 'lib'),
            os.path.join(base_dir, '_internal', 'torch', 'lib'),
            os.path.join(os.path.dirname(sys.executable), '_internal', 'torch', 'lib'),
            os.path.join(os.path.dirname(sys.executable), 'torch', 'lib'),
            base_dir,
            os.path.join(base_dir, '_internal'),
        ]
        for d in dll_dirs:
            if os.path.exists(d):
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass
                os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')

    try:
        os.chdir(base_dir)
    except Exception:
        pass

# Critical: Import torch FIRST on Windows to avoid DLL collision with PyQt6/OpenMP
try:
    import torch
except Exception as e:
    print(f"Warning loading torch: {e}")

import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QStatusBar,
    QFrame, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QLineEdit, QComboBox, QGroupBox, QSplitter, QTextEdit, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon

# Import MemoryTrack components
try:
    from main import MemoryTrackPipeline
    from utils.video_loader import SingleVideoLoader
    from utils.db_manager import DatabaseManager
except ImportError as e:
    print(f"Error importing MemoryTrack components: {e}")
    sys.exit(1)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_stats_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, pipeline, source_type="video", source_path=None):
        super().__init__()
        self.pipeline = pipeline
        self.source_type = source_type
        self.source_path = source_path
        self._run_flag = True

    def run(self):
        try:
            if self.source_type == "video":
                self._process_video()
            elif self.source_type == "webcam":
                self._process_webcam()
        except Exception as e:
            self.update_stats_signal.emit(f"Error: {str(e)}")
        finally:
            self.finished_signal.emit()

    def _process_video(self):
        video_loader = SingleVideoLoader(self.source_path, camera_id="main")
        frame_count = 0
        start_time = time.time()

        for camera_frame in video_loader:
            if not self._run_flag:
                break

            frame = camera_frame.frame
            results = self.pipeline.process_frame(
                frame,
                camera_id=camera_frame.camera_id,
                frame_number=camera_frame.frame_number
            )

            processed_frame = results.get('processed_frame', frame)
            fps = frame_count / (time.time() - start_time + 1e-6)
            
            processed_frame = self.pipeline.visualizer.draw_camera_info(
                processed_frame, camera_frame.camera_id, camera_frame.frame_number, fps
            )

            self.change_pixmap_signal.emit(processed_frame)
            self.update_stats_signal.emit(f"Processing Video... FPS: {fps:.1f} | Frame: {frame_count} | Detections: {len(results.get('detections', []))}")
            
            frame_count += 1
            time.sleep(1/30)

    def _process_webcam(self):
        camera_index = 0
        video_loader = SingleVideoLoader(camera_index, camera_id="webcam")
        frame_count = 0
        start_time = time.time()

        for camera_frame in video_loader:
            if not self._run_flag:
                break

            frame = camera_frame.frame
            results = self.pipeline.process_frame(
                frame,
                camera_id=camera_frame.camera_id,
                frame_number=camera_frame.frame_number
            )

            processed_frame = results.get('processed_frame', frame)
            fps = frame_count / (time.time() - start_time + 1e-6)
            
            processed_frame = self.pipeline.visualizer.draw_camera_info(
                processed_frame, camera_frame.camera_id, camera_frame.frame_number, fps
            )

            self.change_pixmap_signal.emit(processed_frame)
            self.update_stats_signal.emit(f"Webcam Live | FPS: {fps:.1f} | Frame: {frame_count} | Detections: {len(results.get('detections', []))}")
            
            frame_count += 1

    def stop(self):
        self._run_flag = False
        self.wait()


class MemoryTrackApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MemoryTrack - Multi-Camera Missing Person Tracking System")
        self.setGeometry(100, 80, 1150, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QWidget { color: #cdd6f4; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            QTabWidget::pane { border: 1px solid #313244; background: #181825; border-radius: 8px; }
            QTabBar::tab {
                background: #181825; color: #a6adc8; padding: 10px 20px;
                margin-right: 4px; border-top-left-radius: 6px; border-top-right-radius: 6px;
                font-weight: bold; font-size: 13px;
            }
            QTabBar::tab:selected { background: #313244; color: #89b4fa; }
            QPushButton { 
                background-color: #89b4fa; color: #11111b; 
                border-radius: 6px; padding: 8px 16px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background-color: #b4befe; }
            QPushButton:disabled { background-color: #45475a; color: #6c7086; }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
                border-radius: 5px; padding: 6px; font-size: 13px;
            }
            QTableWidget {
                background-color: #181825; alternate-background-color: #1e1e2e;
                gridline-color: #313244; color: #cdd6f4; border: 1px solid #313244;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #313244; color: #89b4fa; padding: 6px;
                font-weight: bold; border: 1px solid #181825;
            }
            QGroupBox {
                border: 1px solid #313244; border-radius: 6px; margin-top: 10px;
                font-weight: bold; color: #89b4fa; padding: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)

        self.pipeline = None
        self.thread = None
        self.db = None
        
        self.init_ui()
        self.init_pipeline()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header Title
        title_label = QLabel("MemoryTrack System")
        title_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #89b4fa; margin: 6px;")
        main_layout.addWidget(title_label)

        # Main Tabs Widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Live Video Feed
        self.tab_live = QWidget()
        self._setup_live_tab()
        self.tabs.addTab(self.tab_live, "Live Tracking")

        # Tab 2: Memory Bank & Person Profiles
        self.tab_memory = QWidget()
        self._setup_memory_tab()
        self.tabs.addTab(self.tab_memory, "Memory Bank & Profiles")

        # Tab 3: Missing Person Search
        self.tab_search = QWidget()
        self._setup_search_tab()
        self.tabs.addTab(self.tab_search, "Missing Person Search")

        # Tab 4: System Logs & Database
        self.tab_logs = QWidget()
        self._setup_logs_tab()
        self.tabs.addTab(self.tab_logs, "Database & Activity Logs")

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing system...")
        self.status_bar.setStyleSheet("background-color: #11111b; color: #a6adc8; padding: 4px;")

    def _setup_live_tab(self):
        layout = QVBoxLayout(self.tab_live)

        # Video Display Area
        self.video_label = QLabel("Please load a video file or start webcam to begin live tracking.")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #11111b; border: 2px solid #313244; border-radius: 8px; font-size: 15px;")
        self.video_label.setMinimumSize(750, 450)
        layout.addWidget(self.video_label, stretch=1)

        # Controls Area
        controls_layout = QHBoxLayout()
        self.btn_load_video = QPushButton("Load Video File")
        self.btn_load_video.clicked.connect(self.load_video)
        
        self.btn_start_webcam = QPushButton("Start Webcam")
        self.btn_start_webcam.clicked.connect(self.start_webcam)
        
        self.btn_stop = QPushButton("Stop Processing")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)

        controls_layout.addWidget(self.btn_load_video)
        controls_layout.addWidget(self.btn_start_webcam)
        controls_layout.addWidget(self.btn_stop)
        layout.addLayout(controls_layout)

    def _setup_memory_tab(self):
        layout = QVBoxLayout(self.tab_memory)

        # Action bar
        action_layout = QHBoxLayout()
        self.btn_refresh_memory = QPushButton("Refresh Memory Bank")
        self.btn_refresh_memory.clicked.connect(self.refresh_memory_table)

        self.btn_clear_memory = QPushButton("Clear Database")
        self.btn_clear_memory.setStyleSheet("background-color: #f38ba8; color: #11111b; font-weight: bold;")
        self.btn_clear_memory.clicked.connect(self.clear_all_memory)
        
        self.lbl_memory_stats = QLabel("Memory Bank: Loading...")
        self.lbl_memory_stats.setStyleSheet("color: #89b4fa; font-weight: bold; font-size: 13px;")
        
        action_layout.addWidget(self.btn_refresh_memory)
        action_layout.addWidget(self.btn_clear_memory)
        action_layout.addWidget(self.lbl_memory_stats, stretch=1)
        layout.addLayout(action_layout)

        # Profiles Table with Forensic Details
        self.profiles_table = QTableWidget()
        self.profiles_table.setColumnCount(10)
        self.profiles_table.setHorizontalHeaderLabels([
            "ID", "Name", "Status", "Upper Clothes", "Lower Clothes",
            "Accessories", "Posture", "Snapshots", "Camera", "Timestamp"
        ])
        self.profiles_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.profiles_table.horizontalHeader().setStretchLastSection(True)
        self.profiles_table.setAlternatingRowColors(True)
        layout.addWidget(self.profiles_table)

    def _setup_search_tab(self):
        layout = QVBoxLayout(self.tab_search)

        group_box = QGroupBox("Forensic Search & Person Inspection")
        form_layout = QHBoxLayout(group_box)

        self.input_search_id = QSpinBox()
        self.input_search_id.setRange(1, 99999)
        self.input_search_id.setValue(1)
        
        self.btn_execute_search = QPushButton("Search Person ID")
        self.btn_execute_search.clicked.connect(self.execute_search)

        form_layout.addWidget(QLabel("Target Person ID:"))
        form_layout.addWidget(self.input_search_id)
        form_layout.addWidget(self.btn_execute_search)
        form_layout.addStretch()
        layout.addWidget(group_box)

        # Search Results Area
        self.search_results_text = QTextEdit()
        self.search_results_text.setReadOnly(True)
        self.search_results_text.setStyleSheet("background-color: #11111b; color: #cdd6f4; font-family: 'Consolas', monospace; font-size: 13px; padding: 12px;")
        layout.addWidget(self.search_results_text)

    def _setup_logs_tab(self):
        layout = QVBoxLayout(self.tab_logs)
        
        action_layout = QHBoxLayout()
        self.btn_refresh_logs = QPushButton("Refresh Logs")
        self.btn_refresh_logs.clicked.connect(self.refresh_logs)
        action_layout.addWidget(self.btn_refresh_logs)
        action_layout.addStretch()
        layout.addLayout(action_layout)

        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setStyleSheet("background-color: #11111b; color: #a6adc8; font-family: 'Consolas', monospace; font-size: 12px; padding: 10px;")
        layout.addWidget(self.logs_text)

    def init_pipeline(self):
        try:
            self.pipeline = MemoryTrackPipeline()
            self.db = DatabaseManager()
            self.status_bar.showMessage("System Ready | Models & DB Loaded")
            self.refresh_memory_table()
            self.refresh_logs()

            # Auto-refresh timer every 2 seconds during execution
            from PyQt6.QtCore import QTimer
            self.auto_refresh_timer = QTimer(self)
            self.auto_refresh_timer.timeout.connect(self._on_auto_refresh)
            self.auto_refresh_timer.start(2000)
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize MemoryTrack Pipeline:\n{e}")
            self.status_bar.showMessage("Initialization Failed")

    def _on_auto_refresh(self):
        """Periodically refresh tables if thread is running or periodically."""
        if self.thread and self.thread.isRunning():
            self.refresh_memory_table()
            self.refresh_logs()

    def refresh_memory_table(self):
        """Fetch profiles from database & in-memory memory bank and update table."""
        if not self.db:
            return

        try:
            persons = self.db.list_persons()
            self.profiles_table.setRowCount(len(persons))

            for row_idx, p in enumerate(persons):
                pid = p.get('person_id', row_idx + 1)
                name = p.get('name') or f"Person #{pid}"
                status = p.get('status', 'active')
                upper_col = p.get('upper_color') or "Detecting..."
                lower_col = p.get('lower_color') or "Detecting..."
                accessories = p.get('accessories') or "None"
                posture = p.get('posture') or "Standing"
                
                # Fetch snapshot count
                snapshots = self.db.get_person_snapshots(pid)
                snap_count = str(len(snapshots))
                
                camera = p.get('last_seen_camera') or "cam_0"
                seen_time = str(p.get('last_seen_time') or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                self.profiles_table.setItem(row_idx, 0, QTableWidgetItem(str(pid)))
                self.profiles_table.setItem(row_idx, 1, QTableWidgetItem(name))
                self.profiles_table.setItem(row_idx, 2, QTableWidgetItem(status.upper()))
                self.profiles_table.setItem(row_idx, 3, QTableWidgetItem(upper_col))
                self.profiles_table.setItem(row_idx, 4, QTableWidgetItem(lower_col))
                self.profiles_table.setItem(row_idx, 5, QTableWidgetItem(accessories))
                self.profiles_table.setItem(row_idx, 6, QTableWidgetItem(posture))
                self.profiles_table.setItem(row_idx, 7, QTableWidgetItem(snap_count))
                self.profiles_table.setItem(row_idx, 8, QTableWidgetItem(camera))
                self.profiles_table.setItem(row_idx, 9, QTableWidgetItem(seen_time))

            # Memory bank summary
            if self.pipeline and hasattr(self.pipeline, 'memory_bank'):
                stats = self.pipeline.memory_bank.get_statistics()
                self.lbl_memory_stats.setText(
                    f"Memory Bank: {stats.get('total_profiles', len(persons))} Tracked Persons | "
                    f"{stats.get('total_snapshots', 0)} Feature Snapshots | 720-D Vectors"
                )
        except Exception as e:
            self.lbl_memory_stats.setText(f"Error refreshing: {e}")

    def clear_all_memory(self):
        """Wipe database and reset in-memory memory bank."""
        reply = QMessageBox.question(
            self, "Clear Database",
            "Are you sure you want to delete all tracked persons and snapshot history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self.db:
                self.db.clear_database()
            if self.pipeline and hasattr(self.pipeline, 'memory_bank'):
                self.pipeline.memory_bank.profiles.clear()
                self.pipeline.memory_bank.index.reset()
            if self.pipeline and hasattr(self.pipeline, 'person_temporal_history'):
                self.pipeline.person_temporal_history.clear()
                self.pipeline.track_to_person_map.clear()
            self.refresh_memory_table()
            self.refresh_logs()
            self.status_bar.showMessage("Database and Memory Bank Cleared")

    def execute_search(self):
        """Search person by ID and show forensic details, clothing colors, and snapshots."""
        target_id = self.input_search_id.value()
        self.search_results_text.clear()

        if not self.db:
            self.search_results_text.setText("Database connection unavailable.")
            return

        person = self.db.get_person(target_id)
        if not person:
            self.search_results_text.setText(f"Person ID {target_id} not found in database.\n\nTip: Start video playback or webcam to detect and store person profiles.")
            return

        snapshots = self.db.get_person_snapshots(target_id)
        report = []
        report.append("=" * 65)
        report.append(f"FORENSIC INVESTIGATION PROFILE - PERSON ID: #{target_id}")
        report.append("=" * 65)
        report.append(f"  * Name / Label:          {person.get('name', 'Unknown')}")
        report.append(f"  * Current Status:        {person.get('status', 'ACTIVE').upper()}")
        report.append(f"  * Upper Clothing:        {person.get('upper_color', 'Unknown')}")
        report.append(f"  * Lower Clothing:        {person.get('lower_color', 'Unknown')}")
        report.append(f"  * Carried Items:         {person.get('accessories', 'None')}")
        report.append(f"  * Body Posture / Gait:   {person.get('posture', 'Standing')}")
        report.append(f"  * Last Seen Camera:      {person.get('last_seen_camera', 'N/A')}")
        report.append(f"  * Last Seen Timestamp:   {person.get('last_seen_time', 'N/A')}")
        report.append(f"  * Total Snapshots:       {len(snapshots)}")
        report.append(f"  * Investigation Notes:   {person.get('notes', 'N/A')}")
        report.append("-" * 65)
        report.append("TEMPORAL SNAPSHOT HISTORY (Multi-Modal Vectors):")
        
        for idx, s in enumerate(snapshots[:8]):
            report.append(f"  [Snapshot #{idx+1}] Cam: {s.get('camera_id')} | Conf: {s.get('confidence', 1.0)*100:.1f}% | Frame: {s.get('frame_number', 'N/A')} | Time: {s.get('timestamp')}")
            
        report.append("=" * 65)
        self.search_results_text.setText("\n".join(report))

    def refresh_logs(self):
        """Show recent activity logs from DB."""
        if not self.db:
            return
        try:
            cursor = self.db.connection.cursor()
            cursor.execute("SELECT event_time, event_type, details, severity FROM system_events ORDER BY event_time DESC, event_id DESC LIMIT 50")
            rows = cursor.fetchall()
            if not rows:
                self.logs_text.setText("No system events logged yet. Start video tracking to generate events.")
                return

            log_entries = []
            for r in rows:
                log_entries.append(f"[{r['event_time']}] [{r['severity']}] {r['event_type']}: {r['details']}")
            self.logs_text.setText("\n".join(log_entries))
        except Exception as e:
            self.logs_text.setText(f"Error fetching logs: {e}")


    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.video_label.width(), self.video_label.height(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        return QPixmap.fromImage(p)

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def update_status(self, text):
        self.status_bar.showMessage(text)

    def load_video(self):
        if self.pipeline is None:
            QMessageBox.warning(self, "Warning", "Pipeline is not initialized yet.")
            return

        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.start_thread("video", file_name)

    def start_webcam(self):
        if self.pipeline is None:
            QMessageBox.warning(self, "Warning", "Pipeline is not initialized yet.")
            return
        self.start_thread("webcam")

    def start_thread(self, source_type, source_path=None):
        self.btn_load_video.setEnabled(False)
        self.btn_start_webcam.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.video_label.setText("Starting process...")

        self.thread = VideoThread(self.pipeline, source_type, source_path)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_stats_signal.connect(self.update_status)
        self.thread.finished_signal.connect(self.thread_finished)
        self.thread.start()

    def stop_processing(self):
        if self.thread and self.thread.isRunning():
            self.update_status("Stopping process...")
            self.thread.stop()

    def thread_finished(self):
        self.btn_load_video.setEnabled(True)
        self.btn_start_webcam.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.video_label.clear()
        self.video_label.setText("Process stopped. Please load a video or start webcam.")
        self.status_bar.showMessage("System Ready")
        self.thread = None
        self.refresh_memory_table()
        self.refresh_logs()

    def closeEvent(self, event):
        self.stop_processing()
        if self.pipeline:
            self.pipeline.cleanup()
        if self.db:
            self.db.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MemoryTrackApp()
    window.show()
    sys.exit(app.exec())
