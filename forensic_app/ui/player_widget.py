"""
Forensic Video Player Widget
Renders annotated live video, skeleton trees, and garment masks with playback controls.
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QComboBox, QCheckBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


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
        self.current_video_path = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Video Canvas
        self.canvas = QLabel("Select a video file or connect a webcam to initiate forensic surveillance tracking.")
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setStyleSheet("""
            QLabel {
                background-color: #090d16;
                border: 2px dashed #1e293b;
                border-radius: 8px;
                color: #64748b;
                font-size: 14px;
                font-weight: 500;
            }
        """)
        self.canvas.setMinimumSize(720, 440)
        layout.addWidget(self.canvas, stretch=1)

        # Control Bar
        control_frame = QFrame()
        control_frame.setStyleSheet("background-color: #0f172a; border-radius: 6px; padding: 4px;")
        ctrl_layout = QHBoxLayout(control_frame)
        ctrl_layout.setContentsMargins(6, 4, 6, 4)

        # Buttons
        self.btn_load_video = QPushButton("Load Video File")
        self.btn_load_video.clicked.connect(self._on_load_video)

        self.btn_webcam = QPushButton("Start Webcam")
        self.btn_webcam.clicked.connect(self._on_start_webcam)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self._on_pause_toggle)
        self.btn_pause.setEnabled(False)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btn_danger")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)

        # Camera selection dropdown
        self.combo_cam = QComboBox()
        self.combo_cam.addItems(["CAM_01 (Entrance)", "CAM_02 (Corridor)", "CAM_03 (Perimeter)", "CAM_04 (Parking)"])

        ctrl_layout.addWidget(self.btn_load_video)
        ctrl_layout.addWidget(self.btn_webcam)
        ctrl_layout.addWidget(self.btn_pause)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addSpacing(16)
        ctrl_layout.addWidget(QLabel("Stream Channel:"))
        ctrl_layout.addWidget(self.combo_cam)
        ctrl_layout.addStretch()

        layout.addWidget(control_frame)

    def update_frame(self, cv_frame: np.ndarray) -> None:
        """Render processed OpenCV BGR frame onto the Qt Canvas."""
        if cv_frame is None or cv_frame.size == 0:
            return

        rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale preserving aspect ratio
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.canvas.width(), self.canvas.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.canvas.setPixmap(scaled_pixmap)

    def _on_load_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Surveillance Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if file_path:
            self.current_video_path = file_path
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.btn_load_video.setEnabled(False)
            self.btn_webcam.setEnabled(False)
            self.is_playing = True
            self.is_paused = False
            self.play_requested.emit("video", file_path)

    def _on_start_webcam(self) -> None:
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_load_video.setEnabled(False)
        self.btn_webcam.setEnabled(False)
        self.is_playing = True
        self.is_paused = False
        self.play_requested.emit("webcam", "0")

    def _on_pause_toggle(self) -> None:
        if self.is_paused:
            self.btn_pause.setText("Pause")
            self.is_paused = False
            self.resume_requested.emit()
        else:
            self.btn_pause.setText("Resume")
            self.is_paused = True
            self.pause_requested.emit()

    def _on_stop(self) -> None:
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(False)
        self.btn_load_video.setEnabled(True)
        self.btn_webcam.setEnabled(True)
        self.is_playing = False
        self.is_paused = False
        self.canvas.clear()
        self.canvas.setText("Surveillance pipeline stopped. Ready to load next feed.")
        self.stop_requested.emit()

    def reset_ui_state(self) -> None:
        """Reset buttons when stream reaches end."""
        self._on_stop()
