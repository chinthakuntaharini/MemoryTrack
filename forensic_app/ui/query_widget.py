"""
Forensic Query Builder Widget
Interactive panel to search suspects and missing persons by visual and biometric traits.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QCheckBox,
    QGroupBox, QDoubleSpinBox
)
from PyQt6.QtCore import pyqtSignal

from forensic_app.utils.color_naming import FORENSIC_COLOR_PALETTE


class QueryWidget(QWidget):
    """Visual attribute and biometric query constructor."""

    # Search signal emitting query parameters dictionary
    search_requested = pyqtSignal(dict)
    reset_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(14)

        # 1. Garment Colors Group
        color_group = QGroupBox("Clothing Color Attributes (CIELAB Space)")
        c_layout = QGridLayout(color_group)

        c_layout.addWidget(QLabel("Upper Garment:"), 0, 0)
        self.combo_upper = QComboBox()
        self.combo_upper.addItem("Any")
        for color_name in FORENSIC_COLOR_PALETTE.keys():
            self.combo_upper.addItem(color_name)
        c_layout.addWidget(self.combo_upper, 0, 1)

        c_layout.addWidget(QLabel("Lower Garment:"), 1, 0)
        self.combo_lower = QComboBox()
        self.combo_lower.addItem("Any")
        self.combo_lower.addItem("None / Bare Legs")
        for color_name in FORENSIC_COLOR_PALETTE.keys():
            self.combo_lower.addItem(color_name)
        c_layout.addWidget(self.combo_lower, 1, 1)

        main_layout.addWidget(color_group)

        # 2. Carried Accessories Group
        acc_group = QGroupBox("Carried Accessories")
        acc_layout = QGridLayout(acc_group)

        self.chk_backpack = QCheckBox("Backpack")
        self.chk_handbag = QCheckBox("Handbag")
        self.chk_umbrella = QCheckBox("Umbrella")
        self.chk_cap = QCheckBox("Headwear / Cap")
        self.chk_suitcase = QCheckBox("Suitcase")
        self.chk_bottle = QCheckBox("Bottle")

        acc_layout.addWidget(self.chk_backpack, 0, 0)
        acc_layout.addWidget(self.chk_handbag, 0, 1)
        acc_layout.addWidget(self.chk_umbrella, 1, 0)
        acc_layout.addWidget(self.chk_cap, 1, 1)
        acc_layout.addWidget(self.chk_suitcase, 2, 0)
        acc_layout.addWidget(self.chk_bottle, 2, 1)

        main_layout.addWidget(acc_group)

        # 3. Soft Biometrics Group
        bio_group = QGroupBox("Anthropometric Ratios & Gait")
        bio_layout = QGridLayout(bio_group)

        bio_layout.addWidget(QLabel("Min Torso-to-Leg Ratio:"), 0, 0)
        self.spin_ratio = QDoubleSpinBox()
        self.spin_ratio.setRange(0.0, 2.5)
        self.spin_ratio.setSingleStep(0.05)
        self.spin_ratio.setValue(0.0)
        bio_layout.addWidget(self.spin_ratio, 0, 1)

        bio_layout.addWidget(QLabel("Min Cadence (Hz):"), 1, 0)
        self.spin_cadence = QDoubleSpinBox()
        self.spin_cadence.setRange(0.0, 3.5)
        self.spin_cadence.setSingleStep(0.1)
        self.spin_cadence.setValue(0.0)
        bio_layout.addWidget(self.spin_cadence, 1, 1)

        main_layout.addWidget(bio_group)

        # 4. Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_search = QPushButton("Execute Forensic Search")
        self.btn_search.setStyleSheet("background-color: #0284c7; font-size: 14px; padding: 10px;")
        self.btn_search.clicked.connect(self._on_search)

        self.btn_reset = QPushButton("Reset Filter")
        self.btn_reset.setStyleSheet("background-color: #334155;")
        self.btn_reset.clicked.connect(self._on_reset)

        btn_layout.addWidget(self.btn_search, stretch=2)
        btn_layout.addWidget(self.btn_reset, stretch=1)
        main_layout.addLayout(btn_layout)

        main_layout.addStretch()

    def _on_search(self) -> None:
        accessories = []
        if self.chk_backpack.isChecked():
            accessories.append("backpack")
        if self.chk_handbag.isChecked():
            accessories.append("handbag")
        if self.chk_umbrella.isChecked():
            accessories.append("umbrella")
        if self.chk_cap.isChecked():
            accessories.append("cap")
        if self.chk_suitcase.isChecked():
            accessories.append("suitcase")
        if self.chk_bottle.isChecked():
            accessories.append("bottle")

        query_params = {
            "upper_color": self.combo_upper.currentText(),
            "lower_color": self.combo_lower.currentText(),
            "accessories": accessories,
            "min_torso_leg_ratio": self.spin_ratio.value(),
            "min_cadence": self.spin_cadence.value()
        }
        self.search_requested.emit(query_params)

    def _on_reset(self) -> None:
        self.combo_upper.setCurrentIndex(0)
        self.combo_lower.setCurrentIndex(0)
        self.chk_backpack.setChecked(False)
        self.chk_handbag.setChecked(False)
        self.chk_umbrella.setChecked(False)
        self.chk_cap.setChecked(False)
        self.chk_suitcase.setChecked(False)
        self.chk_bottle.setChecked(False)
        self.spin_ratio.setValue(0.0)
        self.spin_cadence.setValue(0.0)
        self.reset_requested.emit()
