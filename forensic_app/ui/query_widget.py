"""
Forensic Query Builder Widget
Glass panel to search suspects and missing persons by clothing colours, accessories and soft biometrics.
"""

import logging
from typing import Dict, List

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QVBoxLayout, QWidget
)

from forensic_app.core.catalog import ACCESSORY_LABELS
from forensic_app.ui.effects import GlassPanel, make_transparent_scroll, set_role, set_variant, swatch_icon
from forensic_app.utils.color_naming import FORENSIC_COLOR_PALETTE

logger = logging.getLogger(__name__)

ANY_OPTION = "Any"
BARE_LEGS_OPTION = "None / Bare Legs"


def _palette_entries() -> List[tuple]:
    """(name, hex) pairs read dynamically from the forensic colour palette."""
    entries = []
    for name, data in FORENSIC_COLOR_PALETTE.items():
        hex_c = data.get("hex") if isinstance(data, dict) else None
        entries.append((str(name), hex_c))
    return entries


class QueryWidget(QWidget):
    """Visual attribute and biometric query constructor."""

    # Search signal emitting query parameters dictionary
    search_requested = pyqtSignal(dict)
    reset_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.accessory_checks: Dict[str, QCheckBox] = {}
        self._init_ui()

    def _section(self, title: str, hint: str = "") -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(set_role(QLabel(title), "section"))
        head.addStretch()
        if hint:
            head.addWidget(set_role(QLabel(hint), "hint"))
        box.addLayout(head)
        return box

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: rgba(255, 255, 255, 0.08); border: none;")
        return line

    def _color_combo(self, extra: List[str]) -> QComboBox:
        combo = QComboBox()
        combo.setIconSize(QSize(14, 14))
        combo.addItem(swatch_icon(None), ANY_OPTION)
        for item in extra:
            combo.addItem(swatch_icon(None), item)
        for name, hex_c in _palette_entries():
            combo.addItem(swatch_icon(hex_c), name)
        combo.setMaxVisibleItems(14)
        combo.setMinimumWidth(180)
        return combo

    def _init_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.panel = GlassPanel(radius=18)
        outer.addWidget(self.panel)
        main_layout = QVBoxLayout(self.panel)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(12)

        head = QVBoxLayout()
        head.setSpacing(2)
        head.addWidget(set_role(QLabel("Forensic Query Builder"), "title"))
        head.addWidget(set_role(QLabel("Search stored identities by what they wore and carried."), "subtitle"))
        main_layout.addLayout(head)
        main_layout.addWidget(self._divider())

        # Filter sections scroll on short screens; the action buttons stay pinned below
        self.filters_scroll = QScrollArea()
        self.filters_scroll.setWidgetResizable(True)
        self.filters_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        filters = QWidget()
        panel_layout = main_layout
        main_layout = QVBoxLayout(filters)
        main_layout.setContentsMargins(0, 0, 6, 0)
        main_layout.setSpacing(12)

        # 1. Garment Colors
        colors = self._section("CLOTHING COLOURS", "CIEDE2000 matching")
        c_layout = QGridLayout()
        c_layout.setHorizontalSpacing(12)
        c_layout.setVerticalSpacing(8)
        c_layout.addWidget(set_role(QLabel("Upper garment"), "field"), 0, 0)
        self.combo_upper = self._color_combo([])
        c_layout.addWidget(self.combo_upper, 0, 1)
        c_layout.addWidget(set_role(QLabel("Lower garment"), "field"), 1, 0)
        self.combo_lower = self._color_combo([BARE_LEGS_OPTION])
        c_layout.addWidget(self.combo_lower, 1, 1)
        c_layout.setColumnStretch(1, 1)
        colors.addLayout(c_layout)
        main_layout.addLayout(colors)
        main_layout.addWidget(self._divider())

        # 2. Carried Accessories (generated from the canonical catalog)
        accs = self._section("ACCESSORIES", "all selected must match")
        acc_layout = QGridLayout()
        acc_layout.setHorizontalSpacing(14)
        acc_layout.setVerticalSpacing(2)
        for idx, (name, label) in enumerate(ACCESSORY_LABELS.items()):
            chk = QCheckBox(label)
            chk.setProperty("accessory", name)
            chk.setToolTip(f"Require '{name}' on the identity")
            self.accessory_checks[name] = chk
            acc_layout.addWidget(chk, idx // 2, idx % 2)
        accs.addLayout(acc_layout)
        main_layout.addLayout(accs)
        main_layout.addWidget(self._divider())

        # Legacy attribute aliases (kept for scripts that toggle these directly)
        self.chk_backpack = self.accessory_checks.get("backpack")
        self.chk_handbag = self.accessory_checks.get("handbag")
        self.chk_umbrella = self.accessory_checks.get("umbrella")
        self.chk_cap = self.accessory_checks.get("cap")
        self.chk_suitcase = self.accessory_checks.get("suitcase")
        self.chk_bottle = self.accessory_checks.get("bottle")

        # 3. Soft Biometrics
        bio = self._section("SOFT BIOMETRICS", "optional")
        bio_layout = QGridLayout()
        bio_layout.setHorizontalSpacing(12)
        bio_layout.setVerticalSpacing(8)
        bio_layout.addWidget(set_role(QLabel("Min torso-to-leg ratio"), "field"), 0, 0)
        self.spin_ratio = QDoubleSpinBox()
        self.spin_ratio.setRange(0.0, 2.5)
        self.spin_ratio.setSingleStep(0.05)
        self.spin_ratio.setValue(0.0)
        bio_layout.addWidget(self.spin_ratio, 0, 1)

        bio_layout.addWidget(set_role(QLabel("Min gait cadence (Hz)"), "field"), 1, 0)
        self.spin_cadence = QDoubleSpinBox()
        self.spin_cadence.setRange(0.0, 3.5)
        self.spin_cadence.setSingleStep(0.1)
        self.spin_cadence.setValue(0.0)
        bio_layout.addWidget(self.spin_cadence, 1, 1)
        bio_layout.setColumnStretch(1, 1)
        bio.addLayout(bio_layout)
        main_layout.addLayout(bio)
        main_layout.addStretch()

        self.filters_scroll.setWidget(filters)
        make_transparent_scroll(self.filters_scroll)
        panel_layout.addWidget(self.filters_scroll, stretch=1)
        main_layout = panel_layout

        # 4. Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        self.btn_search = set_variant(QPushButton("Run Forensic Search"), "primary")
        self.btn_search.setMinimumHeight(40)
        self.btn_search.clicked.connect(self._on_search)

        self.btn_reset = set_variant(QPushButton("Reset Filters"), "ghost")
        self.btn_reset.setMinimumHeight(40)
        self.btn_reset.clicked.connect(self._on_reset)

        btn_layout.addWidget(self.btn_search, stretch=2)
        btn_layout.addWidget(self.btn_reset, stretch=1)
        main_layout.addLayout(btn_layout)

    def selected_accessories(self) -> List[str]:
        """Canonical names of the checked accessories (catalog order)."""
        return [name for name, chk in self.accessory_checks.items() if chk.isChecked()]

    def _on_search(self) -> None:
        query_params = {
            "upper_color": self.combo_upper.currentText(),
            "lower_color": self.combo_lower.currentText(),
            "accessories": self.selected_accessories(),
            "min_torso_leg_ratio": self.spin_ratio.value(),
            "min_cadence": self.spin_cadence.value()
        }
        self.search_requested.emit(query_params)

    def _on_reset(self) -> None:
        self.combo_upper.setCurrentIndex(0)
        self.combo_lower.setCurrentIndex(0)
        for chk in self.accessory_checks.values():
            chk.setChecked(False)
        self.spin_ratio.setValue(0.0)
        self.spin_cadence.setValue(0.0)
        self.reset_requested.emit()
