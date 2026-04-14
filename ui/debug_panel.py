"""
Floating debug panel for tuning effects in real-time.
Toggle visibility with F12.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox,
    QGroupBox,
)
from PySide6.QtCore import Qt

from core.config import (
    DebugConfig,
    HALATION_THRESHOLD, HALATION_BLUR_RADIUS, HALATION_STRENGTH,
    CHROMATIC_ABERRATION_STRENGTH, CHROMATIC_ABERRATION_STEPS,
    SOFTNESS_SIGMA, GRAIN_STRENGTH, SHARPEN_STRENGTH, SHARPEN_RADIUS,
    CNR_SIGMA, HIGHLIGHT_DESAT_THRESHOLD_L, HIGHLIGHT_DESAT_ROLLOFF_L,
    HIGHLIGHT_DESAT_SIGMA, DITHER_STRENGTH,
)


class DebugPanel(QWidget):
    """Floating debug panel for tuning effects in real-time."""

    def __init__(self, processor, parent=None):
        super().__init__(parent, Qt.Tool)  # Tool window stays on top
        self.processor = processor
        self.parent_editor = parent
        self.setWindowTitle("Flashback Debug Panel (F12 to toggle)")
        self.setMinimumWidth(380)
        self.setMaximumWidth(450)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # Header
        header = QHBoxLayout()
        self.btn_reset = QPushButton("↺ Reset All")
        self.btn_reset.setStyleSheet("QPushButton { background-color: #3d3d3d; color: #d0d0d0; border: none; padding: 6px 12px; border-radius: 4px; } QPushButton:hover { background-color: #4a4a4a; }")
        self.btn_reset.clicked.connect(self.reset_all)
        header.addWidget(self.btn_reset)

        self.btn_reload = QPushButton("Reload Image")
        self.btn_reload.setToolTip("Reload to apply baked effects (Halation, CA, CNR)")
        self.btn_reload.setStyleSheet("QPushButton { background-color: #FF8A35; color: #1a1a1a; font-weight: bold; border: none; padding: 6px 12px; border-radius: 4px; } QPushButton:hover { background-color: #ff9a4f; }")
        self.btn_reload.clicked.connect(self.reload_image)
        header.addWidget(self.btn_reload)
        header.addStretch()
        layout.addLayout(header)

        # Scroll area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        container = QWidget()
        form_layout = QVBoxLayout(container)
        form_layout.setSpacing(15)
        form_layout.setAlignment(Qt.AlignTop)

        # LUT Selection
        lut_layout = QHBoxLayout()
        prefix_label = QLabel("Custom LUT:")
        self.lut_label = QLabel("Current LUT: Default")
        self.btn_load_lut = QPushButton("Load .cube LUT")
        self.btn_load_lut.clicked.connect(self.parent_editor._load_custom_lut)
        lut_layout.addWidget(prefix_label)
        lut_layout.addWidget(self.lut_label)
        lut_layout.addWidget(self.btn_load_lut)
        form_layout.addLayout(lut_layout)

        # --- Baked Effects Group ---
        baked_group = QGroupBox("Baked Effects (Require Image Reload)")
        baked_group.setStyleSheet("QGroupBox { color: #FF8A35; font-weight: bold; border: 1px solid #555; border-radius: 6px; margin-top: 8px; padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        baked_layout = QFormLayout(baked_group)
        baked_layout.setSpacing(8)

        # Halation
        self.chk_halation = QCheckBox("Enable Halation")
        self.chk_halation.setChecked(True)
        self.chk_halation.stateChanged.connect(self.update_config)
        baked_layout.addRow(self.chk_halation)

        self.spin_halation_thresh = self._create_double_spin(0.0, 1.0, HALATION_THRESHOLD, 0.05)
        self.spin_halation_thresh.valueChanged.connect(self.update_config)
        baked_layout.addRow("Threshold:", self.spin_halation_thresh)

        self.spin_halation_blur = self._create_spin(1, 100, HALATION_BLUR_RADIUS)
        self.spin_halation_blur.valueChanged.connect(self.update_config)
        baked_layout.addRow("Blur Radius:", self.spin_halation_blur)

        self.spin_halation_str = self._create_double_spin(0.0, 2.0, HALATION_STRENGTH, 0.1)
        self.spin_halation_str.valueChanged.connect(self.update_config)
        baked_layout.addRow("Strength:", self.spin_halation_str)

        baked_layout.addRow(self._create_separator())

        # CNR
        self.chk_cnr = QCheckBox("Enable Color Noise Reduction")
        self.chk_cnr.setChecked(True)
        self.chk_cnr.stateChanged.connect(self.update_config)
        baked_layout.addRow(self.chk_cnr)

        self.spin_cnr = self._create_double_spin(0.0, 5.0, CNR_SIGMA, 0.1)
        self.spin_cnr.valueChanged.connect(self.update_config)
        baked_layout.addRow("CNR Sigma:", self.spin_cnr)

        baked_layout.addRow(self._create_separator())

        # Highlight Desaturation (Lab)
        self.chk_highlight_desat = QCheckBox("Enable Highlight Desaturation (Lab)")
        self.chk_highlight_desat.setChecked(True)
        self.chk_highlight_desat.stateChanged.connect(self.update_config)
        baked_layout.addRow(self.chk_highlight_desat)

        self.spin_hd_thresh = self._create_double_spin(0.0, 100.0, HIGHLIGHT_DESAT_THRESHOLD_L, 1.0)
        self.spin_hd_thresh.valueChanged.connect(self.update_config)
        baked_layout.addRow("HD Threshold L*:", self.spin_hd_thresh)

        self.spin_hd_rolloff = self._create_double_spin(1.0, 50.0, HIGHLIGHT_DESAT_ROLLOFF_L, 1.0)
        self.spin_hd_rolloff.valueChanged.connect(self.update_config)
        baked_layout.addRow("HD Rolloff L*:", self.spin_hd_rolloff)

        self.spin_hd_sigma = self._create_double_spin(0.0, 20.0, HIGHLIGHT_DESAT_SIGMA, 0.5)
        self.spin_hd_sigma.valueChanged.connect(self.update_config)
        baked_layout.addRow("HD Mask Sigma:", self.spin_hd_sigma)

        form_layout.addWidget(baked_group)

        # --- Real-time Effects Group ---
        live_group = QGroupBox("Real-time Effects")
        live_group.setStyleSheet("QGroupBox { color: #7aa8d9; font-weight: bold; border: 1px solid #555; border-radius: 6px; margin-top: 8px; padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        live_layout = QFormLayout(live_group)
        live_layout.setSpacing(8)

        # Pre-LUT Dither
        self.chk_dither = QCheckBox("Enable Pre-LUT Dither (Anti-Banding)")
        self.chk_dither.setChecked(True)
        self.chk_dither.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_dither)

        self.spin_dither = self._create_double_spin(0.0, 0.01, DITHER_STRENGTH, 0.0005)
        self.spin_dither.valueChanged.connect(self.update_preview)
        live_layout.addRow("Dither Strength:", self.spin_dither)

        # LUT
        self.chk_lut = QCheckBox("Enable LUT")
        self.chk_lut.setChecked(True)
        self.chk_lut.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_lut)

        live_layout.addRow(self._create_separator())

        # Chromatic Aberration
        self.chk_ca = QCheckBox("Enable Chromatic Aberration")
        self.chk_ca.setChecked(True)
        self.chk_ca.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_ca)

        self.spin_ca_str = self._create_double_spin(0.0, 0.02, CHROMATIC_ABERRATION_STRENGTH, 0.001)
        self.spin_ca_str.valueChanged.connect(self.update_preview)
        live_layout.addRow("CA Strength:", self.spin_ca_str)

        self.spin_ca_steps = self._create_spin(1, 10, CHROMATIC_ABERRATION_STEPS)
        self.spin_ca_steps.valueChanged.connect(self.update_preview)
        live_layout.addRow("CA Steps:", self.spin_ca_steps)

        live_layout.addRow(self._create_separator())

        # Softness
        self.chk_softness = QCheckBox("Enable Softness")
        self.chk_softness.setChecked(True)
        self.chk_softness.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_softness)

        self.spin_softness = self._create_double_spin(0.0, 5.0, SOFTNESS_SIGMA, 0.1)
        self.spin_softness.valueChanged.connect(self.update_preview)
        live_layout.addRow("Softness Sigma:", self.spin_softness)

        live_layout.addRow(self._create_separator())

        # Grain
        self.chk_grain = QCheckBox("Enable Grain")
        self.chk_grain.setChecked(True)
        self.chk_grain.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_grain)

        self.spin_grain = self._create_double_spin(0.0, 0.2, GRAIN_STRENGTH, 0.01)
        self.spin_grain.valueChanged.connect(self.update_preview)
        live_layout.addRow("Grain Strength:", self.spin_grain)

        live_layout.addRow(self._create_separator())

        # Sharpen
        self.chk_sharpen = QCheckBox("Enable Sharpen")
        self.chk_sharpen.setChecked(True)
        self.chk_sharpen.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_sharpen)

        self.spin_sharpen_str = self._create_double_spin(0.0, 2.0, SHARPEN_STRENGTH, 0.1)
        self.spin_sharpen_str.valueChanged.connect(self.update_preview)
        live_layout.addRow("Sharpen Strength:", self.spin_sharpen_str)

        self.spin_sharpen_rad = self._create_double_spin(0.1, 5.0, SHARPEN_RADIUS, 0.1)
        self.spin_sharpen_rad.valueChanged.connect(self.update_preview)
        live_layout.addRow("Sharpen Radius:", self.spin_sharpen_rad)

        form_layout.addWidget(live_group)
        form_layout.addStretch()

        scroll.setWidget(container)
        layout.addWidget(scroll)

        self.status_label = QLabel("Ready - F12 to toggle visibility")
        self.status_label.setStyleSheet("color: #626262; font-size: 11px; padding-top: 8px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # ===================================================================
    # HELPERS
    # ===================================================================

    def _create_double_spin(self, min_val, max_val, default, step):
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(3 if step < 0.1 else 2)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.setStyleSheet("QDoubleSpinBox { background-color: #3d3d3d; color: #d0d0d0; border: 1px solid #555; border-radius: 4px; padding: 4px; }")
        return spin

    def _create_spin(self, min_val, max_val, default):
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setStyleSheet("QSpinBox { background-color: #3d3d3d; color: #d0d0d0; border: 1px solid #555; border-radius: 4px; padding: 4px; }")
        return spin

    def _create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #444;")
        return line

    # ===================================================================
    # ACTIONS
    # ===================================================================

    def update_config(self):
        """Update DebugConfig from UI (for baked settings)."""
        DebugConfig.enable_halation = self.chk_halation.isChecked()
        DebugConfig.enable_chromatic_aberration = self.chk_ca.isChecked()
        DebugConfig.enable_cnr = self.chk_cnr.isChecked()

        DebugConfig.halation_threshold = self.spin_halation_thresh.value()
        DebugConfig.halation_blur_radius = self.spin_halation_blur.value()
        DebugConfig.halation_strength = self.spin_halation_str.value()

        DebugConfig.ca_strength = self.spin_ca_str.value()
        DebugConfig.ca_steps = self.spin_ca_steps.value()

        DebugConfig.cnr_sigma = self.spin_cnr.value()

        DebugConfig.enable_highlight_desat = self.chk_highlight_desat.isChecked()
        DebugConfig.highlight_desat_threshold_L = self.spin_hd_thresh.value()
        DebugConfig.highlight_desat_rolloff_L   = self.spin_hd_rolloff.value()
        DebugConfig.highlight_desat_sigma       = self.spin_hd_sigma.value()

        DebugConfig.enable_lut = self.chk_lut.isChecked()
        DebugConfig.enable_softness = self.chk_softness.isChecked()
        DebugConfig.enable_grain = self.chk_grain.isChecked()
        DebugConfig.enable_sharpen = self.chk_sharpen.isChecked()
        DebugConfig.softness_sigma = self.spin_softness.value()
        DebugConfig.grain_strength = self.spin_grain.value()
        DebugConfig.sharpen_strength = self.spin_sharpen_str.value()
        DebugConfig.sharpen_radius = self.spin_sharpen_rad.value()
        DebugConfig.enable_pre_lut_dither = self.chk_dither.isChecked()
        DebugConfig.pre_lut_dither_strength = self.spin_dither.value()

        self.status_label.setText("Config updated. Click 'Reload Image' to apply baked effects.")

    def update_preview(self):
        """Update real-time preview immediately."""
        self.update_config()
        if self.parent_editor:
            self.parent_editor.refresh_from_debug()

    def reload_image(self):
        """Force reload current image."""
        if self.parent_editor:
            self.parent_editor.reload_current_image()
            self.status_label.setText("Image reloaded with new baked settings.")

    def reset_all(self):
        """Reset all parameters to defaults."""
        DebugConfig.reset()

        self.chk_halation.setChecked(True)
        self.spin_halation_thresh.setValue(DebugConfig.halation_threshold)
        self.spin_halation_blur.setValue(DebugConfig.halation_blur_radius)
        self.spin_halation_str.setValue(DebugConfig.halation_strength)

        self.chk_ca.setChecked(True)
        self.spin_ca_str.setValue(DebugConfig.ca_strength)
        self.spin_ca_steps.setValue(DebugConfig.ca_steps)

        self.chk_cnr.setChecked(True)
        self.spin_cnr.setValue(DebugConfig.cnr_sigma)

        self.chk_highlight_desat.setChecked(True)
        self.spin_hd_thresh.setValue(DebugConfig.highlight_desat_threshold_L)
        self.spin_hd_rolloff.setValue(DebugConfig.highlight_desat_rolloff_L)
        self.spin_hd_sigma.setValue(DebugConfig.highlight_desat_sigma)

        self.chk_lut.setChecked(True)
        self.chk_softness.setChecked(True)
        self.spin_softness.setValue(DebugConfig.softness_sigma)
        self.chk_grain.setChecked(True)
        self.spin_grain.setValue(DebugConfig.grain_strength)
        self.chk_sharpen.setChecked(True)
        self.spin_sharpen_str.setValue(DebugConfig.sharpen_strength)
        self.spin_sharpen_rad.setValue(DebugConfig.sharpen_radius)

        self.status_label.setText("Reset to defaults.")
        if self.parent_editor:
            self.parent_editor.refresh_from_debug()

