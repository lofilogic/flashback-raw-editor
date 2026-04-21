"""
Advanced Settings panel for tuning effects in real-time.
Toggle visibility with F12.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox,
    QGroupBox,
)
from PySide6.QtCore import Qt, QSettings

from core.config import (
    DebugConfig,
    HALATION_THRESHOLD, HALATION_BLUR_RADIUS, HALATION_STRENGTH,
    CHROMATIC_ABERRATION_STRENGTH, CHROMATIC_ABERRATION_STEPS,
    SOFTNESS_SIGMA, GRAIN_STRENGTH, SHARPEN_STRENGTH, SHARPEN_RADIUS,
    CNR_SIGMA, HIGHLIGHT_DESAT_THRESHOLD_L, HIGHLIGHT_DESAT_ROLLOFF_L,
    HIGHLIGHT_DESAT_SIGMA, DITHER_STRENGTH,
)


class DebugPanel(QWidget):
    """Advanced Settings panel for tuning effects in real-time."""

    def __init__(self, processor, parent=None):
        super().__init__(parent, Qt.Tool)  # Tool window stays on top
        self.processor = processor
        self.parent_editor = parent
        self.setWindowTitle("Advanced Settings (F12 to toggle)")
        self.setMinimumWidth(380)
        self.setMaximumWidth(450)
        self.resize(450, 900)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # Header
        header = QHBoxLayout()
        self.btn_reset = QPushButton("↺ Reset All")
        self.btn_reset.setStyleSheet("QPushButton { background-color: #3d3d3d; color: #d0d0d0; border: none; padding: 6px 12px; border-radius: 4px; } QPushButton:hover { background-color: #4a4a4a; }")
        self.btn_reset.clicked.connect(self.reset_all)
        header.addWidget(self.btn_reset)

        self.btn_save_defaults = QPushButton("Save as Defaults")
        self.btn_save_defaults.setToolTip("Save current settings as startup defaults")
        self.btn_save_defaults.setStyleSheet("QPushButton { background-color: #3d3d3d; color: #d0d0d0; border: none; padding: 6px 12px; border-radius: 4px; } QPushButton:hover { background-color: #4a4a4a; }")
        self.btn_save_defaults.clicked.connect(self.save_defaults)
        header.addWidget(self.btn_save_defaults)

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

        # --- Experimental Group ---
        exp_group = QGroupBox("Experimental")
        exp_group.setStyleSheet("QGroupBox { color: #FF8A35; font-weight: bold; border: 1px solid #555; border-radius: 6px; margin-top: 8px; padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        exp_layout = QVBoxLayout(exp_group)
        exp_layout.setSpacing(6)

        self.chk_full_res_export = QCheckBox("Reprocess at full resolution on export")
        self.chk_full_res_export.setToolTip(
            "Re-develops the RAW at full sensor resolution before exporting.\n"
            "Pixel-sized effects (softness, sharpen radius, halation, CNR) are\n"
            "scaled 2× so the look matches preview. Export is noticeably slower."
        )
        self.chk_full_res_export.stateChanged.connect(self.update_config)
        exp_layout.addWidget(self.chk_full_res_export)

        exp_note = QLabel("Export will take ~4× longer. Experimental.")
        exp_note.setStyleSheet("color: #888; font-size: 10px;")
        exp_note.setWordWrap(True)
        exp_layout.addWidget(exp_note)

        form_layout.addWidget(exp_group)
        form_layout.addStretch()

        scroll.setWidget(container)
        layout.addWidget(scroll)

        self.status_label = QLabel("Ready - press F12 to close")
        self.status_label.setStyleSheet("color: #626262; font-size: 11px; padding-top: 8px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.load_defaults()

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
        DebugConfig.experimental_full_res_export = self.chk_full_res_export.isChecked()

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

    def save_defaults(self):
        """Persist current DebugConfig values as startup defaults via QSettings."""
        self.update_config()
        s = QSettings("Flashback", "Editor")
        s.beginGroup("debug_defaults")
        s.setValue("enable_halation",          DebugConfig.enable_halation)
        s.setValue("enable_ca",                DebugConfig.enable_chromatic_aberration)
        s.setValue("enable_softness",          DebugConfig.enable_softness)
        s.setValue("enable_grain",             DebugConfig.enable_grain)
        s.setValue("enable_sharpen",           DebugConfig.enable_sharpen)
        s.setValue("enable_cnr",               DebugConfig.enable_cnr)
        s.setValue("enable_lut",               DebugConfig.enable_lut)
        s.setValue("enable_dither",            DebugConfig.enable_pre_lut_dither)
        s.setValue("enable_highlight_desat",   DebugConfig.enable_highlight_desat)
        s.setValue("halation_threshold",       DebugConfig.halation_threshold)
        s.setValue("halation_blur_radius",     DebugConfig.halation_blur_radius)
        s.setValue("halation_strength",        DebugConfig.halation_strength)
        s.setValue("ca_strength",              DebugConfig.ca_strength)
        s.setValue("ca_steps",                 DebugConfig.ca_steps)
        s.setValue("softness_sigma",           DebugConfig.softness_sigma)
        s.setValue("grain_strength",           DebugConfig.grain_strength)
        s.setValue("sharpen_strength",         DebugConfig.sharpen_strength)
        s.setValue("sharpen_radius",           DebugConfig.sharpen_radius)
        s.setValue("cnr_sigma",                DebugConfig.cnr_sigma)
        s.setValue("hd_threshold_L",           DebugConfig.highlight_desat_threshold_L)
        s.setValue("hd_rolloff_L",             DebugConfig.highlight_desat_rolloff_L)
        s.setValue("hd_sigma",                 DebugConfig.highlight_desat_sigma)
        s.setValue("dither_strength",          DebugConfig.pre_lut_dither_strength)
        s.setValue("experimental_full_res_export", DebugConfig.experimental_full_res_export)
        s.endGroup()
        self.status_label.setText("Defaults saved — will apply on next launch.")

    def load_defaults(self):
        """Load saved defaults from QSettings into DebugConfig and update UI."""
        s = QSettings("Flashback", "Editor")
        s.beginGroup("debug_defaults")
        if not s.childKeys():
            s.endGroup()
            return  # Nothing saved yet — keep module defaults

        def b(key, fallback): return s.value(key, fallback, type=bool)
        def f(key, fallback): return s.value(key, fallback, type=float)
        def i(key, fallback): return s.value(key, fallback, type=int)

        DebugConfig.enable_halation              = b("enable_halation",        DebugConfig.enable_halation)
        DebugConfig.enable_chromatic_aberration  = b("enable_ca",              DebugConfig.enable_chromatic_aberration)
        DebugConfig.enable_softness              = b("enable_softness",        DebugConfig.enable_softness)
        DebugConfig.enable_grain                 = b("enable_grain",           DebugConfig.enable_grain)
        DebugConfig.enable_sharpen               = b("enable_sharpen",         DebugConfig.enable_sharpen)
        DebugConfig.enable_cnr                   = b("enable_cnr",             DebugConfig.enable_cnr)
        DebugConfig.enable_lut                   = b("enable_lut",             DebugConfig.enable_lut)
        DebugConfig.enable_pre_lut_dither        = b("enable_dither",          DebugConfig.enable_pre_lut_dither)
        DebugConfig.enable_highlight_desat       = b("enable_highlight_desat", DebugConfig.enable_highlight_desat)
        DebugConfig.halation_threshold           = f("halation_threshold",     DebugConfig.halation_threshold)
        DebugConfig.halation_blur_radius         = f("halation_blur_radius",   DebugConfig.halation_blur_radius)
        DebugConfig.halation_strength            = f("halation_strength",      DebugConfig.halation_strength)
        DebugConfig.ca_strength                  = f("ca_strength",            DebugConfig.ca_strength)
        DebugConfig.ca_steps                     = i("ca_steps",               DebugConfig.ca_steps)
        DebugConfig.softness_sigma               = f("softness_sigma",         DebugConfig.softness_sigma)
        DebugConfig.grain_strength               = f("grain_strength",         DebugConfig.grain_strength)
        DebugConfig.sharpen_strength             = f("sharpen_strength",       DebugConfig.sharpen_strength)
        DebugConfig.sharpen_radius               = f("sharpen_radius",         DebugConfig.sharpen_radius)
        DebugConfig.cnr_sigma                    = f("cnr_sigma",              DebugConfig.cnr_sigma)
        DebugConfig.highlight_desat_threshold_L  = f("hd_threshold_L",        DebugConfig.highlight_desat_threshold_L)
        DebugConfig.highlight_desat_rolloff_L    = f("hd_rolloff_L",           DebugConfig.highlight_desat_rolloff_L)
        DebugConfig.highlight_desat_sigma        = f("hd_sigma",               DebugConfig.highlight_desat_sigma)
        DebugConfig.pre_lut_dither_strength      = f("dither_strength",        DebugConfig.pre_lut_dither_strength)
        DebugConfig.experimental_full_res_export = b("experimental_full_res_export", DebugConfig.experimental_full_res_export)
        s.endGroup()

        # Block signals while syncing UI so that partial updates don't
        # trigger update_config() which would overwrite not-yet-restored values.
        widgets = [
            self.chk_halation, self.chk_ca, self.chk_softness, self.chk_grain,
            self.chk_sharpen, self.chk_cnr, self.chk_lut, self.chk_dither,
            self.chk_full_res_export,
            self.chk_highlight_desat, self.spin_halation_thresh, self.spin_halation_blur,
            self.spin_halation_str, self.spin_ca_str, self.spin_ca_steps,
            self.spin_softness, self.spin_grain, self.spin_sharpen_str,
            self.spin_sharpen_rad, self.spin_cnr, self.spin_hd_thresh,
            self.spin_hd_rolloff, self.spin_hd_sigma, self.spin_dither,
        ]
        for w in widgets:
            w.blockSignals(True)

        self.chk_halation.setChecked(DebugConfig.enable_halation)
        self.chk_ca.setChecked(DebugConfig.enable_chromatic_aberration)
        self.chk_softness.setChecked(DebugConfig.enable_softness)
        self.chk_grain.setChecked(DebugConfig.enable_grain)
        self.chk_sharpen.setChecked(DebugConfig.enable_sharpen)
        self.chk_cnr.setChecked(DebugConfig.enable_cnr)
        self.chk_lut.setChecked(DebugConfig.enable_lut)
        self.chk_dither.setChecked(DebugConfig.enable_pre_lut_dither)
        self.chk_full_res_export.setChecked(DebugConfig.experimental_full_res_export)
        self.chk_highlight_desat.setChecked(DebugConfig.enable_highlight_desat)
        self.spin_halation_thresh.setValue(DebugConfig.halation_threshold)
        self.spin_halation_blur.setValue(DebugConfig.halation_blur_radius)
        self.spin_halation_str.setValue(DebugConfig.halation_strength)
        self.spin_ca_str.setValue(DebugConfig.ca_strength)
        self.spin_ca_steps.setValue(DebugConfig.ca_steps)
        self.spin_softness.setValue(DebugConfig.softness_sigma)
        self.spin_grain.setValue(DebugConfig.grain_strength)
        self.spin_sharpen_str.setValue(DebugConfig.sharpen_strength)
        self.spin_sharpen_rad.setValue(DebugConfig.sharpen_radius)
        self.spin_cnr.setValue(DebugConfig.cnr_sigma)
        self.spin_hd_thresh.setValue(DebugConfig.highlight_desat_threshold_L)
        self.spin_hd_rolloff.setValue(DebugConfig.highlight_desat_rolloff_L)
        self.spin_hd_sigma.setValue(DebugConfig.highlight_desat_sigma)
        self.spin_dither.setValue(DebugConfig.pre_lut_dither_strength)

        for w in widgets:
            w.blockSignals(False)

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

