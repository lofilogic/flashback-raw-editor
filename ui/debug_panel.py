"""
Advanced Settings panel for tuning effects in real-time.
Toggle visibility with F12.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox,
    QGroupBox, QLineEdit,
)
from PySide6.QtCore import Qt

from core.config import (
    VibeConfig, VIBE_FIELD_NAMES,
    HALATION_THRESHOLD_STOPS, HALATION_BLUR_RADIUS, HALATION_STRENGTH_PCT,
    HALATION_WARMTH_PCT,
    CA_PIXELS, CA_STEPS, CA_BLUE_BLUR, CA_ZOOM_BLUR_PCT,
    SOFTNESS_SIGMA, GRAIN_STRENGTH_PCT, SHARPEN_STRENGTH_PCT, SHARPEN_RADIUS,
    CNR_AMOUNT_PCT, CNR_DESPIKE_PCT, CNR_DESPIKE_BIAS_PCT,
    VIGNETTE_STRENGTH_PCT, VIGNETTE_COLOR_PCT, VIGNETTE_CURVE,
    BLOOM_STRENGTH_PCT, BLOOM_THRESHOLD_STOPS,
)


def _current_vibe(parent_editor) -> VibeConfig:
    """Tiny helper: return parent_editor.current_vibe (the active VibeConfig)."""
    return parent_editor.current_vibe


class DebugPanel(QWidget):
    """Advanced Settings panel for tuning effects in real-time."""

    def __init__(self, processor, parent=None):
        super().__init__(parent, Qt.Tool)  # Tool window stays on top
        self.processor = processor
        self.parent_editor = parent
        self.setWindowTitle("Advanced Settings (F12 to toggle)")
        self.setMinimumWidth(380)
        self.resize(620, 900)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # Header — vibe-scoped controls
        self.vibe_header_label = QLabel("Vibe defaults")
        self.vibe_header_label.setStyleSheet("color: #d0d0d0; font-weight: bold;")
        layout.addWidget(self.vibe_header_label)

        btn_style = "QPushButton { background-color: #3d3d3d; color: #d0d0d0; border: none; padding: 6px 12px; border-radius: 4px; } QPushButton:hover { background-color: #4a4a4a; }"

        header = QHBoxLayout()
        self.btn_save_vibe = QPushButton("Save")
        self.btn_save_vibe.setToolTip(
            "Save the current settings as the saved defaults for this vibe.\n"
            "These are loaded on startup and when you Reset to Saved."
        )
        self.btn_save_vibe.setStyleSheet(btn_style)
        self.btn_save_vibe.clicked.connect(self._on_save_vibe)
        header.addWidget(self.btn_save_vibe)

        self.btn_reset_saved = QPushButton("Reset to Saved")
        self.btn_reset_saved.setToolTip(
            "Discard session changes and restore your saved defaults for this vibe.\n"
            "If you have not saved anything, this restores the factory defaults."
        )
        self.btn_reset_saved.setStyleSheet(btn_style)
        self.btn_reset_saved.clicked.connect(self._on_reset_saved)
        header.addWidget(self.btn_reset_saved)

        self.btn_reset_factory = QPushButton("Reset to Factory")
        self.btn_reset_factory.setToolTip(
            "Discard your saved and session changes for this vibe and restore the\n"
            "bundled factory defaults. Only affects the current vibe."
        )
        self.btn_reset_factory.setStyleSheet(btn_style)
        self.btn_reset_factory.clicked.connect(self._on_reset_factory)
        header.addWidget(self.btn_reset_factory)

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

        # --- Baked Effects Group (only halation truly bakes into the intermediate) ---
        baked_group = QGroupBox("Baked Effects (Require Image Reload)")
        baked_group.setStyleSheet("QGroupBox { color: #FF8A35; font-weight: bold; border: 1px solid #555; border-radius: 6px; margin-top: 8px; padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        baked_layout = QFormLayout(baked_group)
        baked_layout.setSpacing(8)

        # Halation
        self.chk_halation = QCheckBox("Enable Halation")
        self.chk_halation.setChecked(True)
        self.chk_halation.stateChanged.connect(self.update_config)
        baked_layout.addRow(self.chk_halation)

        self.spin_halation_thresh = self._create_double_spin(-2.0, 8.0, HALATION_THRESHOLD_STOPS, 0.25, suffix=" EV")
        self.spin_halation_thresh.setToolTip(
            "Stops above 18% middle grey. 0 = middle grey, +5 ≈ scene white,\n"
            "+4 (default) targets specular highlights, +6 only the very brightest."
        )
        self.spin_halation_thresh.valueChanged.connect(self.update_config)
        baked_layout.addRow("Threshold:", self.spin_halation_thresh)

        self.spin_halation_blur = self._create_spin(1, 100, int(HALATION_BLUR_RADIUS))
        self.spin_halation_blur.setSuffix(" px")
        self.spin_halation_blur.valueChanged.connect(self.update_config)
        baked_layout.addRow("Blur Radius:", self.spin_halation_blur)

        self.spin_halation_str = self._create_double_spin(0.0, 300.0, HALATION_STRENGTH_PCT, 5.0, suffix=" %")
        self.spin_halation_str.valueChanged.connect(self.update_config)
        baked_layout.addRow("Strength:", self.spin_halation_str)

        self.spin_halation_warmth = self._create_double_spin(0.0, 300.0, HALATION_WARMTH_PCT, 5.0, suffix=" %")
        self.spin_halation_warmth.setToolTip(
            "Halo chroma. 100% = physical red-orange (and the legacy average "
            "colour); 0% = colourless glow; >100% pushes toward the saturated "
            "no-remjet / CineStill halo. Always reddens outward.")
        self.spin_halation_warmth.valueChanged.connect(self.update_config)
        baked_layout.addRow("Warmth:", self.spin_halation_warmth)

        form_layout.addWidget(baked_group)

        # --- Real-time Effects Group ---
        live_group = QGroupBox("Real-time Effects")
        live_group.setStyleSheet("QGroupBox { color: #7aa8d9; font-weight: bold; border: 1px solid #555; border-radius: 6px; margin-top: 8px; padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        live_layout = QFormLayout(live_group)
        live_layout.setSpacing(8)

        # CNR — applied per-render in core/processor._render(), not baked.
        self.chk_cnr = QCheckBox("Enable Color Noise Reduction")
        self.chk_cnr.setChecked(True)
        self.chk_cnr.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_cnr)

        self.spin_cnr = self._create_double_spin(0.0, 100.0, CNR_AMOUNT_PCT, 1.0, suffix=" %")
        self.spin_cnr.valueChanged.connect(self.update_preview)
        live_layout.addRow("CNR Amount:", self.spin_cnr)

        # Despike: 3x3-median chroma outlier clamp for fireflies/green spikes.
        self.spin_cnr_despike = self._create_double_spin(0.0, 100.0, CNR_DESPIKE_PCT, 1.0, suffix=" %")
        self.spin_cnr_despike.setToolTip(
            "Clamps isolated colour spikes (fireflies) toward their neighbours — "
            "removes green/colour noise spikes the CNR bilateral leaves behind.")
        self.spin_cnr_despike.valueChanged.connect(self.update_preview)
        live_layout.addRow("CNR Despike:", self.spin_cnr_despike)

        self.spin_cnr_despike_bias = self._create_double_spin(0.0, 100.0, CNR_DESPIKE_BIAS_PCT, 1.0, suffix=" %")
        self.spin_cnr_despike_bias.setToolTip(
            "0 = clamp all colours equally; 100 = clamp green (-a*) only.")
        self.spin_cnr_despike_bias.valueChanged.connect(self.update_preview)
        live_layout.addRow("Despike Green Bias:", self.spin_cnr_despike_bias)

        live_layout.addRow(self._create_separator())

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

        self.spin_ca_str = self._create_double_spin(0.0, 40.0, CA_PIXELS, 0.5, suffix=" px")
        self.spin_ca_str.valueChanged.connect(self.update_preview)
        live_layout.addRow("CA Strength:", self.spin_ca_str)

        # CA advanced controls — collapsible (checkable groupbox; unchecked
        # disables children without hiding them, the closest native Qt
        # behavior to "collapse"). Kept on by default so existing presets
        # that depend on zoom_blur stay visible.
        ca_advanced = QGroupBox("Advanced")
        ca_advanced.setCheckable(True)
        ca_advanced.setChecked(True)
        ca_advanced.setStyleSheet("QGroupBox { color: #888; font-weight: normal; border: 1px solid #444; border-radius: 4px; margin-top: 6px; padding-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }")
        ca_adv_layout = QFormLayout(ca_advanced)
        ca_adv_layout.setSpacing(6)

        self.spin_ca_steps = self._create_spin(1, 10, CA_STEPS)
        self.spin_ca_steps.valueChanged.connect(self.update_preview)
        ca_adv_layout.addRow("CA Steps:", self.spin_ca_steps)

        self.spin_ca_blue_blur = self._create_double_spin(0.0, 5.0, CA_BLUE_BLUR, 0.1, suffix=" px")
        self.spin_ca_blue_blur.valueChanged.connect(self.update_preview)
        ca_adv_layout.addRow("CA Blue Blur:", self.spin_ca_blue_blur)

        self.spin_ca_zoom_blur = self._create_double_spin(0.0, 500.0, CA_ZOOM_BLUR_PCT, 5.0, suffix=" %")
        self.spin_ca_zoom_blur.valueChanged.connect(self.update_preview)
        ca_adv_layout.addRow("CA Zoom Blur:", self.spin_ca_zoom_blur)

        live_layout.addRow(ca_advanced)

        live_layout.addRow(self._create_separator())

        # Softness
        self.chk_softness = QCheckBox("Enable Softness")
        self.chk_softness.setChecked(True)
        self.chk_softness.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_softness)

        self.spin_softness = self._create_double_spin(0.0, 5.0, SOFTNESS_SIGMA, 0.1, suffix=" px")
        self.spin_softness.valueChanged.connect(self.update_preview)
        live_layout.addRow("Softness:", self.spin_softness)

        live_layout.addRow(self._create_separator())

        # Grain
        self.chk_grain = QCheckBox("Enable Grain")
        self.chk_grain.setChecked(True)
        self.chk_grain.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_grain)

        self.spin_grain = self._create_double_spin(0.0, 200.0, GRAIN_STRENGTH_PCT, 5.0, suffix=" %")
        self.spin_grain.valueChanged.connect(self.update_preview)
        live_layout.addRow("Grain Strength:", self.spin_grain)

        live_layout.addRow(self._create_separator())

        # Sharpen
        self.chk_sharpen = QCheckBox("Enable Sharpen")
        self.chk_sharpen.setChecked(True)
        self.chk_sharpen.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_sharpen)

        self.spin_sharpen_str = self._create_double_spin(0.0, 500.0, SHARPEN_STRENGTH_PCT, 5.0, suffix=" %")
        self.spin_sharpen_str.valueChanged.connect(self.update_preview)
        live_layout.addRow("Sharpen Strength:", self.spin_sharpen_str)

        self.spin_sharpen_rad = self._create_double_spin(0.1, 20.0, SHARPEN_RADIUS, 0.1, suffix=" px")
        self.spin_sharpen_rad.valueChanged.connect(self.update_preview)
        live_layout.addRow("Sharpen Radius:", self.spin_sharpen_rad)

        live_layout.addRow(self._create_separator())

        # Vignette
        self.chk_vignette = QCheckBox("Enable Vignette")
        self.chk_vignette.setChecked(True)
        self.chk_vignette.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_vignette)

        self.spin_vignette_str = self._create_double_spin(0.0, 100.0, VIGNETTE_STRENGTH_PCT, 1.0, suffix=" %")
        self.spin_vignette_str.valueChanged.connect(self.update_preview)
        live_layout.addRow("Vignette Strength:", self.spin_vignette_str)

        self.spin_vignette_color = self._create_double_spin(0.0, 100.0, VIGNETTE_COLOR_PCT, 1.0, suffix=" %")
        self.spin_vignette_color.valueChanged.connect(self.update_preview)
        live_layout.addRow("Vignette Color Shift:", self.spin_vignette_color)

        # Signed curve: positive = softer falloff, negative = harder edge,
        # 0 = neutral cosine. Maps to a power exponent via 2^(-curve/50).
        self.spin_vignette_feather = self._create_double_spin(-100.0, 100.0, VIGNETTE_CURVE, 1.0)
        self.spin_vignette_feather.valueChanged.connect(self.update_preview)
        live_layout.addRow("Vignette Curve:", self.spin_vignette_feather)

        live_layout.addRow(self._create_separator())

        # Bloom
        self.chk_bloom = QCheckBox("Enable Bloom")
        self.chk_bloom.setChecked(True)
        self.chk_bloom.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_bloom)

        self.spin_bloom_str = self._create_double_spin(0.0, 100.0, BLOOM_STRENGTH_PCT, 1.0, suffix=" %")
        self.spin_bloom_str.valueChanged.connect(self.update_preview)
        live_layout.addRow("Bloom Strength:", self.spin_bloom_str)

        self.spin_bloom_thresh = self._create_double_spin(-2.0, 8.0, BLOOM_THRESHOLD_STOPS, 0.25, suffix=" EV")
        self.spin_bloom_thresh.setToolTip(
            "Stops above 18% middle grey. 0 = middle grey, +5 ≈ scene white,\n"
            "+4 (default) targets specular highlights, +6 only the very brightest."
        )
        self.spin_bloom_thresh.valueChanged.connect(self.update_preview)
        live_layout.addRow("Bloom Threshold:", self.spin_bloom_thresh)

        form_layout.addWidget(live_group)

        # --- DNG Export Group ---
        dng_group = QGroupBox("DNG Export")
        dng_group.setStyleSheet("QGroupBox { color: #d0d0d0; font-weight: bold; border: 1px solid #555; border-radius: 6px; margin-top: 8px; padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        dng_layout = QHBoxLayout(dng_group)
        dng_layout.setSpacing(6)

        dng_layout.addWidget(QLabel("Profile Name:"))
        self.dng_profile_edit = QLineEdit()
        self.dng_profile_edit.setPlaceholderText("Flashback Standard")
        self.dng_profile_edit.setText(
            self.parent_editor.current_vibe.dng_profile_name
            if self.parent_editor else 'Flashback Standard'
        )
        self.dng_profile_edit.setStyleSheet("QLineEdit { background-color: #3d3d3d; color: #d0d0d0; border: 1px solid #555; border-radius: 4px; padding: 4px; }")
        dng_layout.addWidget(self.dng_profile_edit, 1)

        btn_set_profile = QPushButton("Set")
        btn_set_profile.setStyleSheet(btn_style)
        btn_set_profile.clicked.connect(self._on_set_dng_profile)
        dng_layout.addWidget(btn_set_profile)

        form_layout.addWidget(dng_group)

        # --- LUT Profiling Group ---
        lut_prof_group = QGroupBox("LUT Profiling")
        lut_prof_group.setStyleSheet("QGroupBox { color: #d0d0d0; font-weight: bold; border: 1px solid #555; border-radius: 6px; margin-top: 8px; padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        lut_prof_layout = QVBoxLayout(lut_prof_group)
        lut_prof_layout.setSpacing(6)

        lut_prof_info = QLabel(
            "Exports the current image as a 16-bit ACEScct TIFF for building or\n"
            "previewing LUTs in DaVinci Resolve. Default is the app's standard\n"
            "exposure — what the app feeds the LUT — so a hand-graded LUT\n"
            "previews accurately."
        )
        lut_prof_info.setStyleSheet("color: #888; font-size: 11px;")
        lut_prof_layout.addWidget(lut_prof_info)

        self.chk_lut_reverse_ae = QCheckBox("Apply reverse-AE (film-stock profiling)")
        self.chk_lut_reverse_ae.setChecked(False)
        self.chk_lut_reverse_ae.setToolTip(
            "On: undo each frame's in-camera autoexposure (from EXIF ExposureTime)\n"
            "so every frame sits at a common reference level — needed when profiling\n"
            "a film stock. A normally-metered frame can drop several stops.\n\n"
            "Off (default): export at the app's standard exposure — use this to\n"
            "preview a LUT you graded by hand."
        )
        lut_prof_layout.addWidget(self.chk_lut_reverse_ae)

        self.btn_export_lut_tiff = QPushButton("Export ACEScct TIFF…")
        self.btn_export_lut_tiff.setStyleSheet(btn_style)
        self.btn_export_lut_tiff.clicked.connect(self._on_export_lut_tiff)
        lut_prof_layout.addWidget(self.btn_export_lut_tiff)

        form_layout.addWidget(lut_prof_group)

        # --- Default Folders Group ---
        folders_group = QGroupBox("Default Folders")
        folders_group.setStyleSheet("QGroupBox { color: #d0d0d0; font-weight: bold; border: 1px solid #555; border-radius: 6px; margin-top: 8px; padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        folders_layout = QVBoxLayout(folders_group)
        folders_layout.setSpacing(6)

        folders_info = QLabel(
            "These are the folders the app starts each session with. The\n"
            "export folder can be changed temporarily from the main toolbar;\n"
            "that override is not remembered."
        )
        folders_info.setStyleSheet("color: #888; font-size: 11px;")
        folders_layout.addWidget(folders_info)

        self.lbl_default_import = QLabel()
        self.lbl_default_import.setStyleSheet("color: #d0d0d0; font-size: 11px;")
        self.lbl_default_import.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_default_import.setWordWrap(True)
        self.btn_default_import = QPushButton("Set Camera Import Folder…")
        self.btn_default_import.setStyleSheet(btn_style)
        self.btn_default_import.clicked.connect(self._on_set_default_import_dir)
        folders_layout.addWidget(QLabel("Camera import folder:"))
        folders_layout.addWidget(self.lbl_default_import)
        folders_layout.addWidget(self.btn_default_import)

        self.lbl_default_export = QLabel()
        self.lbl_default_export.setStyleSheet("color: #d0d0d0; font-size: 11px;")
        self.lbl_default_export.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_default_export.setWordWrap(True)
        self.btn_default_export = QPushButton("Set Export Folder…")
        self.btn_default_export.setStyleSheet(btn_style)
        self.btn_default_export.clicked.connect(self._on_set_default_export_dir)
        folders_layout.addWidget(QLabel("Export folder:"))
        folders_layout.addWidget(self.lbl_default_export)
        folders_layout.addWidget(self.btn_default_export)

        self._refresh_default_folder_labels()

        form_layout.addWidget(folders_group)

        form_layout.addStretch()

        scroll.setWidget(container)
        layout.addWidget(scroll)

        self.status_label = QLabel("Ready - press F12 to close")
        self.status_label.setStyleSheet("color: #626262; font-size: 11px; padding-top: 8px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Schema-driven field → widget table for sync. Boolean fields use
        # setChecked; numeric fields use setValue.
        self._bool_widgets = {
            'enable_halation': self.chk_halation,
            'enable_chromatic_aberration': self.chk_ca,
            'enable_softness': self.chk_softness,
            'enable_grain': self.chk_grain,
            'enable_sharpen': self.chk_sharpen,
            'enable_cnr': self.chk_cnr,
            'enable_lut': self.chk_lut,
            'enable_vignette': self.chk_vignette,
            'enable_bloom': self.chk_bloom,
        }
        self._numeric_widgets = {
            'halation_threshold_stops': self.spin_halation_thresh,
            'halation_blur_radius': self.spin_halation_blur,
            'halation_strength_pct': self.spin_halation_str,
            'halation_warmth_pct': self.spin_halation_warmth,
            'ca_pixels': self.spin_ca_str,
            'ca_steps': self.spin_ca_steps,
            'ca_blue_blur': self.spin_ca_blue_blur,
            'ca_zoom_blur_pct': self.spin_ca_zoom_blur,
            'softness_sigma': self.spin_softness,
            'grain_strength_pct': self.spin_grain,
            'sharpen_strength_pct': self.spin_sharpen_str,
            'sharpen_radius': self.spin_sharpen_rad,
            'cnr_amount_pct': self.spin_cnr,
            'cnr_despike_pct': self.spin_cnr_despike,
            'cnr_despike_bias_pct': self.spin_cnr_despike_bias,
            'vignette_strength_pct': self.spin_vignette_str,
            'vignette_color_pct': self.spin_vignette_color,
            'vignette_curve': self.spin_vignette_feather,
            'bloom_strength_pct': self.spin_bloom_str,
            'bloom_threshold_stops': self.spin_bloom_thresh,
        }

    # ===================================================================
    # HELPERS
    # ===================================================================

    def _create_double_spin(self, min_val, max_val, default, step, suffix=""):
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(3 if step < 0.1 else 2 if step < 1 else 1)
        spin.setValue(default)
        spin.setSingleStep(step)
        if suffix:
            spin.setSuffix(suffix)
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
        """Write all widget values into the current vibe (schema-driven)."""
        if not self.parent_editor:
            return
        vibe = self.parent_editor.current_vibe
        for name, w in self._bool_widgets.items():
            setattr(vibe, name, bool(w.isChecked()))
        for name, w in self._numeric_widgets.items():
            setattr(vibe, name, w.value())

        self.status_label.setText("Config updated. Click 'Reload Image' to apply baked effects.")

    def sync_from_config(self):
        """Update all panel widgets from the current vibe."""
        if not self.parent_editor:
            return
        vibe = self.parent_editor.current_vibe
        all_widgets = list(self._bool_widgets.values()) + list(self._numeric_widgets.values())
        for w in all_widgets:
            w.blockSignals(True)
        try:
            for name, w in self._bool_widgets.items():
                w.setChecked(bool(getattr(vibe, name)))
            for name, w in self._numeric_widgets.items():
                w.setValue(getattr(vibe, name))
        finally:
            for w in all_widgets:
                w.blockSignals(False)
        self.refresh_lut_label()

    def refresh_lut_label(self):
        """Update the LUT label to show the active LUT name + origin.

        Factory refs show the id ("factory: disposable"); user refs show
        the filename only ("user: my_look.cube"); empty ref reads as
        "Default" (tone-curve fallback)."""
        from pathlib import Path as _P
        from core.config import LUT_REF_FACTORY, LUT_REF_USER
        ref = self.parent_editor.current_vibe.lut_ref if self.parent_editor else ''
        if not ref:
            self.lut_label.setText("Current LUT: Default")
        elif ref.startswith(LUT_REF_FACTORY):
            self.lut_label.setText(f"Current LUT: factory · {ref[len(LUT_REF_FACTORY):]}")
        elif ref.startswith(LUT_REF_USER):
            self.lut_label.setText(f"Current LUT: user · {_P(ref[len(LUT_REF_USER):]).name}")
        else:
            self.lut_label.setText(f"Current LUT: {ref}")

    def update_modified_indicator(self):
        """Show '• modified' next to the vibe header when session ≠ saved-or-factory."""
        if not self.parent_editor or not hasattr(self.parent_editor, 'current_vibe_id'):
            return
        vibe_id = self.parent_editor.current_vibe_id()
        live = self.parent_editor.current_vibe.to_dict()
        baseline = self.parent_editor._vibe_for(vibe_id).to_dict()
        modified = any(live.get(k) != baseline.get(k) for k in VIBE_FIELD_NAMES)
        suffix = "  •  modified" if modified else ""
        self.vibe_header_label.setText(f"Vibe defaults — {vibe_id}{suffix}")

    def _on_set_dng_profile(self):
        name = self.dng_profile_edit.text().strip() or 'Flashback Standard'
        self.dng_profile_edit.setText(name)
        if self.parent_editor:
            self.parent_editor.set_dng_profile_name(name)
        self.status_label.setText(f"DNG profile name set to '{name}'.")

    def _refresh_default_folder_labels(self):
        if not self.parent_editor:
            return
        s = self.parent_editor.app_settings
        self.lbl_default_import.setText(str(s.value("default_camera_import_dir", "—")))
        self.lbl_default_export.setText(str(s.value("default_export_dir", "—")))

    def _on_set_default_import_dir(self):
        from PySide6.QtWidgets import QFileDialog
        if not self.parent_editor:
            return
        start = str(self.parent_editor.app_settings.value(
            "default_camera_import_dir", self.parent_editor.camera_import_dir))
        directory = QFileDialog.getExistingDirectory(
            self, "Set Default Camera Import Folder", start)
        if not directory:
            return
        self.parent_editor.app_settings.setValue("default_camera_import_dir", directory)
        self._refresh_default_folder_labels()
        self.status_label.setText(
            f"Default camera import folder set. Applies on next launch (currently: {self.parent_editor.camera_import_dir}).")

    def _on_set_default_export_dir(self):
        from PySide6.QtWidgets import QFileDialog
        if not self.parent_editor:
            return
        start = str(self.parent_editor.app_settings.value(
            "default_export_dir", self.parent_editor.output_dir))
        directory = QFileDialog.getExistingDirectory(
            self, "Set Default Export Folder", start)
        if not directory:
            return
        self.parent_editor.app_settings.setValue("default_export_dir", directory)
        self._refresh_default_folder_labels()
        self.status_label.setText(
            f"Default export folder set. Applies on next launch (currently: {self.parent_editor.output_dir}).")

    def _on_export_lut_tiff(self):
        from PySide6.QtWidgets import QFileDialog
        if not self.processor or self.processor.intermediate_acescg is None:
            self.status_label.setText("No image loaded.")
            return
        if not self.parent_editor:
            self.status_label.setText("No editor context.")
            return
        output_dir = QFileDialog.getExistingDirectory(
            self, "Export LUT Profile TIFFs — Choose Output Folder")
        if not output_dir:
            return
        success, total = self.parent_editor.export_lut_tiffs(
            output_dir, reverse_ae=self.chk_lut_reverse_ae.isChecked())
        if total == 0:
            self.status_label.setText("No images to export.")
        else:
            self.status_label.setText(f"Exported {success}/{total} TIFFs to {output_dir}")

    def _on_save_vibe(self):
        if self.parent_editor:
            self.parent_editor.save_current_vibe_defaults()

    def _on_reset_saved(self):
        if self.parent_editor:
            self.parent_editor.reset_current_vibe_to_saved()

    def _on_reset_factory(self):
        if self.parent_editor:
            self.parent_editor.reset_current_vibe_to_factory()

    def update_preview(self):
        """Update real-time preview immediately."""
        self.update_config()
        self.update_modified_indicator()
        if self.parent_editor:
            self.parent_editor.refresh_from_debug()

    def reload_image(self):
        """Force reload current image."""
        if self.parent_editor:
            self.parent_editor.reload_current_image()
            self.status_label.setText("Image reloaded with new baked settings.")

