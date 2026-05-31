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
    HALATION_THRESHOLD, HALATION_BLUR_RADIUS, HALATION_STRENGTH,
    CHROMATIC_ABERRATION_STRENGTH, CHROMATIC_ABERRATION_STEPS,
    SOFTNESS_SIGMA, GRAIN_STRENGTH, SHARPEN_STRENGTH, SHARPEN_RADIUS,
    CNR_SIGMA, VIGNETTE_STRENGTH, VIGNETTE_COLOR_SHIFT,
    BLOOM_STRENGTH, BLOOM_THRESHOLD,
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

        self.spin_halation_thresh = self._create_double_spin(0.0, 1.0, HALATION_THRESHOLD, 0.05)
        self.spin_halation_thresh.valueChanged.connect(self.update_config)
        baked_layout.addRow("Threshold:", self.spin_halation_thresh)

        self.spin_halation_blur = self._create_spin(1, 100, HALATION_BLUR_RADIUS)
        self.spin_halation_blur.valueChanged.connect(self.update_config)
        baked_layout.addRow("Blur Radius:", self.spin_halation_blur)

        self.spin_halation_str = self._create_double_spin(0.0, 2.0, HALATION_STRENGTH, 0.1)
        self.spin_halation_str.valueChanged.connect(self.update_config)
        baked_layout.addRow("Strength:", self.spin_halation_str)

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

        self.spin_cnr = self._create_double_spin(0.0, 5.0, CNR_SIGMA, 0.1)
        self.spin_cnr.valueChanged.connect(self.update_preview)
        live_layout.addRow("CNR Sigma:", self.spin_cnr)

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

        self.spin_ca_str = self._create_double_spin(0.0, 0.02, CHROMATIC_ABERRATION_STRENGTH, 0.001)
        self.spin_ca_str.valueChanged.connect(self.update_preview)
        live_layout.addRow("CA Strength:", self.spin_ca_str)

        self.spin_ca_steps = self._create_spin(1, 10, CHROMATIC_ABERRATION_STEPS)
        self.spin_ca_steps.valueChanged.connect(self.update_preview)
        live_layout.addRow("CA Steps:", self.spin_ca_steps)

        self.spin_ca_blue_blur = self._create_double_spin(0.0, 5.0, 0.0, 0.1)
        self.spin_ca_blue_blur.valueChanged.connect(self.update_preview)
        live_layout.addRow("CA Blue Blur:", self.spin_ca_blue_blur)

        self.spin_ca_zoom_blur = self._create_double_spin(0.0, 20.0, 1.0, 0.1)
        self.spin_ca_zoom_blur.valueChanged.connect(self.update_preview)
        live_layout.addRow("CA Zoom Blur:", self.spin_ca_zoom_blur)

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

        self.spin_grain = self._create_double_spin(0.0, 3.0, GRAIN_STRENGTH, 0.05)
        self.spin_grain.valueChanged.connect(self.update_preview)
        live_layout.addRow("Grain Strength:", self.spin_grain)

        live_layout.addRow(self._create_separator())

        # Sharpen
        self.chk_sharpen = QCheckBox("Enable Sharpen")
        self.chk_sharpen.setChecked(True)
        self.chk_sharpen.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_sharpen)

        self.spin_sharpen_str = self._create_double_spin(0.0, 5.0, SHARPEN_STRENGTH, 0.1)
        self.spin_sharpen_str.valueChanged.connect(self.update_preview)
        live_layout.addRow("Sharpen Strength:", self.spin_sharpen_str)

        self.spin_sharpen_rad = self._create_double_spin(0.1, 20.0, SHARPEN_RADIUS, 0.1)
        self.spin_sharpen_rad.valueChanged.connect(self.update_preview)
        live_layout.addRow("Sharpen Radius:", self.spin_sharpen_rad)

        live_layout.addRow(self._create_separator())

        # Vignette
        self.chk_vignette = QCheckBox("Enable Vignette")
        self.chk_vignette.setChecked(True)
        self.chk_vignette.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_vignette)

        self.spin_vignette_str = self._create_double_spin(0.0, 1.0, VIGNETTE_STRENGTH, 0.05)
        self.spin_vignette_str.valueChanged.connect(self.update_preview)
        live_layout.addRow("Vignette Strength:", self.spin_vignette_str)

        self.spin_vignette_color = self._create_double_spin(0.0, 0.2, VIGNETTE_COLOR_SHIFT, 0.005)
        self.spin_vignette_color.valueChanged.connect(self.update_preview)
        live_layout.addRow("Vignette Color Shift:", self.spin_vignette_color)

        self.spin_vignette_feather = self._create_double_spin(0.1, 8.0, 1.0, 0.1)
        self.spin_vignette_feather.valueChanged.connect(self.update_preview)
        live_layout.addRow("Vignette Feather:", self.spin_vignette_feather)

        live_layout.addRow(self._create_separator())

        # Bloom
        self.chk_bloom = QCheckBox("Enable Bloom")
        self.chk_bloom.setChecked(True)
        self.chk_bloom.stateChanged.connect(self.update_preview)
        live_layout.addRow(self.chk_bloom)

        self.spin_bloom_str = self._create_double_spin(0.0, 1.0, BLOOM_STRENGTH, 0.05)
        self.spin_bloom_str.valueChanged.connect(self.update_preview)
        live_layout.addRow("Bloom Strength:", self.spin_bloom_str)

        self.spin_bloom_thresh = self._create_double_spin(0.0, 1.0, BLOOM_THRESHOLD, 0.05)
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
            "Exports the current image as a 16-bit ACEScct TIFF with full\n"
            "reverse-AE and exposure boost applied — for use as training\n"
            "input in DaVinci Resolve LUT creation."
        )
        lut_prof_info.setStyleSheet("color: #888; font-size: 11px;")
        lut_prof_layout.addWidget(lut_prof_info)

        self.btn_export_lut_tiff = QPushButton("Export LUT Profile TIFF…")
        self.btn_export_lut_tiff.setStyleSheet(btn_style)
        self.btn_export_lut_tiff.clicked.connect(self._on_export_lut_tiff)
        lut_prof_layout.addWidget(self.btn_export_lut_tiff)

        form_layout.addWidget(lut_prof_group)

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
            'halation_threshold': self.spin_halation_thresh,
            'halation_blur_radius': self.spin_halation_blur,
            'halation_strength': self.spin_halation_str,
            'ca_strength': self.spin_ca_str,
            'ca_steps': self.spin_ca_steps,
            'ca_blue_blur': self.spin_ca_blue_blur,
            'ca_zoom_blur': self.spin_ca_zoom_blur,
            'softness_sigma': self.spin_softness,
            'grain_strength': self.spin_grain,
            'sharpen_strength': self.spin_sharpen_str,
            'sharpen_radius': self.spin_sharpen_rad,
            'cnr_sigma': self.spin_cnr,
            'vignette_strength': self.spin_vignette_str,
            'vignette_color_shift': self.spin_vignette_color,
            'vignette_feather': self.spin_vignette_feather,
            'bloom_strength': self.spin_bloom_str,
            'bloom_threshold': self.spin_bloom_thresh,
        }

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
        """Update the LUT label to show the active LUT filename."""
        from pathlib import Path as _P
        path = self.parent_editor.current_vibe.lut_path if self.parent_editor else ''
        self.lut_label.setText(f"Current LUT: {_P(path).name}" if path else "Current LUT: Default")

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
        success, total = self.parent_editor.export_lut_tiffs(output_dir)
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

