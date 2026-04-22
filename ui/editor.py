"""
Main application window and fullscreen Zen overlay.

FlashbackEditor    — QMainWindow: file loading, sliders, thumbnail strip,
                     export, keyboard shortcuts, drag & drop.
FullscreenZenOverlay — frameless fullscreen view with gesture-based adjustments.
"""
import sys
import os
import shutil
import time
import traceback
import platform
from pathlib import Path

# core must be imported before colour to apply the NumPy 2.0 compatibility shim
import core  # noqa: F401

import numpy as np
import cv2
import colour

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QFileDialog, QMessageBox, QProgressBar,
    QScrollArea, QFrame, QSizePolicy, QCheckBox,
)
from PySide6.QtCore import (
    Qt, QTimer, QSize, Signal, QPoint, QThread, QEvent,
    QPropertyAnimation, QEasingCurve, QUrl, QStandardPaths, QSettings,
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QCursor, QFont,
    QFontDatabase, QLinearGradient, QMovie, QPainterPath, QColorSpace,
    QSurfaceFormat, QAction, QKeySequence, QIcon, QPalette,
)

from core import resource_path
from core.processor import FlashbackProcessor, export_image
from core.config import _timing_print, DebugConfig

from .widgets import (
    ThumbnailWorker, ThumbnailWidget, ThumbnailStrip,
    FadeOverlayWidget, LoaderOverlay, ZoomableImageWidget, VibePicker,
)
from .debug_panel import DebugPanel
from .scrub_slider import ScrubSlider
from . import theme
from .theme import (
    C, UI_FONT, MONO_FONT,
    icon_btn_qss, section_title_qss, section_reset_link_qss,
    process_btn_qss, format_pill_qss, svg_icon,
)


# =============================================================================
# FULLSCREEN ZEN OVERLAY
# =============================================================================

class FullscreenZenOverlay(QWidget):
    closed = Signal()
    navigated = Signal(int)
    rotated = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #F3F3F3;")
        self.setFocusPolicy(Qt.StrongFocus)

        self.main_window = parent
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.back_btn = QPushButton("←", self)
        self.back_btn.setFixedSize(30, 30)
        self.back_btn.move(25, 25)
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.setStyleSheet("""
            QPushButton { border: none; background: transparent; color: #A0A0A0; font-size: 22px; }
            QPushButton:hover { color: #333; }
        """)
        self.back_btn.clicked.connect(self.close_zen)

        self.drag_start_pos = None
        self.lock_axis = None
        self._current_raw_pixmap = None

    def update_preview(self, pixmap):
        """Stores the pixmap and triggers a redraw."""
        if not pixmap or pixmap.isNull():
            return
        self._current_raw_pixmap = pixmap
        self._recalculate_layout()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._recalculate_layout()

    def _recalculate_layout(self):
        """The math that prevents jumping and handles Retina scaling."""
        if self._current_raw_pixmap is None:
            return

        dpr = self.devicePixelRatio()
        pix = self._current_raw_pixmap.copy()
        pix.setDevicePixelRatio(dpr)

        view_w, view_h = self.width(), self.height()
        if view_w < 100:
            return

        max_w = int(view_w * 0.9)
        max_h = int(view_h * 0.9)

        logical_w = pix.width() / dpr
        logical_h = pix.height() / dpr

        if logical_w > max_w or logical_h > max_h:
            final_pix = pix.scaled(
                int(max_w * dpr), int(max_h * dpr),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            final_pix.setDevicePixelRatio(dpr)
        else:
            final_pix = pix

        self.image_label.setPixmap(final_pix)
        self.image_label.setFixedSize(int(final_pix.width() / dpr), int(final_pix.height() / dpr))

        x = (view_w - self.image_label.width()) // 2
        y = (view_h - self.image_label.height()) // 2
        self.image_label.move(x, y)

    def mousePressEvent(self, event):
        self.drag_start_pos = event.position().toPoint()
        self.lock_axis = None
        self.base_exposure = self.main_window.slider_exposure.value() / 100.0
        self.base_wb = self.main_window.slider_wb.value()
        self.base_tint = self.main_window.slider_tint.value()

    def mouseMoveEvent(self, event):
        if not self.drag_start_pos:
            return

        curr_pos = event.position().toPoint()
        delta = curr_pos - self.drag_start_pos
        dx, dy = delta.x(), delta.y()

        if self.lock_axis is None:
            threshold = 5
            if abs(dx) > threshold:
                self.lock_axis = 'h'
                self.drag_start_pos = curr_pos
            elif abs(dy) > threshold:
                self.lock_axis = 'v'
                self.drag_start_pos = curr_pos
            return

        if event.buttons() & Qt.LeftButton:
            if self.lock_axis == 'v':
                move_y = (curr_pos.y() - self.drag_start_pos.y())
                val = self.base_exposure + (-move_y / 300.0)
                self.main_window.slider_exposure.setValue(int(val * 100))
            elif self.lock_axis == 'h':
                move_x = (curr_pos.x() - self.drag_start_pos.x())
                val = self.base_wb + (move_x * 5.0)
                self.main_window.slider_wb.setValue(int(val))

        elif event.buttons() & Qt.RightButton and self.lock_axis == 'h':
            move_x = (curr_pos.x() - self.drag_start_pos.x())
            val = self.base_tint + (move_x / 5.0)
            self.main_window.slider_tint.setValue(int(val))

    def mouseReleaseEvent(self, event):
        self.drag_start_pos = None
        self.lock_axis = None

        self.main_window.processor.preview_mode = 'hq'
        img_array = self.main_window.processor._render_fast()

        if img_array is not None:
            self.main_window.display_image(img_array)
            self.main_window.update_current_thumbnail(img_array)
            self.main_window.update_mode_label()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close_zen()
        elif event.key() == Qt.Key_Left:
            self.navigated.emit(-1)
        elif event.key() == Qt.Key_Right:
            self.navigated.emit(1)
        elif event.key() == Qt.Key_Up:
            self.rotated.emit(90)
        elif event.key() == Qt.Key_Down:
            self.rotated.emit(-90)
        elif event.key() == Qt.Key_C and (event.modifiers() & Qt.ControlModifier):
            if self.main_window.image_files:
                self.main_window.copy_settings()
        elif event.key() == Qt.Key_V and (event.modifiers() & Qt.ControlModifier):
            if self.main_window.image_files and self.main_window.settings_clipboard:
                self.main_window.paste_settings()
        elif event.key() == Qt.Key_R and (event.modifiers() & Qt.ControlModifier):
            self.main_window.reset_all_sliders()
        elif event.key() in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4):
            vibes = ('disposable', 'point_shoot', 'rangefinder', 'monochrome')
            self.main_window.vibe_picker.set_vibe(vibes[event.key() - Qt.Key_1])

    def close_zen(self):
        self.hide()
        self.closed.emit()


# =============================================================================
# MAIN EDITOR WINDOW
# =============================================================================

class FlashbackEditor(QMainWindow):
    """Main application window for Flashback image editing."""

    SUPPORTED_EXTENSIONS = (
        '.dng', '.raf', '.cr2', '.cr3', '.nef',
        '.arw', '.orf', '.rw2', '.tif', '.tiff'
    )

    def __init__(self):
        super().__init__()

        if sys.platform == 'darwin':
            self.setUnifiedTitleAndToolBarOnMac(True)

        # Application state
        self.processor = None
        self.image_files = []
        self.current_index = 0
        self.image_settings = {}
        self.image_cache = {}
        self.preview_cache = {}
        self.export_tiff_mode = False
        self.thumbnail_cache = {}
        self.thumbnail_settings = {}

        pictures_loc = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        base_dir = pictures_loc if pictures_loc else str(Path.home())

        self.output_dir = os.path.join(base_dir, "Flashback_Output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.app_settings = QSettings("Flashback", "Editor")
        self.pending_file_path = None

        # Theme: load persisted choice (default: light) and register listener
        # so every setStyleSheet()/icon registered below can be re-applied.
        saved_theme = self.app_settings.value("theme", "light")
        if saved_theme in ("light", "dark"):
            theme.set_theme(saved_theme)
        self._themed_styles: list = []   # (widget, style_builder) pairs
        self._themed_icons: list = []    # (button, rel_path, color_token, size) tuples
        self._themed_repaint: list = []  # widgets whose paintEvent uses palette
        theme.register_theme_listener(self._apply_theme)

        self.thumbnails_loading = False
        self.thumbnail_worker = None
        self.add_thumbnail_worker = None
        self._thumbnails_dirty = set()
        self._lut_cache: dict = {}

        self._tint_manual_offset = 0.0  # user's manual tint correction on top of WB coupling

        self.pending_render = False

        self._slider_render_timer = QTimer(self)
        self._slider_render_timer.setSingleShot(True)
        self._slider_render_timer.setInterval(40)
        self._slider_render_timer.timeout.connect(self._on_slider_render_tick)

        lut_path = resource_path("assets/luts/look.cube")
        if not os.path.exists(lut_path):
            lut_path = None
        self.processor = FlashbackProcessor(lut_path)

        self.init_ui()

        self.debug_panel = DebugPanel(self.processor, self)
        self.debug_panel.sync_from_config()
        self.debug_panel.hide()

        screen = QApplication.primaryScreen().geometry()
        main_geo = self.geometry()
        debug_x = main_geo.right() + 20
        if debug_x + 400 > screen.width():
            debug_x = main_geo.left() - 420
        self.debug_panel.move(max(0, debug_x), main_geo.y())

        QTimer.singleShot(500, self.detect_camera)

        self.zen_overlay = FullscreenZenOverlay(self)
        self.zen_overlay.closed.connect(self.on_zen_closed)
        self.zen_overlay.navigated.connect(self.on_zen_navigate)
        self.zen_overlay.rotated.connect(self.on_zen_rotate)

    # ===================================================================
    # ZEN MODE
    # ===================================================================

    def enter_zen_mode(self):
        if not self.processor or self.image_label._original_pixmap is None:
            return
        self.zen_overlay.showFullScreen()
        QTimer.singleShot(50, lambda: self.zen_overlay.update_preview(self.image_label._original_pixmap))
        self.zen_overlay.raise_()
        self.zen_overlay.activateWindow()
        self.zen_overlay.setFocus()

    def on_zen_navigate(self, direction):
        new_index = self.current_index + direction
        if 0 <= new_index < len(self.image_files):
            self.current_index = new_index
        self.load_current_image()

    def on_zen_rotate(self, angle):
        if angle == 90:
            self.rotate_clockwise()
        else:
            self.rotate_counterclockwise()
        if hasattr(self, 'zen_overlay') and self.zen_overlay.isVisible():
            self.zen_overlay.update_preview(self.image_label._original_pixmap)

    def on_zen_closed(self):
        self.refresh_from_debug()

    # ===================================================================
    # ROTATION
    # ===================================================================

    def rotate_clockwise(self):
        if not self.image_files:
            return
        img_array = self.processor.rotate_clockwise()
        self.display_image(img_array)
        self.update_current_thumbnail(img_array)
        file_path = str(self.image_files[self.current_index])
        self.image_cache[file_path] = self.processor.intermediate_acescct.copy()

    def rotate_counterclockwise(self):
        if not self.image_files:
            return
        img_array = self.processor.rotate_counterclockwise()
        self.display_image(img_array)
        self.update_current_thumbnail(img_array)
        file_path = str(self.image_files[self.current_index])
        self.image_cache[file_path] = self.processor.intermediate_acescct.copy()

    # ===================================================================
    # LUT LOADING
    # ===================================================================

    def _load_custom_lut(self):
        """Prompt for a .cube file and update the processor LUT."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select LUT", "", "LUT Files (*.cube)"
        )
        if file_path:
            try:
                custom_lut = colour.io.read_LUT(file_path)
                if self.processor:
                    self.processor.lut_preview = custom_lut
                    self.processor.lut_full = custom_lut
                lut_name = Path(file_path).name
                self.debug_panel.lut_label.setText(f"LUT: {lut_name}")
                self.refresh_from_debug()
            except Exception as e:
                QMessageBox.warning(self, "LUT Load Error", f"Failed to parse LUT file:\n{e}")

    # ===================================================================
    # EVENT FILTER (double-click sliders to reset)
    # ===================================================================

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.MouseButtonDblClick:
            if source == self.slider_exposure:
                self.reset_exposure_slider()
                return True
            elif source == self.slider_wb:
                self.reset_wb_slider()
                return True
            elif source == self.slider_tint:
                self.reset_tint_slider()
                return True
        return super().eventFilter(source, event)

    def reset_exposure_slider(self):
        self.slider_exposure.blockSignals(True)
        self.slider_exposure.setValue(0)
        self.slider_exposure.blockSignals(False)
        self.label_exposure.setText("0.0 EV")
        self.processor.user_settings['exposure_ev'] = 0.0
        self.processor.preview_mode = 'hq'
        img_array = self.processor.render_preview()
        if img_array is not None:
            self.display_image(img_array)
            self.update_current_thumbnail(img_array)
            self.save_current_settings()
            self.update_mode_label()

    def reset_wb_slider(self):
        self.slider_wb.blockSignals(True)
        self.slider_wb.setValue(0)
        self.slider_wb.blockSignals(False)
        self.label_wb.setText("5600 K")
        self.processor.user_settings['wb_temp'] = 0.0
        if self.chk_wb_link.isChecked():
            self._tint_manual_offset = 0.0
            self.slider_tint.blockSignals(True)
            self.slider_tint.setValue(0)
            self.slider_tint.blockSignals(False)
            self.label_tint.setText("+0")
            self.processor.user_settings['tint'] = 0.0
        self.processor.preview_mode = 'hq'
        img_array = self.processor.render_preview()
        if img_array is not None:
            self.display_image(img_array)
            self.update_current_thumbnail(img_array)
            self.save_current_settings()
            self.update_mode_label()

    def reset_tint_slider(self):
        self._tint_manual_offset = 0.0
        self.slider_tint.blockSignals(True)
        self.slider_tint.setValue(0)
        self.slider_tint.blockSignals(False)
        self.label_tint.setText("+0")
        self.processor.user_settings['tint'] = 0.0
        self.processor.preview_mode = 'hq'
        img_array = self.processor.render_preview()
        if img_array is not None:
            self.display_image(img_array)
            self.update_current_thumbnail(img_array)
            self.save_current_settings()
            self.update_mode_label()

    # ===================================================================
    # THEME HELPERS
    # ===================================================================

    def _themed(self, widget, style_builder):
        """Apply `style_builder()` now and remember it for theme swaps."""
        widget.setStyleSheet(style_builder())
        self._themed_styles.append((widget, style_builder))
        return widget

    def _themed_icon(self, button, rel_path, color_token="text_label", size=14):
        """Set an SVG icon on a button and remember it for re-tinting."""
        button.setIcon(svg_icon(rel_path, color_token, size))
        button.setIconSize(QSize(size, size))
        self._themed_icons.append((button, rel_path, color_token, size))
        return button

    def _apply_theme(self):
        """Re-run every registered stylesheet and icon with the current palette."""
        for widget, builder in self._themed_styles:
            try:
                widget.setStyleSheet(builder())
            except Exception:
                pass
        for button, rel_path, color_token, size in self._themed_icons:
            try:
                button.setIcon(svg_icon(rel_path, color_token, size))
            except Exception:
                pass
        # Regenerate drag-overlay strings (they hold cached accent/text colours)
        if hasattr(self, "_rebuild_drag_styles"):
            self._rebuild_drag_styles()
        # Dynamic styles (format pills, mode label, process-button-done, etc.)
        # aren't registered — they're reapplied by their owners on the next
        # state change. Trigger that here so the palette swap is immediate.
        if hasattr(self, "btn_export_jpeg") and hasattr(self, "export_tiff_mode"):
            try:
                self.set_export_mode(self.export_tiff_mode)
            except Exception:
                pass
        if hasattr(self, "mode_label"):
            try:
                self.update_mode_label()
            except Exception:
                pass
        # Force a repaint on widgets that read the palette inside paintEvent
        for w in self._themed_repaint:
            try:
                w.update()
            except Exception:
                pass
        # Apple/Windows native chrome needs to follow the theme too
        if getattr(self, "_native_chrome_applied", False):
            try:
                from ui import native_chrome
                native_chrome.apply(self, theme.current_theme())
            except Exception:
                pass

    def toggle_theme(self):
        new_name = theme.toggle_theme()   # listeners fire → _apply_theme()
        self.app_settings.setValue("theme", new_name)
        self._refresh_theme_toggle_icon()

    def _refresh_theme_toggle_icon(self):
        btn = getattr(self, "btn_theme_toggle", None)
        if btn is None:
            return
        # Unicode glyphs sidestep the need for bundled sun/moon SVGs.
        btn.setText("☀" if theme.current_theme() == "dark" else "☾")

    def _rebuild_drag_styles(self):
        """Rebuild the cached drag-overlay stylesheets with current palette values."""
        accent = C['accent']
        text_dim = C['text_dim']
        self._drag_style_active = (
            f"QFrame {{ background: rgba(0,0,0,0.55); border: 2px dashed {accent}; border-radius: 8px; }}"
            f"QLabel {{ color: {accent}; font-size: 16px; font-weight: 600; background: transparent; border: none; }}"
        )
        self._drag_style_dim = (
            f"QFrame {{ background: rgba(0,0,0,0.35); border: 2px dashed {text_dim}; border-radius: 8px; }}"
            f"QLabel {{ color: {text_dim}; font-size: 14px; font-weight: 500; background: transparent; border: none; }}"
        )
        # If they're currently visible, refresh whichever one is shown.
        if hasattr(self, "drag_overlay"):
            self.drag_overlay.setStyleSheet(self._drag_style_active)
        if hasattr(self, "drag_overlay_add"):
            self.drag_overlay_add.setStyleSheet(self._drag_style_dim)

    # ===================================================================
    # UI CONSTRUCTION
    # ===================================================================

    def init_ui(self):
        self.setWindowTitle("Flashback One35 v2 Editor")
        self.resize(1200, 760)
        QTimer.singleShot(0, self.center_window)

        theme.load_app_fonts()
        app_font = theme.ui_font(10, QFont.Normal)
        self.setFont(app_font)

        main_widget = QWidget()
        main_widget.setObjectName("MainWidget")
        main_widget.setAttribute(Qt.WA_StyledBackground, True)
        self._themed(
            main_widget,
            lambda: f"QWidget#MainWidget {{ background-color: {C['bg_window']}; }}",
        )
        self.setCentralWidget(main_widget)
        self.setAcceptDrops(True)

        # Drag overlays — strings are rebuilt on each theme change so their
        # accent/text colours stay in sync.
        self.drag_overlay = QFrame(main_widget)
        drag_layout = QVBoxLayout(self.drag_overlay)
        self._drag_label = QLabel("Drop DNG files here")
        self._drag_label.setAlignment(Qt.AlignCenter)
        drag_layout.addWidget(self._drag_label)
        self.drag_overlay.hide()

        self.drag_overlay_add = QFrame(main_widget)
        add_drag_layout = QVBoxLayout(self.drag_overlay_add)
        self._add_drag_label = QLabel("Add images")
        self._add_drag_label.setAlignment(Qt.AlignCenter)
        add_drag_layout.addWidget(self._add_drag_label)
        self.drag_overlay_add.hide()
        self._rebuild_drag_styles()

        # === Root vertical layout: [toolbar | body | filmstrip | statusbar]
        root = QVBoxLayout(main_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ─────────────── SUB-TOOLBAR ───────────────
        root.addWidget(self._build_sub_toolbar())

        # ─────────────── BODY: image + right rail ───────────────
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Image column
        image_col = QWidget()
        image_col.setObjectName("ImageCol")
        image_col.setAttribute(Qt.WA_StyledBackground, True)
        self._themed(
            image_col,
            lambda: f"QWidget#ImageCol {{ background: {C['bg_window']}; }}",
        )
        image_col_layout = QVBoxLayout(image_col)
        image_col_layout.setContentsMargins(20, 16, 20, 12)
        image_col_layout.setSpacing(10)

        self.image_label = ZoomableImageWidget()
        self.image_label.setMinimumSize(640, 480)
        image_col_layout.addWidget(self.image_label, 1)

        image_col_layout.addWidget(self._build_image_meta_row())
        body_layout.addWidget(image_col, 1)

        # Right rail
        body_layout.addWidget(self._build_right_rail())
        root.addWidget(body, 1)

        # ─────────────── FILMSTRIP ───────────────
        filmstrip = QWidget()
        filmstrip.setObjectName("Filmstrip")
        filmstrip.setAttribute(Qt.WA_StyledBackground, True)
        self._themed(
            filmstrip,
            lambda: (
                f"QWidget#Filmstrip {{"
                f"  background: {C['bg_strip']};"
                f"  border-top: 1px solid {C['border_soft']};"
                f"}}"
            ),
        )
        filmstrip.setFixedHeight(96)
        fl = QHBoxLayout(filmstrip)
        fl.setContentsMargins(12, 12, 12, 12)
        fl.setSpacing(0)

        self.thumbnail_strip = ThumbnailStrip()
        self.thumbnail_strip.thumbnail_clicked.connect(self.on_thumbnail_click)
        self.thumbnail_strip.thumbnail_right_clicked.connect(self.on_thumbnail_right_click)
        self.thumbnail_strip.thumbnail_paste_selected.connect(self.on_thumbnail_paste_selected)
        fl.addWidget(self.thumbnail_strip)

        self.fade_overlay = FadeOverlayWidget(self.thumbnail_strip)
        root.addWidget(filmstrip)

        # ─────────────── STATUS BAR ───────────────
        root.addWidget(self._build_status_bar())

        self.loader_overlay = LoaderOverlay(self.centralWidget())
        self.settings_clipboard = None

        # Keyboard: ↑ rotates clockwise, ↓ rotates counter-clockwise
        rotate_cw_sc = QAction(self)
        rotate_cw_sc.setShortcut(QKeySequence(Qt.Key_Up))
        rotate_cw_sc.triggered.connect(self.rotate_clockwise)
        self.addAction(rotate_cw_sc)
        rotate_ccw_sc = QAction(self)
        rotate_ccw_sc.setShortcut(QKeySequence(Qt.Key_Down))
        rotate_ccw_sc.triggered.connect(self.rotate_counterclockwise)
        self.addAction(rotate_ccw_sc)

        # ⌘R resets all sliders
        reset_sc = QAction(self)
        reset_sc.setShortcut(QKeySequence("Ctrl+R"))
        reset_sc.triggered.connect(self.reset_all_sliders)
        self.addAction(reset_sc)

        # 1–4 select vibe preset
        for key, vibe_id in (('1', 'disposable'), ('2', 'point_shoot'),
                             ('3', 'rangefinder'), ('4', 'monochrome')):
            sc = QAction(self)
            sc.setShortcut(QKeySequence(key))
            sc.triggered.connect(lambda _=False, v=vibe_id: self.vibe_picker.set_vibe(v))
            self.addAction(sc)

        self._build_menu_bar()
        self._on_vibe_selected('disposable')

    # ── sub-toolbar ─────────────────────────────────────────────────────
    def _build_sub_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("SubToolbar")
        bar.setAttribute(Qt.WA_StyledBackground, True)
        bar.setFixedHeight(44)
        self._themed(
            bar,
            lambda: (
                f"QWidget#SubToolbar {{"
                f"  background: {C['bg_toolbar']};"
                f"  border-bottom: 1px solid {C['border_soft']};"
                f"}}"
            ),
        )
        l = QHBoxLayout(bar)
        l.setContentsMargins(12, 0, 12, 0)
        l.setSpacing(6)

        def icon_btn(tooltip, svg_name=None, text=None, size=28):
            b = QPushButton()
            b.setFixedSize(size, size)
            b.setToolTip(tooltip)
            b.setCursor(Qt.PointingHandCursor)
            self._themed(b, lambda s=size: icon_btn_qss(s, 4))
            if svg_name:
                self._themed_icon(b, svg_name, "text_label", 14)
            elif text:
                b.setText(text)
                f = theme.ui_font(13, QFont.Medium)
                b.setFont(f)
            return b

        self.btn_open = icon_btn("Open folder (⌘O)", "assets/icons/folder.svg")
        self.btn_open.clicked.connect(self.open_files)
        l.addWidget(self.btn_open)

        self.btn_detect_camera = icon_btn("Import from camera", "assets/icons/camera.svg")
        self.btn_detect_camera.clicked.connect(self.detect_camera)
        l.addWidget(self.btn_detect_camera)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        self._themed(
            sep1,
            lambda: f"color: {C['border_input']}; background: {C['border_input']};",
        )
        sep1.setFixedSize(1, 16)
        l.addSpacing(4)
        l.addWidget(sep1)
        l.addSpacing(4)

        self.zen_btn = icon_btn("Zen mode (fullscreen)", text="⛶")
        self.zen_btn.clicked.connect(self.enter_zen_mode)
        l.addWidget(self.zen_btn)

        l.addStretch(1)

        # Theme toggle (light ↔ dark). Glyph flips to indicate *destination* theme.
        self.btn_theme_toggle = icon_btn("Toggle light / dark theme", text="☾")
        self.btn_theme_toggle.clicked.connect(self.toggle_theme)
        l.addWidget(self.btn_theme_toggle)
        self._refresh_theme_toggle_icon()

        return bar

    # ── image meta row (under the image) ────────────────────────────────
    def _build_image_meta_row(self) -> QWidget:
        row = QWidget()
        row.setFixedHeight(28)
        rl = QHBoxLayout(row)
        rl.setContentsMargins(2, 0, 2, 0)
        rl.setSpacing(8)

        self.label_filename = QLabel("")
        self.label_filename.setFont(theme.ui_font(11, QFont.Medium))
        self._themed(self.label_filename, lambda: f"color: {C['text_secondary']};")
        rl.addWidget(self.label_filename)

        rl.addStretch(1)

        def small_btn(arrow, tooltip, slot):
            b = QPushButton(arrow)
            b.setFixedSize(24, 24)
            b.setFont(theme.ui_font(13, QFont.Medium))
            b.setToolTip(tooltip)
            b.setCursor(Qt.PointingHandCursor)
            self._themed(b, lambda: icon_btn_qss(24, 3))
            b.clicked.connect(slot)
            return b

        # rotate buttons — pulled down from the toolbar to shorten travel
        self.btn_rotate_ccw = small_btn("↺", "Rotate left (↓)", self.rotate_counterclockwise)
        self.btn_rotate_cw = small_btn("↻", "Rotate right (↑)", self.rotate_clockwise)
        rl.addWidget(self.btn_rotate_ccw)
        rl.addWidget(self.btn_rotate_cw)

        rl.addSpacing(12)

        # counter
        self.label_counter = QLabel("0 / 0")
        self.label_counter.setFont(theme.mono_font(10, QFont.Medium))
        self._themed(self.label_counter, lambda: f"color: {C['text_dim']};")
        rl.addWidget(self.label_counter)

        rl.addSpacing(8)

        self.btn_prev_image = small_btn("‹", "Previous (←)", self.prev_image)
        self.btn_next_image = small_btn("›", "Next (→)", self.next_image)
        rl.addWidget(self.btn_prev_image)
        rl.addWidget(self.btn_next_image)
        return row

    # ── right rail ──────────────────────────────────────────────────────
    def _build_right_rail(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("RightRail")
        rail.setAttribute(Qt.WA_StyledBackground, True)
        rail.setFixedWidth(300)
        self._themed(
            rail,
            lambda: (
                f"QWidget#RightRail {{"
                f"  background: {C['bg_rail']};"
                f"  border-left: 1px solid {C['border_soft']};"
                f"}}"
            ),
        )
        v = QVBoxLayout(rail)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        v.addWidget(self._build_vibe_section())
        v.addWidget(self._divider())
        v.addWidget(self._build_tone_section())
        v.addWidget(self._divider())
        v.addWidget(self._build_color_section())
        v.addStretch(1)
        v.addWidget(self._divider())
        v.addWidget(self._build_export_footer())
        return rail

    def _divider(self) -> QFrame:
        d = QFrame()
        d.setFixedHeight(1)
        self._themed(d, lambda: f"background: {C['border_soft']};")
        return d

    def _section_header(self, title: str, aside: QWidget = None) -> QWidget:
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        lbl = QLabel(title)
        self._themed(lbl, lambda: section_title_qss())
        hl.addWidget(lbl)
        hl.addStretch(1)
        if aside is not None:
            hl.addWidget(aside)
        return w

    def _slider_row(self, label_text: str, value_label: QLabel, slider: ScrubSlider) -> QWidget:
        """Label row + slider as a single column."""
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        l = QLabel(label_text)
        l.setFont(theme.ui_font(11, QFont.Medium))
        self._themed(
            l,
            lambda: f"color: {C['text_label']}; letter-spacing: 0.4px;",
        )
        header.addWidget(l)
        header.addStretch(1)
        value_label.setFont(theme.mono_font(12, QFont.Medium))
        self._themed(
            value_label,
            lambda: f"color: {C['text_secondary']}; padding: 2px 4px;",
        )
        header.addWidget(value_label)

        header_w = QWidget()
        header_w.setLayout(header)
        bl.addWidget(header_w)
        bl.addWidget(slider)
        return box

    def _build_vibe_section(self) -> QWidget:
        sec = QWidget()
        v = QVBoxLayout(sec)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(10)

        v.addWidget(self._section_header("VIBE"))

        self.vibe_picker = VibePicker()
        self.vibe_picker.vibe_changed.connect(self._on_vibe_selected)
        v.addWidget(self.vibe_picker)
        return sec

    def _on_vibe_selected(self, vibe_id: str):
        from core.config import DebugConfig, VIBE_PRESETS
        s = VIBE_PRESETS[vibe_id]
        DebugConfig.enable_chromatic_aberration = s['enable_ca']
        DebugConfig.ca_strength = s['ca_strength']
        DebugConfig.softness_sigma = s['softness']
        DebugConfig.sharpen_strength = s['sharpness']
        DebugConfig.sharpen_radius = s['sharpen_radius']
        DebugConfig.grain_strength = s['grain']
        lut_path = resource_path(s['lut'])
        try:
            if lut_path not in self._lut_cache:
                self._lut_cache[lut_path] = colour.io.read_LUT(lut_path)
            lut = self._lut_cache[lut_path]
            self.processor.lut_preview = lut
            self.processor.lut_full = lut
        except Exception as e:
            print(f"⚠ Could not load vibe LUT '{lut_path}': {e}")
        if hasattr(self, 'debug_panel'):
            self.debug_panel.sync_from_config()
        self.refresh_from_debug()
        self._refresh_all_thumbnails()

    _DEFAULT_USER_SETTINGS = {'exposure_ev': 0.0, 'wb_temp': 0, 'tint': 0.0}

    def _refresh_all_thumbnails(self):
        """Re-render every cached thumbnail — used after vibe/global-effect changes."""
        if not self.image_files:
            return
        for idx, path in enumerate(self.image_files):
            if idx == self.current_index:
                continue
            settings = self.image_settings.get(str(path), self._DEFAULT_USER_SETTINGS)
            self.update_thumbnail_for_settings(idx, settings)
            if idx % 5 == 0:
                QApplication.processEvents()

    def _build_tone_section(self) -> QWidget:
        sec = QWidget()
        v = QVBoxLayout(sec)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(10)

        # Floating "Reset" link (no TONE headline per design)
        reset_link = QPushButton("Reset")
        self._themed(reset_link, lambda: section_reset_link_qss())
        reset_link.setCursor(Qt.PointingHandCursor)
        reset_link.clicked.connect(self.reset_all_sliders)
        self.btn_reset_all = reset_link
        reset_row = QHBoxLayout()
        reset_row.setContentsMargins(0, 0, 0, 0)
        reset_row.addStretch(1)
        reset_row.addWidget(reset_link)
        reset_w = QWidget()
        reset_w.setLayout(reset_row)
        v.addWidget(reset_w)

        # Exposure
        self.label_exposure = QLabel("0.0 EV")
        self.label_exposure.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider_exposure = ScrubSlider(dual=True)
        self.slider_exposure.setMinimum(-20)
        self.slider_exposure.setMaximum(20)
        self.slider_exposure.setValue(0)
        self.slider_exposure.valueChanged.connect(self.on_exposure_slider_moved)
        self.slider_exposure.sliderReleased.connect(self.on_exposure_released)
        self.slider_exposure.installEventFilter(self)
        v.addWidget(self._slider_row("EXPOSURE", self.label_exposure, self.slider_exposure))
        return sec

    def _build_color_section(self) -> QWidget:
        sec = QWidget()
        v = QVBoxLayout(sec)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(10)

        # Temperature (WB)
        self.label_wb = QLabel("5600 K")
        self.label_wb.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider_wb = ScrubSlider(dual=True)
        self.slider_wb.setMinimum(-3000)
        self.slider_wb.setMaximum(3000)
        self.slider_wb.setValue(0)
        self.slider_wb.valueChanged.connect(self.on_wb_slider_moved)
        self.slider_wb.sliderReleased.connect(self.on_wb_released)
        self.slider_wb.installEventFilter(self)
        v.addWidget(self._slider_row("TEMPERATURE", self.label_wb, self.slider_wb))

        # Tint — with AUTO toggle pill in the header
        self.label_tint = QLabel("+0")
        self.label_tint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        auto_pill = QPushButton("AUTO")
        auto_pill.setCheckable(True)
        auto_pill.setCursor(Qt.PointingHandCursor)
        auto_pill.setFont(theme.ui_font(9, QFont.DemiBold))
        auto_pill.setFixedHeight(18)
        self._themed(auto_pill, lambda: (
            f"QPushButton {{"
            f"  background: {C['bg_input']};"
            f"  border: 1px solid {C['border_input']};"
            f"  border-radius: 9px;"
            f"  color: {C['text_dim']};"
            f"  letter-spacing: 0.8px;"
            f"  padding: 0 8px;"
            f"}}"
            f"QPushButton:hover {{ color: {C['text_primary']}; }}"
            f"QPushButton:checked {{"
            f"  background: {C['accent']};"
            f"  border-color: {C['accent']};"
            f"  color: #1a1410;"
            f"}}"
        ))
        auto_pill.setToolTip(
            "Auto tint: WB moves tint proportionally.\n"
            "Manual tint nudges are preserved on top."
        )
        auto_pill.toggled.connect(self._on_wb_link_toggled)
        self.chk_wb_link = auto_pill

        tint_header = QHBoxLayout()
        tint_header.setContentsMargins(0, 0, 0, 0)
        tint_header.setSpacing(8)
        tl = QLabel("TINT")
        tl.setFont(theme.ui_font(11, QFont.Medium))
        self._themed(tl, lambda: f"color: {C['text_label']}; letter-spacing: 0.4px;")
        tint_header.addWidget(tl)
        tint_header.addWidget(auto_pill)
        tint_header.addStretch(1)
        self.label_tint.setFont(theme.mono_font(12, QFont.Medium))
        self._themed(self.label_tint, lambda: f"color: {C['text_secondary']}; padding: 2px 4px;")
        tint_header.addWidget(self.label_tint)

        tint_box = QWidget()
        tbl = QVBoxLayout(tint_box)
        tbl.setContentsMargins(0, 0, 0, 0)
        tbl.setSpacing(6)
        hw = QWidget(); hw.setLayout(tint_header)
        tbl.addWidget(hw)

        self.slider_tint = ScrubSlider(dual=True)
        self.slider_tint.setMinimum(-50)
        self.slider_tint.setMaximum(50)
        self.slider_tint.setValue(0)
        self.slider_tint.valueChanged.connect(self.on_tint_slider_moved)
        self.slider_tint.sliderReleased.connect(self.on_tint_released)
        self.slider_tint.installEventFilter(self)
        tbl.addWidget(self.slider_tint)
        v.addWidget(tint_box)
        return sec

    def _build_export_footer(self) -> QWidget:
        sec = QWidget()
        sec.setObjectName("ExportFooter")
        sec.setAttribute(Qt.WA_StyledBackground, True)
        self._themed(
            sec,
            lambda: f"QWidget#ExportFooter {{ background: rgba(0, 0, 0, 0.08); }}",
        )
        v = QVBoxLayout(sec)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(8)

        v.addWidget(self._section_header("EXPORT"))

        # Format pills (JPEG / TIFF)
        pills_row = QHBoxLayout()
        pills_row.setSpacing(6)
        self.btn_export_jpeg = QPushButton("JPEG")
        self.btn_export_tiff = QPushButton("TIFF")
        for b in (self.btn_export_jpeg, self.btn_export_tiff):
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(28)
        self.btn_export_jpeg.setToolTip("Final film look (JPEG)")
        self.btn_export_tiff.setToolTip("Intermediate ACEScct log (TIFF for Resolve)")
        self.btn_export_jpeg.clicked.connect(lambda: self.set_export_mode(False))
        self.btn_export_tiff.clicked.connect(lambda: self.set_export_mode(True))
        pills_row.addWidget(self.btn_export_jpeg, 1)
        pills_row.addWidget(self.btn_export_tiff, 1)
        v.addLayout(pills_row)

        # Output path row — single flat shape (no nested button outline)
        out_row = QWidget()
        self._themed(out_row, lambda: (
            f"QWidget {{"
            f"  background: {C['bg_input']};"
            f"  border: 1px solid {C['border_input']};"
            f"  border-radius: 3px;"
            f"}}"
            f"QLabel, QPushButton {{ background: transparent; border: none; }}"
        ))
        out_row.setFixedHeight(28)
        out_row.setCursor(Qt.PointingHandCursor)
        ol = QHBoxLayout(out_row)
        ol.setContentsMargins(8, 0, 8, 0)
        ol.setSpacing(6)

        folder_ico = QLabel()
        folder_ico.setPixmap(svg_icon("assets/icons/folder.svg", "text_label", 12).pixmap(12, 12))
        folder_ico.setCursor(Qt.PointingHandCursor)
        folder_ico.mousePressEvent = lambda _: self.select_output_dir()
        ol.addWidget(folder_ico)

        self.label_output = QLabel(self.output_dir)
        self.label_output.setFont(theme.mono_font(10, QFont.Medium))
        self._themed(
            self.label_output,
            lambda: f"color: {C['text_label']}; background: transparent;",
        )
        self.label_output.setWordWrap(False)
        self.label_output.setTextFormat(Qt.PlainText)
        self.label_output.setCursor(Qt.PointingHandCursor)
        self.label_output.setToolTip(self.output_dir)
        self.label_output.mousePressEvent = lambda _: self.select_output_dir()
        ol.addWidget(self.label_output, 1)
        v.addWidget(out_row)

        # Process button + thin progress bar above it
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(2)
        self.progress_bar.setVisible(False)
        self._themed(self.progress_bar, lambda: (
            f"QProgressBar {{"
            f"  background: {C['border_input']};"
            f"  border: none; border-radius: 1px;"
            f"}}"
            f"QProgressBar::chunk {{"
            f"  background: {C['accent']}; border-radius: 1px;"
            f"}}"
        ))
        v.addWidget(self.progress_bar)

        self.btn_process_all = QPushButton("Process 0 / 0")
        self.btn_process_all.setEnabled(False)
        self.btn_process_all.setFixedHeight(40)
        self.btn_process_all.setCursor(Qt.PointingHandCursor)
        self._themed(self.btn_process_all, lambda: process_btn_qss())
        self.btn_process_all.clicked.connect(self.process_all_images)
        v.addWidget(self.btn_process_all)

        # Initialize pill state to default (JPEG)
        self.set_export_mode(False)
        return sec

    # ── status bar ──────────────────────────────────────────────────────
    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("StatusBar")
        bar.setAttribute(Qt.WA_StyledBackground, True)
        bar.setFixedHeight(22)
        self._themed(bar, lambda: (
            f"QWidget#StatusBar {{"
            f"  background: {C['bg_strip']};"
            f"  border-top: 1px solid {C['border_soft']};"
            f"}}"
        ))
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(12, 0, 12, 0)
        hl.setSpacing(14)

        self.status_dot = QLabel("●")
        self._themed(self.status_dot, lambda: f"color: {C['processed']};")
        hl.addWidget(self.status_dot)

        self.mode_label = QLabel("Ready")
        self.mode_label.setFont(theme.mono_font(10, QFont.Medium))
        self._themed(self.mode_label, lambda: f"color: {C['text_dim']};")
        hl.addWidget(self.mode_label)

        hl.addStretch(1)

        mod = "⌘" if sys.platform == "darwin" else "Ctrl"
        hints = [
            ("← →", "navigate"),
            ("↑ ↓", "rotate"),
            (f"{mod}O", "open"),
            (f"{mod}R", "reset"),
            (f"{mod}C / {mod}V", "copy · paste"),
            ("Esc", "clear selection"),
        ]
        for keys, desc in hints:
            chip = QLabel(f"{keys}  {desc}")
            chip.setFont(theme.mono_font(10, QFont.Medium))
            self._themed(chip, lambda: f"color: {C['text_dim']};")
            hl.addWidget(chip)

        return bar

    def _build_menu_bar(self):
        """Build the native menu bar."""
        from _version import __version__
        mb = self.menuBar()

        # ── Flashback / Help ──────────────────────────────────────────
        # "About" with AboutRole moves to the app menu automatically on macOS.
        # We put it in a Help menu so it appears somewhere on Windows/Linux too.
        help_menu = mb.addMenu("Help")

        act_about = QAction("About Flashback One35 v2", self)
        act_about.setMenuRole(QAction.MenuRole.AboutRole)
        act_about.triggered.connect(self.show_about)
        help_menu.addAction(act_about)

        # ── File ──────────────────────────────────────────────────────
        file_menu = mb.addMenu("File")

        act_open = QAction("Open…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)  # Cmd+O / Ctrl+O
        act_open.triggered.connect(self.open_files)
        file_menu.addAction(act_open)

        file_menu.addSeparator()

        act_export_jpg = QAction("Export JPGs", self)
        act_export_jpg.triggered.connect(self.export_as_jpeg)
        file_menu.addAction(act_export_jpg)

        act_export_tif = QAction("Export TIFs", self)
        act_export_tif.triggered.connect(self.export_as_tiff)
        file_menu.addAction(act_export_tif)

        file_menu.addSeparator()

        act_output_dir = QAction("Set Output Directory…", self)
        act_output_dir.triggered.connect(self.select_output_dir)
        file_menu.addAction(act_output_dir)

        file_menu.addSeparator()

        # ApplicationSpecificRole → macOS app menu; stays in File on Windows/Linux
        act_prefs = QAction("Advanced Settings", self)
        act_prefs.setMenuRole(QAction.MenuRole.ApplicationSpecificRole)
        act_prefs.setShortcut(QKeySequence("F12"))
        act_prefs.triggered.connect(self._toggle_advanced_settings)
        file_menu.addAction(act_prefs)

        # ── Edit ──────────────────────────────────────────────────────
        # Note: macOS automatically appends "Start Dictation" and "Emoji & Symbols"
        # to any menu titled exactly "Edit". Naming it differently avoids that.
        edit_menu = mb.addMenu("Adjustments")

        act_copy = QAction("Copy Settings", self)
        act_copy.setShortcut(QKeySequence.StandardKey.Copy)   # Cmd+C / Ctrl+C
        act_copy.triggered.connect(self.copy_settings)
        edit_menu.addAction(act_copy)

        act_paste = QAction("Paste Settings", self)
        act_paste.setShortcut(QKeySequence.StandardKey.Paste)  # Cmd+V / Ctrl+V
        act_paste.triggered.connect(self.paste_settings)
        edit_menu.addAction(act_paste)

        act_select_all = QAction("Select All for Paste", self)
        act_select_all.setShortcut(QKeySequence.StandardKey.SelectAll)  # Cmd+A / Ctrl+A
        act_select_all.triggered.connect(self._menu_select_all_paste)
        edit_menu.addAction(act_select_all)

        act_deselect = QAction("Deselect All for Paste", self)
        act_deselect.setShortcut(QKeySequence("Ctrl+D"))  # Cmd+D on macOS
        act_deselect.triggered.connect(self._menu_deselect_all_paste)
        edit_menu.addAction(act_deselect)

        edit_menu.addSeparator()

        act_reset = QAction("Reset Settings", self)
        act_reset.triggered.connect(self.reset_all_sliders)
        edit_menu.addAction(act_reset)

        # ── View ──────────────────────────────────────────────────────
        view_menu = mb.addMenu("View")

        act_zen = QAction("Zen Mode", self)
        act_zen.setIcon(self._char_icon("⛶"))
        act_zen.setShortcut(QKeySequence("Ctrl+Return"))
        act_zen.triggered.connect(self.enter_zen_mode)
        view_menu.addAction(act_zen)

    # ───────────────────────────────────────────────────────────────────
    # MENU ACTIONS
    # ───────────────────────────────────────────────────────────────────

    def _char_icon(self, char, size=16):
        """Render a Unicode character as a QIcon for use in menus."""
        px = QPixmap(size, size)
        px.fill(Qt.transparent)
        painter = QPainter(px)
        color = QApplication.palette().color(QPalette.ColorRole.WindowText)
        painter.setPen(color)
        font = painter.font()
        font.setPixelSize(size)
        painter.setFont(font)
        painter.drawText(px.rect(), Qt.AlignCenter, char)
        painter.end()
        return QIcon(px)

    def show_about(self):
        from _version import __version__
        QMessageBox.about(
            self,
            "About Flashback One35 v2",
            f"<b>Flashback One35 v2</b><br>"
            f"Version {__version__}<br><br>"
            "A RAW editor for Flashback film cameras.<br><br>"
            "© 2026 Flashback"
        )

    def _toggle_advanced_settings(self):
        if self.debug_panel.isVisible():
            self.debug_panel.hide()
        else:
            self.debug_panel.show()
            self.debug_panel.raise_()

    def _menu_select_all_paste(self):
        if not self.image_files:
            return
        self.thumbnail_strip.select_all_for_paste()
        count = len(self.thumbnail_strip.get_paste_selected_indices())
        self.mode_label.setText(f"{count} selected for paste")
        self.mode_label.setStyleSheet(f"color: {C['accent']};")
        QTimer.singleShot(2000, self.update_mode_label)

    def _menu_deselect_all_paste(self):
        self.thumbnail_strip.clear_paste_selection()
        self.mode_label.setText("Paste selection cleared")
        self.mode_label.setStyleSheet(f"color: {C['accent']};")
        QTimer.singleShot(1500, self.update_mode_label)

    def export_as_jpeg(self):
        self.set_export_mode(False)
        self.process_all_images()

    def export_as_tiff(self):
        self.set_export_mode(True)
        self.process_all_images()

    def center_window(self):
        frame_geo = self.frameGeometry()
        screen_geo = QApplication.primaryScreen().availableGeometry()
        frame_geo.moveCenter(screen_geo.center())
        self.move(frame_geo.topLeft())

    # ===================================================================
    # FILE MANAGEMENT
    # ===================================================================

    # Volume labels that identify a Flashback camera
    CAMERA_VOLUME_NAMES = {'ONE35 V2', 'ONE35'}

    def detect_camera(self):
        """Auto-detect Flashback camera by volume name (cross-platform)."""
        mount_points = []

        if sys.platform == 'darwin':
            volumes_path = Path("/Volumes")
            if volumes_path.exists():
                mount_points.extend(volumes_path.iterdir())
        elif sys.platform == 'win32':
            import string
            for letter in string.ascii_uppercase:
                drive = Path(f"{letter}:\\")
                if drive.exists():
                    mount_points.append(drive)
        else:  # Linux
            for base in (Path("/media"), Path("/mnt")):
                if base.exists():
                    mount_points.extend(base.iterdir())

        for mount in mount_points:
            if not mount.is_dir():
                continue
            if mount.name not in self.CAMERA_VOLUME_NAMES:
                continue
            dng_files = list(mount.glob("*.dng")) + list(mount.glob("*.DNG"))
            if dng_files:
                reply = QMessageBox.question(
                    self, "Camera Detected",
                    f"Found {len(dng_files)} DNG files on {mount.name}.\n\nLoad these files?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.load_image_files(dng_files)
            else:
                QMessageBox.information(
                    self, "Camera Connected",
                    f"{mount.name} is connected but contains no DNG files."
                )
            return

    def open_files(self):
        default_dir = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation) or str(Path.home())
        start_dir = self.app_settings.value("last_open_dir", default_dir)

        lower_exts = [f"*{ext}" for ext in self.SUPPORTED_EXTENSIONS]
        upper_exts = [f"*{ext.upper()}" for ext in self.SUPPORTED_EXTENSIONS]
        filter_string = f"Supported Images ({' '.join(lower_exts + upper_exts)})"

        files, _ = QFileDialog.getOpenFileNames(self, "Select Image Files", start_dir, filter_string)

        if files:
            new_dir = str(Path(files[0]).parent)
            self.app_settings.setValue("last_open_dir", new_dir)
            self.load_image_files([Path(f) for f in files])

    def load_image_files(self, image_files):
        if not image_files:
            return

        self.image_files = image_files
        self.current_index = 0
        self.image_cache.clear()
        self.thumbnail_strip.clear()

        self.btn_process_all.setEnabled(True)
        self.update_process_button_text()

        expected_thumb_width = 105
        layout_spacing = 5
        final_width = len(self.image_files) * (expected_thumb_width + layout_spacing)
        self.thumbnail_strip.container.setMinimumWidth(final_width)

        # Process main image synchronously FIRST (isolates Numba to main thread)
        self.load_current_image()

        if hasattr(self, 'loader_overlay'):
            self.loader_overlay.fade_in()
            self.loader_overlay.update_progress(0, len(self.image_files))

        self.thumbnail_worker = ThumbnailWorker(
            self.image_files,
            self.processor.lut_preview,
            self.processor.lut_full
        )

        self.thumbnail_worker.progress.connect(self.loader_overlay.update_progress)
        self.thumbnail_worker.thumbnail_ready.connect(self._add_thumbnail_to_ui)

        if hasattr(self, '_on_thumbnail_error'):
            self.thumbnail_worker.error.connect(self._on_thumbnail_error)
        if hasattr(self, '_on_thumbnails_finished'):
            self.thumbnail_worker.finished.connect(self._on_thumbnails_finished)

        self.thumbnail_worker.setStackSize(32 * 1024 * 1024)  # 32MB — Numba JIT needs deep stack
        self.thumbnail_worker.start()

    def _on_thumbnail_error(self, index, error_message):
        print(f"  ✗ Failed thumbnail {index}: {error_message}")
        try:
            if hasattr(self, 'loader_overlay') and self.loader_overlay.isVisible():
                self.loader_overlay.progress_label.setText(f"Error at {index}: {error_message}")
                QTimer.singleShot(1500, lambda: self.loader_overlay.update_progress(index + 1, len(self.image_files)))
        except Exception:
            pass

    def _on_thumbnails_finished(self):
        self.thumbnails_loading = False
        print("✓ Thumbnail generation complete!")
        self.thumbnail_strip.container.setUpdatesEnabled(True)
        if self.thumbnail_worker:
            self.thumbnail_worker.deleteLater()
            self.thumbnail_worker = None
        try:
            if hasattr(self, 'loader_overlay'):
                self.loader_overlay.clear_and_hide()
        except Exception:
            pass

    def add_image_files(self, new_files):
        """Append new images to the current session without resetting existing ones."""
        if not new_files:
            return

        existing_paths = {str(f) for f in self.image_files}
        files_to_add = [f for f in new_files if str(f) not in existing_paths]
        if not files_to_add:
            return

        offset = len(self.image_files)
        self.image_files.extend(files_to_add)

        self.btn_process_all.setEnabled(True)
        self.update_process_button_text()

        expected_thumb_width = 105
        layout_spacing = 5
        final_width = len(self.image_files) * (expected_thumb_width + layout_spacing)
        self.thumbnail_strip.container.setMinimumWidth(final_width)

        if hasattr(self, 'add_thumbnail_worker') and self.add_thumbnail_worker and self.add_thumbnail_worker.isRunning():
            self.add_thumbnail_worker._is_running = False
            self.add_thumbnail_worker.wait()

        if hasattr(self, 'loader_overlay'):
            self.loader_overlay.fade_in()
            self.loader_overlay.update_progress(0, len(files_to_add))

        self.add_thumbnail_worker = ThumbnailWorker(
            files_to_add,
            self.processor.lut_preview,
            self.processor.lut_full
        )
        self.add_thumbnail_worker.progress.connect(self.loader_overlay.update_progress)
        self.add_thumbnail_worker.thumbnail_ready.connect(
            lambda i, t, mid, off=offset: self._add_thumbnail_to_ui(i + off, t, mid)
        )
        self.add_thumbnail_worker.finished.connect(self._on_add_thumbnails_finished)
        self.add_thumbnail_worker.setStackSize(32 * 1024 * 1024)
        self.add_thumbnail_worker.start()

    def _on_add_thumbnails_finished(self):
        print("✓ Add-images thumbnail generation complete!")
        if hasattr(self, 'add_thumbnail_worker') and self.add_thumbnail_worker:
            self.add_thumbnail_worker.deleteLater()
            self.add_thumbnail_worker = None
        try:
            if hasattr(self, 'loader_overlay'):
                self.loader_overlay.clear_and_hide()
        except Exception:
            pass
        self.update_mode_label()

    # ===================================================================
    # THUMBNAIL MANAGEMENT
    # ===================================================================

    def update_thumbnail_for_settings(self, index, settings):
        if not self.image_files or index >= len(self.image_files):
            return

        file_path = str(self.image_files[index])

        if index == self.current_index:
            try:
                img_display = self.processor.render_preview()
                if img_display is not None:
                    h, w = img_display.shape[:2]
                    scale = 70 / h
                    new_w = int(w * scale)
                    thumb_array = cv2.resize(img_display, (new_w, 70), interpolation=cv2.INTER_LINEAR)
                    self.thumbnail_cache[file_path] = thumb_array
                    self.thumbnail_strip.update_thumbnail(index, thumb_array)
                    return
            except Exception as e:
                print(f"  ✗ Failed to update current thumbnail: {e}")
        else:
            try:
                if file_path in self.image_cache:
                    temp_processor = FlashbackProcessor(None)
                    temp_processor.lut_preview = self.processor.lut_preview
                    temp_processor.lut_full = self.processor.lut_full
                    temp_processor.intermediate_acescct = self.image_cache[file_path].copy()
                    temp_processor.current_file = file_path
                    temp_processor.user_settings = settings.copy()
                    img_display = temp_processor.render_preview()
                    if img_display is not None:
                        h, w = img_display.shape[:2]
                        scale = 70 / h
                        new_w = int(w * scale)
                        thumb_array = cv2.resize(img_display, (new_w, 70), interpolation=cv2.INTER_LINEAR)
                        self.thumbnail_cache[file_path] = thumb_array
                        self.thumbnail_strip.update_thumbnail(index, thumb_array)
                        return
            except Exception as e:
                print(f"  ✗ Failed to update thumbnail {index}: {e}")

    def update_current_thumbnail(self, img_array=None):
        if not self.image_files:
            return
        file_path = str(self.image_files[self.current_index])
        try:
            if img_array is None:
                img_array = self.processor.render_preview()
            if img_array is not None:
                h, w = img_array.shape[:2]
                scale = 70 / h
                new_w = int(w * scale)
                thumb_array = cv2.resize(img_array, (new_w, 70), interpolation=cv2.INTER_LINEAR)
                self.thumbnail_cache[file_path] = thumb_array
                self.thumbnail_strip.update_thumbnail(self.current_index, thumb_array)
        except Exception as e:
            _timing_print(f"  [Thumbnail] Update failed silently: {e}")

    def _add_thumbnail_to_ui(self, index, thumb_array, intermediate=None):
        self.thumbnail_strip.container.setUpdatesEnabled(False)
        filename = self.image_files[index].name if self.image_files and index < len(self.image_files) else None
        self.thumbnail_strip.add_thumbnail(thumb_array, index, filename=filename)
        self.thumbnail_strip.container.setUpdatesEnabled(True)

        if self.image_files and index < len(self.image_files):
            file_path = str(self.image_files[index])
            if self._is_processed(file_path):
                self.thumbnail_strip.set_processed(index, True)
            if index == self.current_index:
                self.thumbnail_strip.set_current_index(index)

        if intermediate is not None and self.image_files:
            file_path = str(self.image_files[index])
            self.image_cache[file_path] = intermediate

    def on_thumbnail_click(self, index):
        if 0 <= index < len(self.image_files):
            self.current_index = index
            self.load_current_image()

    def on_thumbnail_right_click(self, index):
        if 0 <= index < len(self.image_files):
            self.thumbnail_strip.toggle_process_selection(index)
            self.update_process_button_text()
            count = len(self.thumbnail_strip.get_process_selected_indices())
            if count > 0:
                self.mode_label.setText(f"{count} selected for processing")
            else:
                self.mode_label.setText("All images will be processed")
            self.mode_label.setStyleSheet(f"color: {C['accent']};")
            QTimer.singleShot(2000, self.update_mode_label)

    def on_thumbnail_paste_selected(self, index, is_selected):
        count = len(self.thumbnail_strip.get_paste_selected_indices())
        if count > 0:
            paste_key = "Cmd+V" if sys.platform == 'darwin' else "Ctrl+V"
            self.mode_label.setText(f"{count} selected for paste ({paste_key})")
            self.mode_label.setStyleSheet(f"color: {C['accent']};")
        else:
            self.update_mode_label()

    def update_process_button_text(self):
        self.btn_process_all.setStyleSheet(process_btn_qss())
        selected = self.thumbnail_strip.get_process_selected_indices()
        if selected:
            self.btn_process_all.setText(f"Process {len(selected)} / {len(self.image_files)}")
        else:
            self.btn_process_all.setText(f"Process {len(self.image_files)} / {len(self.image_files)}")

    def _set_process_button_done(self, count: int):
        """Post-export state: checkmark + 'N frames processed'."""
        self.btn_process_all.setText(f"✓  {count} frame{'s' if count != 1 else ''} processed")
        self.btn_process_all.setStyleSheet(f"""
            QPushButton {{
                background: {C['bg_input']};
                color: {C['text_label']};
                border: 1px solid {C['border_input']};
                border-radius: 3px;
                font-family: "{UI_FONT}";
                font-size: 12px;
                font-weight: 600;
                padding: 10px 12px;
            }}
        """)

    # ===================================================================
    # IMAGE LOADING & DISPLAY
    # ===================================================================

    def load_current_image(self):
        if not self.image_files:
            self.label_filename.setText("")
            self.label_counter.setText("0 / 0")
            return

        file_path = str(self.image_files[self.current_index])
        self.label_filename.setText(Path(file_path).name)
        self.label_counter.setText(f"{self.current_index + 1} / {len(self.image_files)}")
        if hasattr(self, 'thumbnail_strip'):
            self.thumbnail_strip.set_current_index(self.current_index)

        if file_path in self.image_settings:
            settings = self.image_settings[file_path]
            self.processor.set_settings(settings)
            self.chk_wb_link.blockSignals(True)
            self.chk_wb_link.setChecked(settings.get('auto_tint', False))
            self.chk_wb_link.blockSignals(False)
            self.update_sliders_from_processor()
        else:
            self.processor.user_settings = {'exposure_ev': 0.0, 'wb_temp': 0, 'tint': 0.0}
            self.chk_wb_link.blockSignals(True)
            self.chk_wb_link.setChecked(False)
            self.chk_wb_link.blockSignals(False)
            self.update_sliders_from_processor()

        self.processor.preview_mode = 'hq'

        if file_path in self.image_cache:
            self.processor.intermediate_acescct = self.image_cache[file_path]
            self.processor.current_file = file_path
            img_array = self.processor.render_preview()
            self.display_image(img_array)
            self.update_current_thumbnail(img_array)
            self.update_mode_label()
        else:
            print(f"[Load] Image not in cache, loading from disk...")
            img_array = self.processor.load_image(file_path)
            if img_array is not None:
                self.image_cache[file_path] = self.processor.intermediate_acescct.copy()
                self.update_current_thumbnail(img_array)
                self.display_image(img_array)
                self.save_current_settings()
                self.update_mode_label()
            else:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{file_path}")

    def display_image(self, img_array):
        if img_array is None:
            return
        if hasattr(self, 'zen_overlay') and self.zen_overlay.isVisible():
            # Build the pixmap once and hand it directly to the zen overlay.
            # Skipping image_label._update_display() avoids a second SmoothTransformation
            # scale on the hidden main window — that was doubling the per-frame cost.
            img_8bit = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
            h, w, c = img_8bit.shape
            q_image = QImage(img_8bit.data, w, h, c * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label._original_pixmap = pixmap  # keep in sync for when zen closes
            self.zen_overlay.update_preview(pixmap)
        else:
            self.image_label.set_image(img_array)

    def update_sliders_from_processor(self):
        settings = self.processor.user_settings

        self.slider_exposure.blockSignals(True)
        self.slider_wb.blockSignals(True)
        self.slider_tint.blockSignals(True)

        self.slider_exposure.setValue(int(settings['exposure_ev'] * 10))
        self.slider_wb.setValue(int(settings['wb_temp']))
        self.slider_tint.setValue(int(round(settings['tint'] * 5)))

        self.label_exposure.setText(f"{settings['exposure_ev']:.1f} EV")
        temp_absolute = 5600 + int(settings['wb_temp'])
        self.label_wb.setText(f"{temp_absolute} K")
        self.label_tint.setText(f"{int(round(settings['tint'] * 5)):+d}")

        self.slider_exposure.blockSignals(False)
        self.slider_wb.blockSignals(False)
        self.slider_tint.blockSignals(False)

        # Keep the manual tint offset coherent with the newly loaded settings
        if self.chk_wb_link.isChecked():
            self._tint_manual_offset = settings['tint'] - self._coupled_tint(settings['wb_temp'])

    # ===================================================================
    # SLIDER HANDLERS
    # ===================================================================

    def _on_slider_render_tick(self):
        img_array = self.processor._render_fast()
        if img_array is not None:
            self.display_image(img_array)
            self.update_mode_label()

    def on_exposure_slider_moved(self, value):
        self.processor.preview_mode = 'fast'
        ev = value / 10.0
        self.label_exposure.setText(f"{ev:.1f} EV")
        self.processor.user_settings['exposure_ev'] = ev
        self._slider_render_timer.start()

    def on_exposure_released(self):
        self._slider_render_timer.stop()
        self.processor.preview_mode = 'hq'
        img_array = self.processor._render_fast()
        if img_array is not None:
            self.display_image(img_array)
            self.update_current_thumbnail(img_array)
            self.update_mode_label()
        self.save_current_settings()

    def _on_wb_link_toggled(self, checked):
        self.save_current_settings()

    def _coupled_tint(self, wb_offset):
        """Coupled tint value for a given WB offset from neutral (5600K).
        Linear ±6: 0 → 0,  -2000 → +6,  +2000 → -6.
        Returns tint in actual units (same as processor.user_settings['tint']).
        """
        return wb_offset / 2000.0 * -6.0

    def _apply_wb_tint_link(self, wb_value):
        """When the link is active: compute coupled tint + manual offset, update
        the tint slider/label without triggering on_tint_slider_moved, and update
        the processor setting.  Returns the new tint value."""
        coupled = self._coupled_tint(wb_value)
        new_tint = max(-10.0, min(10.0, coupled + self._tint_manual_offset))
        self.processor.user_settings['tint'] = new_tint
        self.slider_tint.blockSignals(True)
        self.slider_tint.setValue(int(round(new_tint * 5)))
        self.label_tint.setText(f"{int(round(new_tint * 5)):+d}")
        self.slider_tint.blockSignals(False)
        return new_tint

    def on_wb_slider_moved(self, value):
        self.processor.preview_mode = 'fast'
        temp_absolute = 5600 + value
        self.label_wb.setText(f"{temp_absolute} K")

        if self.chk_wb_link.isChecked():
            self._apply_wb_tint_link(value)
            self.processor.user_settings['wb_temp'] = value
        else:
            self.processor.user_settings['wb_temp'] = value

        self._slider_render_timer.start()

    def on_wb_released(self):
        self._slider_render_timer.stop()
        self.processor.preview_mode = 'hq'
        img_array = self.processor._render_fast()
        if img_array is not None:
            self.display_image(img_array)
            self.update_current_thumbnail(img_array)
            self.update_mode_label()
        self.save_current_settings()

    def on_tint_slider_moved(self, value):
        self.processor.preview_mode = 'fast'
        tint = value / 5.0
        self.label_tint.setText(f"{value:+d}")
        if self.chk_wb_link.isChecked():
            self._tint_manual_offset = tint - self._coupled_tint(self.slider_wb.value())
        self.processor.user_settings['tint'] = tint
        self._slider_render_timer.start()

    def on_tint_released(self):
        self._slider_render_timer.stop()
        self.processor.preview_mode = 'hq'
        img_array = self.processor._render_fast()
        if img_array is not None:
            self.display_image(img_array)
            self.update_current_thumbnail(img_array)
            self.update_mode_label()
        self.save_current_settings()

    def adjust_exposure(self, delta):
        current = self.slider_exposure.value()
        self.slider_exposure.setValue(current + int(delta * 10))
        self.on_exposure_released()

    def adjust_wb(self, delta):
        current = self.slider_wb.value()
        self.slider_wb.setValue(current + delta)
        self.on_wb_released()

    def adjust_tint(self, delta):
        current = self.slider_tint.value()
        self.slider_tint.setValue(current + delta)
        self.on_tint_released()

    def reset_all_sliders(self):
        self.slider_exposure.blockSignals(True)
        self.slider_wb.blockSignals(True)
        self.slider_tint.blockSignals(True)

        self.label_exposure.setText("0.0 EV")
        self.label_wb.setText("5600 K")
        self.label_tint.setText("+0")
        self.slider_exposure.setValue(0)
        self.slider_wb.setValue(0)
        self.slider_tint.setValue(0)

        self.slider_exposure.blockSignals(False)
        self.slider_wb.blockSignals(False)
        self.slider_tint.blockSignals(False)

        self._tint_manual_offset = 0.0
        self.processor.user_settings = {'exposure_ev': 0.0, 'wb_temp': 0, 'tint': 0.0}
        img_array = self.processor.render_preview()
        self.display_image(img_array)
        self.update_current_thumbnail(img_array)
        self.save_current_settings()
        self.update_mode_label()

    def update_mode_label(self):
        """Default status line when the processor is idle."""
        if self.image_files:
            total = len(self.image_files)
            processed = sum(
                1 for p in self.image_files if self._is_processed(str(p))
            )
            pending = total - processed
            self.mode_label.setText(f"Ready   {processed} processed · {pending} pending")
        else:
            self.mode_label.setText("Ready")
        self.mode_label.setStyleSheet(f"color: {C['text_dim']};")
        self.status_dot.setStyleSheet(f"color: {C['processed']};")

    def _is_processed(self, file_path: str) -> bool:
        """True if an export file exists in output_dir for this source image."""
        try:
            base = Path(file_path).stem
            for suffix in ("_processed.jpg", "_intermediate.tif"):
                if os.path.exists(os.path.join(self.output_dir, base + suffix)):
                    return True
        except Exception:
            pass
        return False

    def save_current_settings(self):
        if self.image_files:
            file_path = str(self.image_files[self.current_index])
            settings = self.processor.get_settings()
            settings['auto_tint'] = self.chk_wb_link.isChecked()
            self.image_settings[file_path] = settings

    # ===================================================================
    # COPY / PASTE SETTINGS
    # ===================================================================

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()

    def copy_settings(self):
        self.settings_clipboard = self.processor.get_settings()
        self.settings_clipboard['auto_tint'] = self.chk_wb_link.isChecked()
        self.mode_label.setText("Settings copied")
        self.mode_label.setStyleSheet(f"color: {C['accent']};")
        QTimer.singleShot(2000, self.update_mode_label)

    def paste_settings(self):
        if not self.settings_clipboard:
            return

        paste_selected = self.thumbnail_strip.get_paste_selected_indices()

        if paste_selected:
            indices_to_apply = sorted(paste_selected)
            total = len(indices_to_apply)

            success_count = 0
            self.mode_label.setText(f"Applying to {total} images...")
            self.mode_label.setStyleSheet(f"color: {C['accent']};")
            QApplication.processEvents()

            for idx in indices_to_apply:
                file_path = str(self.image_files[idx])
                self.image_settings[file_path] = self.settings_clipboard.copy()
                success_count += 1
                if success_count % 5 == 0:
                    self.mode_label.setText(f"Applied {success_count}/{total}...")
                    QApplication.processEvents()

            if self.current_index in paste_selected:
                img_array = self.processor.set_settings(self.settings_clipboard)
                self.chk_wb_link.blockSignals(True)
                self.chk_wb_link.setChecked(self.settings_clipboard.get('auto_tint', False))
                self.chk_wb_link.blockSignals(False)
                self.update_sliders_from_processor()
                self.display_image(img_array)
                self.update_thumbnail_for_settings(self.current_index, self.settings_clipboard)

            self.mode_label.setText(f"Updating thumbnails...")
            for idx in indices_to_apply:
                if idx != self.current_index:
                    self.update_thumbnail_for_settings(idx, self.settings_clipboard)

            self.thumbnail_strip.clear_paste_selection()
            self.mode_label.setText(f"Settings applied to {success_count} images")
            QTimer.singleShot(2000, self.update_mode_label)

        else:
            img_array = self.processor.set_settings(self.settings_clipboard)
            self.chk_wb_link.blockSignals(True)
            self.chk_wb_link.setChecked(self.settings_clipboard.get('auto_tint', False))
            self.chk_wb_link.blockSignals(False)
            self.update_sliders_from_processor()
            self.display_image(img_array)
            self.save_current_settings()
            self.update_mode_label()
            self.update_thumbnail_for_settings(self.current_index, self.settings_clipboard)
            self.mode_label.setText("Settings pasted")
            self.mode_label.setStyleSheet(f"color: {C['accent']};")
            QTimer.singleShot(2000, self.update_mode_label)

    # ===================================================================
    # EXPORT
    # ===================================================================

    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir)
        if directory:
            self.output_dir = directory
            self.label_output.setText(directory)
            self.label_output.setToolTip(directory)

    def set_export_mode(self, tiff: bool):
        """Select JPEG or TIFF export; sync both pills and the Process button label."""
        self.export_tiff_mode = tiff
        self.btn_export_jpeg.setChecked(not tiff)
        self.btn_export_tiff.setChecked(tiff)
        self.btn_export_jpeg.setStyleSheet(format_pill_qss(not tiff))
        self.btn_export_tiff.setStyleSheet(format_pill_qss(tiff))
        if hasattr(self, "btn_process_all") and hasattr(self, "thumbnail_strip"):
            self.update_process_button_text()

    def process_all_images(self):
        if not self.image_files:
            return

        selected_indices = self.thumbnail_strip.get_process_selected_indices()
        if selected_indices:
            indices_to_process = sorted(selected_indices)
        else:
            indices_to_process = list(range(len(self.image_files)))

        # Low disk space: inline warning in the status bar, no modal.
        mb_per_image = 50 if self.export_tiff_mode else 5
        required_mb = len(indices_to_process) * mb_per_image
        try:
            free_mb = shutil.disk_usage(self.output_dir).free // (1024 * 1024)
            if free_mb < required_mb:
                self.mode_label.setText(
                    f"Low disk space: ~{required_mb} MB needed, {free_mb} MB free"
                )
                self.mode_label.setStyleSheet(f"color: {C['accent']};")
        except OSError:
            pass

        self.progress_bar.setMaximum(len(indices_to_process))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_process_all.setEnabled(False)

        success_count = 0
        total = len(indices_to_process)

        for i, idx in enumerate(indices_to_process):
            file_path = str(self.image_files[idx])

            try:
                self.progress_bar.setValue(i)
                self.btn_process_all.setText(f"Processing {i + 1} / {total}")
                self.mode_label.setText(f"Processing {i+1}/{total}...")
                QApplication.processEvents()

                if file_path in self.image_cache:
                    self.processor.intermediate_acescct = self.image_cache[file_path].copy()
                    self.processor.current_file = file_path
                else:
                    self.processor.load_image(file_path)
                    self.image_cache[file_path] = self.processor.intermediate_acescct.copy()

                if file_path in self.image_settings:
                    self.processor.set_settings(self.image_settings[file_path])

                base_name = Path(file_path).stem
                if self.export_tiff_mode:
                    output_path = os.path.join(self.output_dir, f"{base_name}_intermediate.tif")
                else:
                    output_path = os.path.join(self.output_dir, f"{base_name}_processed.jpg")

                if export_image(self.processor, output_path, as_tiff=self.export_tiff_mode):
                    success_count += 1
                    self.thumbnail_strip.set_processed(idx, True)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                traceback.print_exc()

            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

        self.progress_bar.setVisible(False)
        self.btn_process_all.setEnabled(True)
        self.load_current_image()
        self.update_mode_label()

        if success_count == total and total > 0:
            self._set_process_button_done(success_count)
        else:
            self.update_process_button_text()

    # ===================================================================
    # DEBUG / REFRESH
    # ===================================================================

    def refresh_from_debug(self):
        print("Refreshing...")
        if self.processor and self.processor.intermediate_acescct is not None:
            img_array = self.processor.render_preview()
            self.display_image(img_array)
            self.update_current_thumbnail(img_array)

    def reload_current_image(self):
        if not self.image_files:
            return
        file_path = str(self.image_files[self.current_index])
        if file_path in self.image_cache:
            del self.image_cache[file_path]
        self.load_current_image()

    # ===================================================================
    # KEYBOARD & DRAG/DROP
    # ===================================================================

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_C and (event.modifiers() & Qt.ControlModifier):
            if self.image_files:
                self.copy_settings()
            event.accept()
        elif event.key() == Qt.Key_V and (event.modifiers() & Qt.ControlModifier):
            if self.image_files and self.settings_clipboard:
                self.paste_settings()
            event.accept()
        elif event.key() == Qt.Key_A and (event.modifiers() & Qt.ControlModifier):
            if self.image_files:
                self.thumbnail_strip.select_all_for_paste()
                count = len(self.thumbnail_strip.get_paste_selected_indices())
                self.mode_label.setText(f"{count} selected for paste")
                self.mode_label.setStyleSheet(f"color: {C['accent']};")
                QTimer.singleShot(2000, self.update_mode_label)
            event.accept()
        elif event.key() == Qt.Key_Escape:
            self.thumbnail_strip.clear_paste_selection()
            self.mode_label.setText("Paste selection cleared")
            self.mode_label.setStyleSheet(f"color: {C['accent']};")
            QTimer.singleShot(1500, self.update_mode_label)
            event.accept()
        elif event.key() == Qt.Key_Left:
            if self.image_files and self.current_index > 0:
                self.current_index -= 1
                self.load_current_image()
            event.accept()
        elif event.key() == Qt.Key_Right:
            if self.image_files and self.current_index < len(self.image_files) - 1:
                self.current_index += 1
                self.load_current_image()
            event.accept()
        elif event.key() == Qt.Key_F12:
            if self.debug_panel.isVisible():
                self.debug_panel.hide()
            else:
                self.debug_panel.show()
                self.debug_panel.raise_()
            event.accept()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        # Needs a real native window; defer one event loop pass so the
        # NSWindow is fully constructed before we drive AppKit against it.
        if not getattr(self, "_native_chrome_applied", False):
            self._native_chrome_applied = True

            def _do_apply():
                try:
                    from ui import native_chrome
                    native_chrome.apply(self, theme.current_theme())
                except Exception as e:
                    print(f"[native_chrome] apply failed: {e}")

            QTimer.singleShot(0, _do_apply)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'loader_overlay') and self.loader_overlay.isVisible():
            self.loader_overlay.setGeometry(self.rect())
        if hasattr(self, 'drag_overlay') and self.drag_overlay.isVisible():
            self._update_drag_overlay_geometry()

    def _update_drag_overlay_geometry(self):
        """Position the two drag overlays: upper = replace, lower = add (thumbnail strip)."""
        cw = self.centralWidget()
        if cw is None:
            return
        cw_w = cw.width()
        cw_h = cw.height()
        m = 8  # margin

        if self.image_files and hasattr(self, 'drag_overlay_add'):
            # Split at the thumbnail strip top edge; add-overlay fills the full strip.
            strip = self.thumbnail_strip.parentWidget() or self.thumbnail_strip
            strip_top = strip.mapTo(cw, QPoint(0, 0)).y()
            strip_h = strip.height()
            self.drag_overlay.setGeometry(m, m, cw_w - 2 * m, max(40, strip_top - 2 * m))
            self.drag_overlay_add.setGeometry(0, strip_top, cw_w, strip_h)
        else:
            # No images loaded — full-area replace overlay only
            self.drag_overlay.setGeometry(m, m, cw_w - 2 * m, cw_h - 2 * m)

    def _set_drag_hover(self, over_strip):
        """Highlight the active drag zone and dim the inactive one."""
        if not self.image_files:
            return
        if over_strip:
            self.drag_overlay.setStyleSheet(self._drag_style_dim)
            self.drag_overlay_add.setStyleSheet(self._drag_style_active)
        else:
            self.drag_overlay.setStyleSheet(self._drag_style_active)
            self.drag_overlay_add.setStyleSheet(self._drag_style_dim)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(self.SUPPORTED_EXTENSIONS):
                    self._update_drag_overlay_geometry()
                    self.drag_overlay.raise_()
                    self.drag_overlay.show()
                    if self.image_files:
                        self.drag_overlay_add.raise_()
                        self.drag_overlay_add.show()
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        if not (self.drag_overlay.isVisible() and self.image_files):
            return
        cw = self.centralWidget()
        strip_top = self.thumbnail_strip.mapTo(cw, QPoint(0, 0)).y()
        pos_in_cw = cw.mapFrom(self, event.position().toPoint())
        self._set_drag_hover(pos_in_cw.y() >= strip_top)
        event.acceptProposedAction()

    def dropEvent(self, event):
        self.drag_overlay.hide()
        self.drag_overlay_add.hide()
        # Restore default styles
        self.drag_overlay.setStyleSheet(self._drag_style_active)
        self.drag_overlay_add.setStyleSheet(self._drag_style_dim)

        urls = event.mimeData().urls()
        image_files = []
        for url in urls:
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(self.SUPPORTED_EXTENSIONS):
                    image_files.append(Path(file_path))

        if not image_files:
            event.ignore()
            return

        # Determine drop zone: below thumbnail strip top → add; above → replace
        cw = self.centralWidget()
        strip_top = self.thumbnail_strip.mapTo(cw, QPoint(0, 0)).y()
        pos_in_cw = cw.mapFrom(self, event.position().toPoint())
        drop_on_strip = self.image_files and (pos_in_cw.y() >= strip_top)

        if drop_on_strip:
            self.add_image_files(image_files)
        else:
            new_dir = str(image_files[0].parent)
            self.app_settings.setValue("last_open_dir", new_dir)
            self.load_image_files(image_files)

        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.drag_overlay.hide()
        self.drag_overlay_add.hide()
        self.drag_overlay.setStyleSheet(self._drag_style_active)
        self.drag_overlay_add.setStyleSheet(self._drag_style_dim)
        super().dragLeaveEvent(event)
