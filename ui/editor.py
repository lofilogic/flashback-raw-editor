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
from core.config import _timing_print

from .widgets import (
    ThumbnailWorker, ThumbnailWidget, ThumbnailStrip,
    FadeOverlayWidget, LoaderOverlay, ZoomableImageWidget,
)
from .debug_panel import DebugPanel


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

        self.thumbnails_loading = False
        self.thumbnail_worker = None
        self.add_thumbnail_worker = None
        self._thumbnails_dirty = set()

        self._tint_manual_offset = 0.0  # user's manual tint correction on top of WB coupling

        self.pending_render = False

        lut_path = resource_path("assets/luts/look.cube")
        if not os.path.exists(lut_path):
            lut_path = None
        self.processor = FlashbackProcessor(lut_path)

        self.init_ui()

        self.debug_panel = DebugPanel(self.processor, self)
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
        self.label_wb.setText("5600K")
        self.processor.user_settings['wb_temp'] = 0.0
        if self.chk_wb_link.isChecked():
            self._tint_manual_offset = 0.0
            self.slider_tint.blockSignals(True)
            self.slider_tint.setValue(0)
            self.slider_tint.blockSignals(False)
            self.label_tint.setText("+0.0")
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
        self.label_tint.setText("+0.0")
        self.processor.user_settings['tint'] = 0.0
        self.processor.preview_mode = 'hq'
        img_array = self.processor.render_preview()
        if img_array is not None:
            self.display_image(img_array)
            self.update_current_thumbnail(img_array)
            self.save_current_settings()
            self.update_mode_label()

    # ===================================================================
    # UI CONSTRUCTION
    # ===================================================================

    def init_ui(self):
        self.setWindowTitle("Flashback One35 v2 Editor")
        self.resize(1100, 600)
        QTimer.singleShot(0, self.center_window)

        roboto_path = resource_path("assets/fonts/Roboto.ttf")
        if os.path.exists(roboto_path):
            QFontDatabase.addApplicationFont(roboto_path)
        font_family = "Roboto" if QFontDatabase.hasFamily("Roboto") else "Arial"
        font = QFont(font_family)
        font.setPointSize(10)
        self.setFont(font)

        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #313131;")
        self.setCentralWidget(main_widget)
        self.setAcceptDrops(True)

        # Drag overlays (replace + add)
        _drag_style_active = """
            QFrame {
                background-color: rgba(255, 138, 53, 0.25);
                border: 3px solid #FF8A35;
                border-radius: 10px;
            }
            QLabel {
                background: transparent;
                border: none;
                color: #FF8A35;
                font-size: 18px;
                font-weight: bold;
            }
        """
        _drag_style_dim = """
            QFrame {
                background-color: rgba(255, 138, 53, 0.07);
                border: 2px dashed #888;
                border-radius: 10px;
            }
            QLabel {
                background: transparent;
                border: none;
                color: #888;
                font-size: 18px;
                font-weight: bold;
            }
        """
        self._drag_style_active = _drag_style_active
        self._drag_style_dim = _drag_style_dim

        self.drag_overlay = QFrame(main_widget)
        self.drag_overlay.setStyleSheet(_drag_style_active)
        drag_layout = QVBoxLayout(self.drag_overlay)
        drag_label = QLabel("Drop DNG files here")
        drag_label.setAlignment(Qt.AlignCenter)
        drag_layout.addWidget(drag_label)
        self.drag_overlay.hide()

        self.drag_overlay_add = QFrame(main_widget)
        self.drag_overlay_add.setStyleSheet(_drag_style_dim)
        add_drag_layout = QVBoxLayout(self.drag_overlay_add)
        add_drag_label = QLabel("Add images")
        add_drag_label.setAlignment(Qt.AlignCenter)
        add_drag_layout.addWidget(add_drag_label)
        self.drag_overlay_add.hide()

        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 10, 20, 10)
        main_layout.setSpacing(20)

        main_vlayout = QVBoxLayout()
        main_vlayout.setContentsMargins(0, 0, 0, 0)
        main_vlayout.setSpacing(5)
        main_layout.addLayout(main_vlayout, 1)

        # === TOP SECTION: Image + Controls ===
        top_section = QWidget()
        top_layout = QHBoxLayout(top_section)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        top_layout.addStretch(1)

        # Image display
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)

        self.image_label = ZoomableImageWidget()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #313131;
                border-radius: 12px;
            }
        """)
        image_layout.addWidget(self.image_label, 1)

        self.zen_btn = QPushButton("⛶")
        self.zen_btn.setFixedSize(30, 30)
        self.zen_btn.setToolTip("Zen Mode (Fullscreen)")
        self.zen_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d; border-radius: 4px; color: #d0d0d0; font-size: 16px;
            }
            QPushButton:hover { background-color: #4d4d4d; }
        """)
        self.zen_btn.clicked.connect(self.enter_zen_mode)

        rotate_layout = QHBoxLayout()
        rotate_layout.setAlignment(Qt.AlignCenter)
        rotate_layout.setSpacing(15)
        rotate_layout.addWidget(self.zen_btn)

        self.btn_rotate_ccw = QPushButton("↺")
        self.btn_rotate_ccw.setFixedSize(36, 36)
        self.btn_rotate_ccw.setStyleSheet("""
            QPushButton { background-color: #3d3d3d; color: #626262; border: none; border-radius: 8px; font-size: 16px; }
            QPushButton:hover { background-color: #4a4a4a; }
        """)
        self.btn_rotate_ccw.clicked.connect(self.rotate_counterclockwise)
        rotate_layout.addWidget(self.btn_rotate_ccw)

        self.btn_rotate_cw = QPushButton("↻")
        self.btn_rotate_cw.setFixedSize(36, 36)
        self.btn_rotate_cw.setStyleSheet("""
            QPushButton { background-color: #3d3d3d; color: #626262; border: none; border-radius: 8px; font-size: 16px; }
            QPushButton:hover { background-color: #4a4a4a; }
        """)
        self.btn_rotate_cw.clicked.connect(self.rotate_clockwise)
        rotate_layout.addWidget(self.btn_rotate_cw)

        image_layout.addSpacing(6)
        image_layout.addLayout(rotate_layout)
        top_layout.addWidget(image_container, 2)
        top_layout.addStretch(1)

        # Controls panel
        right_wrapper = QWidget()
        right_wrapper_layout = QVBoxLayout(right_wrapper)
        right_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        right_wrapper_layout.addStretch()

        right_container = QWidget()
        right_container.setFixedWidth(280)
        self.right_container = right_container
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(20, 0, 0, 0)
        right_layout.setSpacing(5)
        right_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # Folder / Camera icons
        icons_layout = QHBoxLayout()
        icons_layout.setAlignment(Qt.AlignCenter)
        icons_layout.setSpacing(15)

        self.btn_open = QPushButton()
        self.btn_open.setFixedSize(44, 44)
        self.btn_open.setIcon(QPixmap(resource_path("assets/icons/folder.png")))
        self.btn_open.setIconSize(QSize(24, 24))
        self.btn_open.setStyleSheet("""
            QPushButton { background-color: #3d3d3d; border: none; border-radius: 10px; }
            QPushButton:hover { background-color: #4a4a4a; }
        """)
        self.btn_open.clicked.connect(self.open_files)
        icons_layout.addWidget(self.btn_open)

        self.btn_detect_camera = QPushButton()
        self.btn_detect_camera.setFixedSize(44, 44)
        self.btn_detect_camera.setIcon(QPixmap(resource_path("assets/icons/camera.png")))
        self.btn_detect_camera.setIconSize(QSize(24, 24))
        self.btn_detect_camera.setStyleSheet("""
            QPushButton { background-color: #3d3d3d; border: none; border-radius: 10px; }
            QPushButton:hover { background-color: #4a4a4a; }
        """)
        self.btn_detect_camera.clicked.connect(self.detect_camera)
        icons_layout.addWidget(self.btn_detect_camera)

        right_layout.addLayout(icons_layout)
        right_layout.addSpacing(35)

        # === SLIDERS ===

        # Exposure
        exp_value = QLabel("0.0 EV")
        exp_value.setStyleSheet("color: #626262; font-size: 14px;")
        exp_value.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(exp_value)
        self.label_exposure = exp_value
        right_layout.addSpacing(5)

        exp_row = QHBoxLayout()
        exp_row.setSpacing(8)

        self.btn_exp_minus = QPushButton("◀")
        self.btn_exp_minus.setFixedSize(24, 24)
        self.btn_exp_minus.setStyleSheet("QPushButton { background-color: transparent; color: #555; border: none; font-size: 12px; padding: 0; } QPushButton:hover { color: #777; }")
        self.btn_exp_minus.clicked.connect(lambda: self.adjust_exposure(-0.1))
        exp_row.addWidget(self.btn_exp_minus)

        self.slider_exposure = QSlider(Qt.Horizontal)
        self.slider_exposure.setMinimum(-20)
        self.slider_exposure.setMaximum(20)
        self.slider_exposure.setValue(0)
        self.slider_exposure.valueChanged.connect(self.on_exposure_slider_moved)
        self.slider_exposure.sliderReleased.connect(self.on_exposure_released)
        self.slider_exposure.installEventFilter(self)
        self.slider_exposure.setStyleSheet("""
            QSlider::groove:horizontal { height: 8px; background: #4a4a4a; border-radius: 4px; }
            QSlider::handle:horizontal { background: #242424; border: none; width: 16px; height: 16px; margin: -4px 0; border-radius: 8px; }
        """)
        exp_row.addWidget(self.slider_exposure, 1)

        self.btn_exp_plus = QPushButton("▶")
        self.btn_exp_plus.setFixedSize(24, 24)
        self.btn_exp_plus.setStyleSheet("QPushButton { background-color: transparent; color: #aaa; border: none; font-size: 12px; padding: 0; } QPushButton:hover { color: #ccc; }")
        self.btn_exp_plus.clicked.connect(lambda: self.adjust_exposure(0.1))
        exp_row.addWidget(self.btn_exp_plus)
        right_layout.addLayout(exp_row)
        right_layout.addSpacing(15)

        # White Balance
        wb_value = QLabel("5600K")
        wb_value.setStyleSheet("color: #626262; font-size: 14px;")
        wb_value.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(wb_value)
        self.label_wb = wb_value
        right_layout.addSpacing(5)

        wb_row = QHBoxLayout()
        wb_row.setSpacing(8)

        self.btn_wb_minus = QPushButton("◀")
        self.btn_wb_minus.setFixedSize(24, 24)
        self.btn_wb_minus.setStyleSheet("QPushButton { background-color: transparent; color: #7aa8d9; border: none; font-size: 12px; padding: 0; } QPushButton:hover { color: #9ac4e8; }")
        self.btn_wb_minus.clicked.connect(lambda: self.adjust_wb(-50))
        wb_row.addWidget(self.btn_wb_minus)

        self.slider_wb = QSlider(Qt.Horizontal)
        self.slider_wb.setMinimum(-2000)
        self.slider_wb.setMaximum(2000)
        self.slider_wb.setValue(0)
        self.slider_wb.valueChanged.connect(self.on_wb_slider_moved)
        self.slider_wb.sliderReleased.connect(self.on_wb_released)
        self.slider_wb.installEventFilter(self)
        self.slider_wb.setStyleSheet("""
            QSlider::groove:horizontal { height: 8px; background: #4a4a4a; border-radius: 4px; }
            QSlider::handle:horizontal { background: #242424; border: none; width: 16px; height: 16px; margin: -4px 0; border-radius: 8px; }
        """)
        wb_row.addWidget(self.slider_wb, 1)

        self.btn_wb_plus = QPushButton("▶")
        self.btn_wb_plus.setFixedSize(24, 24)
        self.btn_wb_plus.setStyleSheet("QPushButton { background-color: transparent; color: #e8b896; border: none; font-size: 12px; padding: 0; } QPushButton:hover { color: #f5c9a8; }")
        self.btn_wb_plus.clicked.connect(lambda: self.adjust_wb(50))
        wb_row.addWidget(self.btn_wb_plus)
        right_layout.addLayout(wb_row)
        right_layout.addSpacing(10)

        # Tint label + Auto tint checkbox in one row
        tint_header = QHBoxLayout()
        tint_header.setContentsMargins(0, 0, 0, 0)
        tint_header.setSpacing(0)

        self.chk_wb_link = QCheckBox("Auto tint")
        self.chk_wb_link.setChecked(False)
        self.chk_wb_link.setStyleSheet("""
            QCheckBox { color: #505050; font-size: 11px; spacing: 4px; }
            QCheckBox::indicator { width: 11px; height: 11px; border-radius: 2px;
                border: 1px solid #555; background: #3d3d3d; }
            QCheckBox::indicator:checked { background: #FF8A35; border-color: #FF8A35; }
            QCheckBox::indicator:hover { border-color: #888; }
        """)
        self.chk_wb_link.setToolTip(
            "Auto tint: WB moves tint proportionally.\n"
            "Manual tint nudges are preserved on top."
        )
        self.chk_wb_link.toggled.connect(self._on_wb_link_toggled)

        # Invisible spacer on the left matching checkbox width to keep label centered
        left_spacer = QWidget()
        left_spacer.setFixedWidth(self.chk_wb_link.sizeHint().width())
        tint_header.addWidget(left_spacer)

        tint_value = QLabel("+0.0")
        tint_value.setStyleSheet("color: #626262; font-size: 14px;")
        tint_value.setAlignment(Qt.AlignCenter)
        self.label_tint = tint_value
        tint_header.addWidget(tint_value, 1, Qt.AlignVCenter)

        tint_header.addWidget(self.chk_wb_link, 0, Qt.AlignVCenter)

        right_layout.addLayout(tint_header)
        right_layout.addSpacing(10)

        tint_row = QHBoxLayout()
        tint_row.setSpacing(8)

        self.btn_tint_minus = QPushButton("◀")
        self.btn_tint_minus.setFixedSize(24, 24)
        self.btn_tint_minus.setStyleSheet("QPushButton { background-color: transparent; color: #8fbf8f; border: none; font-size: 12px; padding: 0; } QPushButton:hover { color: #a5d4a5; }")
        self.btn_tint_minus.clicked.connect(lambda: self.adjust_tint(-1))
        tint_row.addWidget(self.btn_tint_minus)

        self.slider_tint = QSlider(Qt.Horizontal)
        self.slider_tint.setMinimum(-20)
        self.slider_tint.setMaximum(20)
        self.slider_tint.setValue(0)
        self.slider_tint.valueChanged.connect(self.on_tint_slider_moved)
        self.slider_tint.sliderReleased.connect(self.on_tint_released)
        self.slider_tint.installEventFilter(self)
        self.slider_tint.setStyleSheet("""
            QSlider::groove:horizontal { height: 8px; background: #4a4a4a; border-radius: 4px; }
            QSlider::handle:horizontal { background: #242424; border: none; width: 16px; height: 16px; margin: -4px 0; border-radius: 8px; }
        """)
        tint_row.addWidget(self.slider_tint, 1)

        self.btn_tint_plus = QPushButton("▶")
        self.btn_tint_plus.setFixedSize(24, 24)
        self.btn_tint_plus.setStyleSheet("QPushButton { background-color: transparent; color: #d49fc9; border: none; font-size: 12px; padding: 0; } QPushButton:hover { color: #e8b5de; }")
        self.btn_tint_plus.clicked.connect(lambda: self.adjust_tint(1))
        tint_row.addWidget(self.btn_tint_plus)
        right_layout.addLayout(tint_row)
        right_layout.addSpacing(30)

        # Reset button
        reset_layout = QHBoxLayout()
        reset_layout.setAlignment(Qt.AlignCenter)
        self.btn_reset_all = QPushButton("↻")
        self.btn_reset_all.setFixedSize(36, 36)
        self.btn_reset_all.setStyleSheet("""
            QPushButton { background-color: #3d3d3d; color: #626262; border: none; border-radius: 8px; font-size: 18px; }
            QPushButton:hover { background-color: #4a4a4a; }
        """)
        self.btn_reset_all.clicked.connect(self.reset_all_sliders)
        reset_layout.addWidget(self.btn_reset_all)
        right_layout.addLayout(reset_layout)
        right_layout.addSpacing(40)

        # Export Mode Toggle
        self.btn_export_mode = QPushButton("Export: Final JPEG")
        self.btn_export_mode.setFixedHeight(30)
        self.btn_export_mode.setStyleSheet("""
            QPushButton { background-color: #3d3d3d; color: #d0d0d0; border: 1px solid #555; border-radius: 15px; font-size: 11px; }
            QPushButton:hover { background-color: #4a4a4a; }
        """)
        self.btn_export_mode.setToolTip("Toggle between final film look (JPEG) and intermediate log (TIFF for Resolve)")
        self.btn_export_mode.clicked.connect(self.toggle_export_mode)
        right_layout.addWidget(self.btn_export_mode)
        right_layout.addSpacing(20)

        # Output path row
        output_row = QHBoxLayout()
        output_row.setSpacing(8)
        self.label_output = QLabel(self.output_dir)
        self.label_output.setStyleSheet("color: #626262; font-size: 12px;")
        self.label_output.setWordWrap(False)
        output_row.addWidget(self.label_output, 1)

        self.btn_select_output = QPushButton("⋯")
        self.btn_select_output.setFixedSize(28, 28)
        self.btn_select_output.setStyleSheet("""
            QPushButton { background-color: #3d3d3d; color: #626262; border: none; border-radius: 6px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #4a4a4a; }
        """)
        self.btn_select_output.clicked.connect(self.select_output_dir)
        output_row.addWidget(self.btn_select_output)
        right_layout.addLayout(output_row)
        right_layout.addSpacing(15)

        # Process button
        process_layout = QHBoxLayout()
        process_layout.setAlignment(Qt.AlignCenter)
        self.btn_process_all = QPushButton("Process 0 / 0")
        self.btn_process_all.setEnabled(False)
        self.btn_process_all.setFixedSize(180, 44)
        self.btn_process_all.setStyleSheet("""
            QPushButton { background-color: #FF8A35; color: #1a1a1a; border: none; border-radius: 22px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #ff9a4f; }
            QPushButton:pressed { background-color: #e67a25; }
            QPushButton:disabled { background-color: #555; color: #333; }
        """)
        self.btn_process_all.clicked.connect(self.process_all_images)
        process_layout.addWidget(self.btn_process_all)
        right_layout.addLayout(process_layout)

        right_wrapper_layout.addWidget(right_container)
        right_wrapper_layout.addStretch()

        # Progress bar
        self.progress_bar = QSlider(Qt.Horizontal)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setEnabled(False)
        self.progress_bar.setStyleSheet("""
            QSlider::groove:horizontal { height: 8px; background: #4a4a4a; border-radius: 4px; }
            QSlider::handle:horizontal { background: transparent; border: none; width: 0px; height: 0px; }
            QSlider::sub-page:horizontal { background: #FF8A35; border-radius: 4px; }
        """)
        right_layout.addSpacing(10)
        right_layout.addWidget(self.progress_bar)

        self.mode_label = QLabel("")
        self.mode_label.setStyleSheet("color: #FF8A35; font-size: 12px;")
        self.mode_label.setAlignment(Qt.AlignCenter)
        self.mode_label.setFixedHeight(20)
        right_layout.addWidget(self.mode_label)

        top_layout.addWidget(right_wrapper)
        top_layout.addStretch(1)
        main_vlayout.addWidget(top_section, 1)

        # === BOTTOM: Thumbnail Strip ===
        thumb_container = QWidget()
        thumb_layout = QHBoxLayout(thumb_container)
        thumb_layout.setContentsMargins(0, 0, 0, 0)

        self.thumbnail_strip = ThumbnailStrip()
        self.thumbnail_strip.thumbnail_clicked.connect(self.on_thumbnail_click)
        self.thumbnail_strip.thumbnail_right_clicked.connect(self.on_thumbnail_right_click)
        self.thumbnail_strip.thumbnail_paste_selected.connect(self.on_thumbnail_paste_selected)
        thumb_layout.addWidget(self.thumbnail_strip)

        self.fade_overlay = FadeOverlayWidget(self.thumbnail_strip)
        self.loader_overlay = LoaderOverlay(self.centralWidget())

        main_vlayout.addWidget(thumb_container)

        self.settings_clipboard = None

        self._build_menu_bar()

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
        self.mode_label.setStyleSheet("color: #FF8A35; font-size: 12px;")
        QTimer.singleShot(2000, lambda: self.mode_label.setText(""))

    def _menu_deselect_all_paste(self):
        self.thumbnail_strip.clear_paste_selection()
        self.mode_label.setText("Paste selection cleared")
        self.mode_label.setStyleSheet("color: #FF8A35; font-size: 12px;")
        QTimer.singleShot(1500, lambda: self.mode_label.setText(""))

    def export_as_jpeg(self):
        self.export_tiff_mode = False
        self.btn_export_mode.setText("Export: Final JPEG")
        self.btn_export_mode.setStyleSheet("""
            QPushButton { background-color: #3d3d3d; color: #d0d0d0; border: 1px solid #555; border-radius: 15px; font-size: 11px; }
        """)
        self.process_all_images()

    def export_as_tiff(self):
        self.export_tiff_mode = True
        self.btn_export_mode.setText("Export: Intermediate TIFF")
        self.btn_export_mode.setStyleSheet("""
            QPushButton { background-color: #4a4a4a; color: #FF8A35; border: 1px solid #FF8A35; border-radius: 15px; font-size: 11px; }
        """)
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

        self.add_thumbnail_worker = ThumbnailWorker(
            files_to_add,
            self.processor.lut_preview,
            self.processor.lut_full
        )
        self.add_thumbnail_worker.thumbnail_ready.connect(
            lambda i, t, mid, off=offset: self._add_thumbnail_to_ui(i + off, t, mid)
        )
        self.add_thumbnail_worker.finished.connect(self._on_add_thumbnails_finished)
        self.add_thumbnail_worker.setStackSize(32 * 1024 * 1024)
        self.add_thumbnail_worker.start()

        n = len(files_to_add)
        self.mode_label.setText(f"Adding {n} image{'s' if n != 1 else ''}...")
        self.mode_label.setStyleSheet("color: #FF8A35; font-size: 12px;")

    def _on_add_thumbnails_finished(self):
        print("✓ Add-images thumbnail generation complete!")
        self.mode_label.setText("")
        if hasattr(self, 'add_thumbnail_worker') and self.add_thumbnail_worker:
            self.add_thumbnail_worker.deleteLater()
            self.add_thumbnail_worker = None

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
            self.mode_label.setStyleSheet("color: #FF8A35; font-size: 12px;")
            QTimer.singleShot(2000, lambda: self.mode_label.setText(""))

    def on_thumbnail_paste_selected(self, index, is_selected):
        count = len(self.thumbnail_strip.get_paste_selected_indices())
        if count > 0:
            paste_key = "Cmd+V" if sys.platform == 'darwin' else "Ctrl+V"
            self.mode_label.setText(f"{count} selected for paste ({paste_key})")
            self.mode_label.setStyleSheet("color: #FF8A35; font-size: 12px;")
        else:
            self.mode_label.setText("")

    def update_process_button_text(self):
        selected = self.thumbnail_strip.get_process_selected_indices()
        if selected:
            self.btn_process_all.setText(f"Process {len(selected)} / {len(self.image_files)}")
        else:
            self.btn_process_all.setText(f"Process {len(self.image_files)} / {len(self.image_files)}")

    # ===================================================================
    # IMAGE LOADING & DISPLAY
    # ===================================================================

    def load_current_image(self):
        if not self.image_files:
            return

        file_path = str(self.image_files[self.current_index])

        if file_path in self.image_settings:
            settings = self.image_settings[file_path]
            self.processor.set_settings(settings)
            self.update_sliders_from_processor()
        else:
            self.processor.user_settings = {'exposure_ev': 0.0, 'wb_temp': 0, 'tint': 0.0}
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
        self.slider_tint.setValue(int(settings['tint'] * 2))

        self.label_exposure.setText(f"{settings['exposure_ev']:.1f} EV")
        temp_absolute = 5600 + int(settings['wb_temp'])
        self.label_wb.setText(f"{temp_absolute}K")
        self.label_tint.setText(f"{settings['tint']:+.1f}")

        self.slider_exposure.blockSignals(False)
        self.slider_wb.blockSignals(False)
        self.slider_tint.blockSignals(False)

        # Keep the manual tint offset coherent with the newly loaded settings
        if self.chk_wb_link.isChecked():
            self._tint_manual_offset = settings['tint'] - self._coupled_tint(settings['wb_temp'])

    # ===================================================================
    # SLIDER HANDLERS
    # ===================================================================

    def on_exposure_slider_moved(self, value):
        self.processor.preview_mode = 'fast'
        ev = value / 10.0
        self.label_exposure.setText(f"{ev:.1f} EV")
        img_array = self.processor.update_setting('exposure_ev', ev)
        self.display_image(img_array)
        self.update_current_thumbnail(img_array)
        self.save_current_settings()
        self.update_mode_label()

    def on_exposure_released(self):
        self.processor.preview_mode = 'hq'
        img_array = self.processor._render_fast()
        if img_array is not None:
            self.display_image(img_array)
            self.update_current_thumbnail(img_array)
            self.update_mode_label()

    def _on_wb_link_toggled(self, checked):
        if checked:
            self.slider_wb.setMinimum(-3000)
            self.slider_wb.setMaximum(3000)
        else:
            clamped = max(-2000, min(2000, self.slider_wb.value()))
            self.slider_wb.setMinimum(-2000)
            self.slider_wb.setMaximum(2000)
            if self.slider_wb.value() != clamped:
                self.slider_wb.setValue(clamped)

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
        self.slider_tint.setValue(int(round(new_tint * 2)))
        self.label_tint.setText(f"{new_tint:+.1f}")
        self.slider_tint.blockSignals(False)
        return new_tint

    def on_wb_slider_moved(self, value):
        self.processor.preview_mode = 'fast'
        temp_absolute = 5600 + value
        self.label_wb.setText(f"{temp_absolute}K")

        if self.chk_wb_link.isChecked():
            self._apply_wb_tint_link(value)
            self.processor.user_settings['wb_temp'] = value
            img_array = self.processor.render_preview()
        else:
            img_array = self.processor.update_setting('wb_temp', value)

        self.display_image(img_array)
        self.update_current_thumbnail(img_array)
        self.save_current_settings()
        self.update_mode_label()

    def on_wb_released(self):
        self.processor.preview_mode = 'hq'
        img_array = self.processor._render_fast()
        if img_array is not None:
            self.display_image(img_array)
            self.update_current_thumbnail(img_array)
            self.update_mode_label()

    def on_tint_slider_moved(self, value):
        self.processor.preview_mode = 'fast'
        tint = value / 2.0
        self.label_tint.setText(f"{tint:+.1f}")
        # Record how far the user has nudged tint away from the coupled position
        if self.chk_wb_link.isChecked():
            self._tint_manual_offset = tint - self._coupled_tint(self.slider_wb.value())
        img_array = self.processor.update_setting('tint', tint)
        self.display_image(img_array)
        self.update_current_thumbnail(img_array)
        self.save_current_settings()
        self.update_mode_label()

    def on_tint_released(self):
        self.processor.preview_mode = 'hq'
        img_array = self.processor._render_fast()
        if img_array is not None:
            self.display_image(img_array)
            self.update_current_thumbnail(img_array)
            self.update_mode_label()

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
        self.label_wb.setText("5600K")
        self.label_tint.setText("+0.0")
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
        self.mode_label.setText("")

    def save_current_settings(self):
        if self.image_files:
            file_path = str(self.image_files[self.current_index])
            self.image_settings[file_path] = self.processor.get_settings()

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
        self.mode_label.setText("Settings copied")
        self.mode_label.setStyleSheet("color: #FF8A35; font-size: 12px;")
        QTimer.singleShot(2000, lambda: self.mode_label.setText(""))

    def paste_settings(self):
        if not self.settings_clipboard:
            return

        paste_selected = self.thumbnail_strip.get_paste_selected_indices()

        if paste_selected:
            indices_to_apply = sorted(paste_selected)
            total = len(indices_to_apply)

            if total > 1:
                reply = QMessageBox.question(
                    self, "Apply Settings",
                    f"Apply copied settings to {total} selected images?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            success_count = 0
            self.mode_label.setText(f"Applying to {total} images...")
            self.mode_label.setStyleSheet("color: #FF8A35; font-size: 12px;")
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
                self.update_sliders_from_processor()
                self.display_image(img_array)
                self.update_thumbnail_for_settings(self.current_index, self.settings_clipboard)

            self.mode_label.setText(f"Updating thumbnails...")
            for idx in indices_to_apply:
                if idx != self.current_index:
                    self.update_thumbnail_for_settings(idx, self.settings_clipboard)

            self.thumbnail_strip.clear_paste_selection()
            self.mode_label.setText(f"Settings applied to {success_count} images")
            QTimer.singleShot(2000, lambda: self.mode_label.setText(""))

        else:
            img_array = self.processor.set_settings(self.settings_clipboard)
            self.update_sliders_from_processor()
            self.display_image(img_array)
            self.save_current_settings()
            self.update_mode_label()
            self.update_thumbnail_for_settings(self.current_index, self.settings_clipboard)
            self.mode_label.setText("Settings pasted")
            self.mode_label.setStyleSheet("color: #FF8A35; font-size: 12px;")
            QTimer.singleShot(2000, lambda: self.mode_label.setText(""))

    # ===================================================================
    # EXPORT
    # ===================================================================

    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir)
        if directory:
            self.output_dir = directory
            self.label_output.setText(directory)

    def toggle_export_mode(self):
        self.export_tiff_mode = not self.export_tiff_mode
        if self.export_tiff_mode:
            self.btn_export_mode.setText("Export: Intermediate TIFF")
            self.btn_export_mode.setStyleSheet("""
                QPushButton { background-color: #4a4a4a; color: #FF8A35; border: 1px solid #FF8A35; border-radius: 15px; font-size: 11px; }
            """)
        else:
            self.btn_export_mode.setText("Export: Final JPEG")
            self.btn_export_mode.setStyleSheet("""
                QPushButton { background-color: #3d3d3d; color: #d0d0d0; border: 1px solid #555; border-radius: 15px; font-size: 11px; }
            """)

    def process_all_images(self):
        if not self.image_files:
            return

        selected_indices = self.thumbnail_strip.get_process_selected_indices()
        if selected_indices:
            indices_to_process = sorted(selected_indices)
            mode_text = f"{len(indices_to_process)} selected"
        else:
            indices_to_process = list(range(len(self.image_files)))
            mode_text = f"all {len(self.image_files)}"

        reply = QMessageBox.question(
            self, "Process Images",
            f"Process {mode_text} images to:\n{self.output_dir}",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        # Estimate ~50 MB per TIFF, ~5 MB per JPEG, warn if disk space is low
        mb_per_image = 50 if self.export_tiff_mode else 5
        required_mb = len(indices_to_process) * mb_per_image
        try:
            free_mb = shutil.disk_usage(self.output_dir).free // (1024 * 1024)
            if free_mb < required_mb:
                reply = QMessageBox.warning(
                    self, "Low Disk Space",
                    f"Export may need ~{required_mb} MB but only {free_mb} MB free.\nContinue anyway?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
        except OSError:
            pass

        self.progress_bar.setMaximum(len(indices_to_process))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_process_all.setEnabled(False)
        self.mode_label.setText("Processing...")

        success_count = 0

        for i, idx in enumerate(indices_to_process):
            file_path = str(self.image_files[idx])

            try:
                self.progress_bar.setValue(i)
                self.mode_label.setText(f"Processing {i+1}/{len(indices_to_process)}...")
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

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                traceback.print_exc()

            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

        self.progress_bar.setVisible(False)
        self.btn_process_all.setEnabled(True)
        self.mode_label.setText("")
        self.update_process_button_text()
        self.load_current_image()

        QMessageBox.information(
            self, "Complete",
            f"Processed {success_count}/{len(indices_to_process)} images\nSaved to: {self.output_dir}"
        )

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
                self.mode_label.setStyleSheet("color: #FF8A35; font-size: 12px;")
                QTimer.singleShot(2000, lambda: self.mode_label.setText(""))
            event.accept()
        elif event.key() == Qt.Key_Escape:
            self.thumbnail_strip.clear_paste_selection()
            self.mode_label.setText("Paste selection cleared")
            self.mode_label.setStyleSheet("color: #FF8A35; font-size: 12px;")
            QTimer.singleShot(1500, lambda: self.mode_label.setText(""))
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
            # Split at the thumbnail strip top edge
            strip_top = self.thumbnail_strip.mapTo(cw, QPoint(0, 0)).y()
            strip_h = self.thumbnail_strip.height()
            self.drag_overlay.setGeometry(m, m, cw_w - 2 * m, max(40, strip_top - 2 * m))
            self.drag_overlay_add.setGeometry(m, strip_top + m, cw_w - 2 * m, max(30, strip_h - 2 * m))
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
