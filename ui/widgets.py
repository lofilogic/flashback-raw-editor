"""
Custom Qt widgets for the Flashback editor.

  ThumbnailWorker   — background QThread for generating thumbnails
  ThumbnailWidget   — individual thumbnail with dual selection states
  FadeOverlayWidget — edge-fade gradient over the thumbnail strip
  LoaderOverlay     — animated full-window loading overlay
  ThumbnailStrip    — horizontal scrollable strip of ThumbnailWidgets
  RoundedLabel      — QLabel with anti-aliased rounded corners
  ZoomableImageWidget — pan/zoom image viewer
"""
import sys
import os
import time
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QWidget, QLabel, QScrollArea, QFrame, QVBoxLayout, QHBoxLayout,
    QApplication, QSizePolicy, QToolTip,
)
from PySide6.QtCore import (
    Qt, QTimer, QSize, Signal, QThread, QEvent, QPropertyAnimation, QEasingCurve,
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QCursor, QLinearGradient,
    QMovie, QPainterPath, QColorSpace,
)

from core import resource_path
from core.processor import FlashbackProcessor
from core.config import _timing_print


# =============================================================================
# THUMBNAIL WORKER
# =============================================================================

class ThumbnailWorker(QThread):
    """
    Background worker thread for generating thumbnails.
    Loads each RAW file in fast mode (LINEAR demosaic) and emits
    the display image and ACEScct intermediate for each.
    """

    progress = Signal(int, int)            # current, total
    thumbnail_ready = Signal(int, object, object)  # index, thumb_array, intermediate
    finished = Signal()
    error = Signal(int, str)               # index, error_message

    def __init__(self, image_files, processor_lut_preview, processor_lut_full):
        super().__init__()
        self.image_files = image_files
        self.processor_lut_preview = processor_lut_preview
        self.processor_lut_full = processor_lut_full
        self._is_running = True

    def run(self):
        """Generate thumbnails in background."""
        import gc

        processor = FlashbackProcessor(None)
        if self.processor_lut_preview is not None:
            processor.lut_preview = self.processor_lut_preview
            processor.lut_full = self.processor_lut_full

        total = len(self.image_files)
        total_start = time.time()

        for i in range(total):
            if not self._is_running:
                break

            file_path_str = str(self.image_files[i])

            try:
                img_display = processor.load_image(file_path_str, fast_mode=True)

                if img_display is not None and self._is_running:
                    intermediate = processor.intermediate_acescct.copy()
                    h, w = img_display.shape[:2]
                    scale = 70 / h
                    new_w = int(w * scale)
                    thumb_array = cv2.resize(img_display, (new_w, 70), interpolation=cv2.INTER_LINEAR)

                    self.thumbnail_ready.emit(i, thumb_array, intermediate)
                    self.progress.emit(i + 1, total)

            except Exception as e:
                print(f"  ✗ Failed thumbnail {i}: {e}")
                self.error.emit(i, str(e))

        total_time = time.time() - total_start
        _timing_print(f"\n✓ Thumbnails complete! {total} images in {total_time:.1f}s (avg: {total_time/total*1000:.0f}ms per image)\n")

        del processor
        self.finished.emit()


# =============================================================================
# THUMBNAIL WIDGET
# =============================================================================

class ThumbnailWidget(QFrame):
    """
    Individual thumbnail widget with TWO independent selection states:
    1. Process selection: orange border (right-click)
    2. Paste selection: white overlay (shift/cmd + left-click)
    """

    clicked = Signal(int)        # left click
    right_clicked = Signal(int)  # right click

    THUMBNAIL_HEIGHT = 70  # Fixed height, variable width based on aspect ratio

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.is_process_selected = False
        self.is_paste_selected = False
        self.pixmap = None
        self.setFixedSize(int(self.THUMBNAIL_HEIGHT * 1.5), self.THUMBNAIL_HEIGHT)
        self.setFrameStyle(QFrame.NoFrame)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def enterEvent(self, event):
        if self.toolTip():
            self._tooltip_timer = QTimer(self)
            self._tooltip_timer.setSingleShot(True)
            self._tooltip_timer.timeout.connect(
                lambda: QToolTip.showText(QCursor.pos(), self.toolTip(), self)
            )
            self._tooltip_timer.start(700)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if hasattr(self, '_tooltip_timer'):
            self._tooltip_timer.stop()
        QToolTip.hideText()
        super().leaveEvent(event)

    def set_pixmap(self, pixmap):
        """Set the thumbnail image and update size to match aspect ratio."""
        self.pixmap = pixmap
        if pixmap:
            aspect_ratio = pixmap.width() / pixmap.height()
            new_width = int(self.THUMBNAIL_HEIGHT * aspect_ratio)
            self.setFixedSize(new_width, self.THUMBNAIL_HEIGHT)
        self.update()

    def set_process_selected(self, selected):
        self.is_process_selected = selected
        self.update()

    def set_paste_selected(self, selected):
        self.is_paste_selected = selected
        self.update()

    def paintEvent(self, event):
        """Custom paint with dual selection highlights and rounded corners."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rounded_rect = QPainterPath()
        rounded_rect.addRoundedRect(self.rect(), 6, 6)
        painter.setClipPath(rounded_rect)

        if self.pixmap:
            scaled = self.pixmap.scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.fillRect(self.rect(), QColor(49, 49, 49))

        if self.is_paste_selected:
            painter.fillRect(self.rect(), QColor(255, 255, 255, 60))

        painter.setClipping(False)

        if self.is_process_selected:
            pen = QPen(QColor(255, 138, 53))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRoundedRect(1, 1, self.width() - 3, self.height() - 3, 4, 4)
        else:
            pen = QPen(QColor(60, 60, 60))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, 6, 6)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
            event.accept()
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit(self.index)
            event.accept()
        elif event.button() == Qt.MiddleButton:
            event.ignore()

    def wheelEvent(self, event):
        event.ignore()


# =============================================================================
# FADE OVERLAY
# =============================================================================

class FadeOverlayWidget(QWidget):
    """Overlay that draws fade gradients on left/right edges of the thumbnail strip."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        if parent is not None:
            parent.installEventFilter(self)

    def _update_geometry(self):
        parent = self.parent()
        if parent:
            self.setGeometry(parent.rect())

    def eventFilter(self, watched, event):
        if watched is self.parent() and event.type() in (QEvent.Resize, QEvent.Show):
            self._update_geometry()
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        self._update_geometry()
        super().showEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        fade_width = 60

        left_gradient = QLinearGradient(0, 0, fade_width, 0)
        left_gradient.setColorAt(0, QColor(49, 49, 49))
        left_gradient.setColorAt(1, QColor(49, 49, 49, 0))
        painter.fillRect(0, 0, fade_width, height, left_gradient)

        right_gradient = QLinearGradient(width - fade_width, 0, width, 0)
        right_gradient.setColorAt(0, QColor(49, 49, 49, 0))
        right_gradient.setColorAt(1, QColor(49, 49, 49))
        painter.fillRect(width - fade_width, 0, fade_width, height, right_gradient)


# =============================================================================
# LOADER OVERLAY
# =============================================================================

class LoaderOverlay(QWidget):
    """Full-window semi-opaque loader overlay with animated GIF and progress text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self.setStyleSheet("background-color: rgba(26,26,26,128);")

        container = QWidget(self)
        container.setAttribute(Qt.WA_TransparentForMouseEvents)
        container.setStyleSheet("background: transparent;")
        container_layout = QVBoxLayout(container)
        container_layout.setAlignment(Qt.AlignCenter)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(6)

        self.movie_label = QLabel()
        self.movie_label.setFixedSize(75, 75)
        self.movie_label.setAlignment(Qt.AlignCenter)
        self.movie_label.setStyleSheet("background: transparent;")
        container_layout.addWidget(self.movie_label, alignment=Qt.AlignCenter)

        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("background: transparent; color: #d0d0d0; font-family: Roboto, Arial, Helvetica; font-size: 13px;")
        container_layout.addWidget(self.progress_label, alignment=Qt.AlignCenter)

        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignCenter)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(container, alignment=Qt.AlignCenter)

        gif_path = resource_path("assets/icons/loading.gif")
        self.movie = None
        if os.path.exists(gif_path):
            try:
                self.movie = QMovie(gif_path)
                self.movie.setCacheMode(QMovie.CacheAll)
                self.movie.setScaledSize(QSize(75, 75))
                self.movie_label.setMovie(self.movie)
                self.movie.start()
            except Exception:
                self.movie = None

        if self.movie is None:
            pix = QPixmap(75, 75)
            pix.fill(Qt.transparent)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QColor(200, 200, 200))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(4, 4, 67, 67)
            painter.end()
            self.movie_label.setPixmap(pix)

        self.hide()
        self._fade_anim = None

        try:
            p = self.parent()
            if p is not None:
                p.installEventFilter(self)
        except Exception:
            pass

    def _update_geometry(self):
        parent = self.parent() or self.window()
        if parent:
            try:
                rect = parent.rect()
            except Exception:
                rect = parent.geometry()
            self.setGeometry(rect)

    def eventFilter(self, watched, event):
        if watched is self.parent():
            if event.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
                self._update_geometry()
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        self._update_geometry()
        self.raise_()
        super().showEvent(event)

    def fade_in(self, duration_ms: int = 300):
        self._update_geometry()
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()

        if self._fade_anim is not None:
            self._fade_anim.stop()

        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(duration_ms)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.InOutQuad)
        self._fade_anim.start()

    def fade_out(self, duration_ms: int = 200):
        if not self.isVisible():
            return

        if self._fade_anim is not None:
            self._fade_anim.stop()

        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(duration_ms)
        self._fade_anim.setStartValue(self.windowOpacity() if self.windowOpacity() > 0 else 1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.InOutQuad)
        self._fade_anim.finished.connect(self.hide)
        self._fade_anim.start()

    def update_progress(self, current: int, total: int):
        self.progress_label.setText(f"Developing {current} / {total}")
        if not self.isVisible():
            self.fade_in(300)
        self.raise_()

    def clear_and_hide(self):
        self.progress_label.setText("")
        self.fade_out(200)


# =============================================================================
# THUMBNAIL STRIP
# =============================================================================

class ThumbnailStrip(QScrollArea):
    """
    Horizontal scrollable thumbnail strip with TWO independent selection systems:

    1. PROCESS SELECTION (orange border):
       - Right-click: Toggle selection for processing
       - None selected = process all images

    2. PASTE SELECTION (white overlay):
       - Shift+Left click: Toggle single / range select
       - Cmd+Left click: Toggle individual
       - Cmd+A: Select all
       - Used by: Paste settings (Cmd+V)

    3. PLAIN LEFT CLICK:
       - Just loads the image in preview
       - Does NOT affect either selection
    """

    thumbnail_clicked = Signal(int)              # plain left click - load image
    thumbnail_right_clicked = Signal(int)        # right click - toggle process selection
    thumbnail_paste_selected = Signal(int, bool) # (index, is_selected) feedback

    def __init__(self, parent=None):
        super().__init__(parent)

        self.thumbnails = []
        self.process_selected_indices = set()
        self.paste_selected_indices = set()
        self._last_paste_click_index = 0
        self._middle_dragging = False
        self._last_mouse_x = 0

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(90)
        self.setStyleSheet("QScrollArea { border: none; background-color: #313131; }")

        self.container = QWidget()
        self.layout = QHBoxLayout(self.container)
        self.layout.setSpacing(8)
        self.layout.setContentsMargins(60, 0, 60, 0)
        self.layout.setAlignment(Qt.AlignLeft)
        self.setWidget(self.container)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._middle_dragging = True
            self._last_mouse_x = event.pos().x()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._middle_dragging:
            delta = self._last_mouse_x - event.pos().x()
            self._last_mouse_x = event.pos().x()
            hbar = self.horizontalScrollBar()
            hbar.setValue(hbar.value() + delta)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._middle_dragging = False
            self.unsetCursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        hbar = self.horizontalScrollBar()
        if delta > 0:
            hbar.setValue(hbar.value() - 100)
        else:
            hbar.setValue(hbar.value() + 100)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            event.ignore()
        else:
            super().keyPressEvent(event)

    def clear(self):
        for thumb in self.thumbnails:
            thumb.deleteLater()
        self.thumbnails.clear()
        self.process_selected_indices.clear()
        self.paste_selected_indices.clear()

    def add_thumbnail(self, pixmap, index, filename=None):
        thumb = ThumbnailWidget(index)

        if filename:
            thumb.setToolTip(str(filename))

        if isinstance(pixmap, np.ndarray):
            pixmap = self._array_to_pixmap(pixmap)

        thumb.set_pixmap(pixmap)
        thumb.clicked.connect(self._on_left_click)
        thumb.right_clicked.connect(self._on_right_click)

        self.thumbnails.append(thumb)
        self.layout.addWidget(thumb)

    def update_thumbnail(self, index, pixmap):
        if 0 <= index < len(self.thumbnails):
            if isinstance(pixmap, np.ndarray):
                pixmap = self._array_to_pixmap(pixmap)
            self.thumbnails[index].set_pixmap(pixmap)

    # ===================================================================
    # PROCESS SELECTION
    # ===================================================================

    def set_process_selected(self, index, selected):
        if 0 <= index < len(self.thumbnails):
            self.thumbnails[index].set_process_selected(selected)
            if selected:
                self.process_selected_indices.add(index)
            else:
                self.process_selected_indices.discard(index)

    def toggle_process_selection(self, index):
        if 0 <= index < len(self.thumbnails):
            is_selected = index in self.process_selected_indices
            self.set_process_selected(index, not is_selected)
            return not is_selected
        return False

    def get_process_selected_indices(self):
        return self.process_selected_indices.copy()

    def clear_process_selection(self):
        for i in range(len(self.thumbnails)):
            self.thumbnails[i].set_process_selected(False)
        self.process_selected_indices.clear()

    # ===================================================================
    # PASTE SELECTION
    # ===================================================================

    def set_paste_selected(self, index, selected):
        if 0 <= index < len(self.thumbnails):
            self.thumbnails[index].set_paste_selected(selected)
            if selected:
                self.paste_selected_indices.add(index)
            else:
                self.paste_selected_indices.discard(index)

    def toggle_paste_selection(self, index):
        if 0 <= index < len(self.thumbnails):
            is_selected = index in self.paste_selected_indices
            new_state = not is_selected
            self.set_paste_selected(index, new_state)
            self.thumbnail_paste_selected.emit(index, new_state)
            return new_state
        return False

    def get_paste_selected_indices(self):
        return self.paste_selected_indices.copy()

    def clear_paste_selection(self):
        for i in range(len(self.thumbnails)):
            self.thumbnails[i].set_paste_selected(False)
        self.paste_selected_indices.clear()

    def select_paste_range(self, start_idx, end_idx):
        min_idx = min(start_idx, end_idx)
        max_idx = max(start_idx, end_idx)
        for i in range(min_idx, max_idx + 1):
            if 0 <= i < len(self.thumbnails):
                self.set_paste_selected(i, True)
        self._last_paste_click_index = end_idx

    def select_all_for_paste(self):
        for i in range(len(self.thumbnails)):
            self.set_paste_selected(i, True)

    # ===================================================================
    # CLICK HANDLERS
    # ===================================================================

    def _on_left_click(self, index):
        modifiers = QApplication.keyboardModifiers()

        if modifiers & Qt.ShiftModifier:
            if len(self.paste_selected_indices) == 0:
                self.toggle_paste_selection(index)
            else:
                self.select_paste_range(self._last_paste_click_index, index)
            self._last_paste_click_index = index
        elif modifiers & Qt.ControlModifier:
            self.toggle_paste_selection(index)
            self._last_paste_click_index = index
        else:
            self._last_paste_click_index = index
            self.thumbnail_clicked.emit(index)

    def _on_right_click(self, index):
        self.thumbnail_right_clicked.emit(index)

    def _array_to_pixmap(self, img_array):
        img_8bit = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        height, width, channels = img_8bit.shape
        bytes_per_line = channels * width
        q_image = QImage(img_8bit.data, width, height, bytes_per_line, QImage.Format_RGB888)
        q_image.setColorSpace(QColorSpace(QColorSpace.SRgb))
        return QPixmap.fromImage(q_image)


# =============================================================================
# ROUNDED LABEL
# =============================================================================

class RoundedLabel(QLabel):
    """QLabel with anti-aliased rounded corners."""

    def __init__(self, parent=None, radius=12):
        super().__init__(parent)
        self._radius = radius
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def pixmap(self):
        return self._pixmap

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), self._radius, self._radius)
        painter.setClipPath(path)
        painter.fillRect(self.rect(), QColor(49, 49, 49))

        if self._pixmap:
            x = (self.width() - self._pixmap.width()) // 2
            y = (self.height() - self._pixmap.height()) // 2
            painter.drawPixmap(x, y, self._pixmap)

        painter.end()


# =============================================================================
# ZOOMABLE IMAGE WIDGET
# =============================================================================

class ZoomableImageWidget(QScrollArea):
    """
    Custom zoomable image viewer with pan support.

    - Left click: zoom to 125% (or pan if already zoomed)
    - Scroll: zoom in/out through fixed steps
    - Mouse drag: pan when zoomed in
    - Double click: fit to window
    """

    ZOOM_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

    def __init__(self, parent=None):
        super().__init__(parent)

        self._pixmap = None
        self._original_pixmap = None
        self._zoom_level = 0.75
        self._fit_to_window = True
        self._panning = False
        self._last_mouse_pos = None

        self._zoom_cursor = self._create_zoom_cursor()

        self.setWidgetResizable(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QScrollArea { border: none; background-color: #313131; }")

        self.image_label = RoundedLabel(radius=12)
        self.image_label.setMouseTracking(True)

        self.placeholder_label = QLabel("Drag & drop DNG files here\nor use the Folder Icon")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("""
            QLabel {
                color: #626262;
                font-size: 14px;
                background: transparent;
            }
        """)
        self.setWidget(self.placeholder_label)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self._update_cursor()

    def _create_zoom_cursor(self):
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(200, 200, 200))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(4, 4, 18, 18)
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawLine(18, 18, 28, 28)
        painter.end()
        return QCursor(pixmap, 8, 8)

    def set_image(self, img_array):
        if img_array is None:
            return
        img_8bit = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        height, width, channels = img_8bit.shape
        bytes_per_line = channels * width
        q_image = QImage(img_8bit.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self._original_pixmap = QPixmap.fromImage(q_image)
        self._pixmap = self._original_pixmap

        if self.widget() == self.placeholder_label:
            self.setWidget(self.image_label)

        self._update_display()

    def set_pixmap(self, pixmap):
        self._original_pixmap = pixmap
        self._pixmap = pixmap
        self._update_display()

    def clear(self):
        self._pixmap = None
        self._original_pixmap = None
        self.image_label.clear()
        self.image_label.setFixedSize(1, 1)
        self.setWidget(self.placeholder_label)

    def _update_display(self):
        if self._original_pixmap is None:
            if self.widget() != self.placeholder_label:
                self.setWidget(self.placeholder_label)
            return

        if self._fit_to_window:
            viewport_size = self.viewport().size()
            scaled = self._original_pixmap.scaled(
                viewport_size - QSize(20, 20),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            self.image_label.setFixedSize(scaled.size())
            self._zoom_level = scaled.width() / self._original_pixmap.width()
        else:
            new_width = int(self._original_pixmap.width() * self._zoom_level)
            new_height = int(self._original_pixmap.height() * self._zoom_level)
            scaled = self._original_pixmap.scaled(
                new_width, new_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            self.image_label.setFixedSize(scaled.size())

    def _get_fit_zoom(self):
        if self._original_pixmap is None:
            return 1.0
        viewport_size = self.viewport().size()
        scale_w = (viewport_size.width() - 20) / self._original_pixmap.width()
        scale_h = (viewport_size.height() - 20) / self._original_pixmap.height()
        return min(scale_w, scale_h)

    def _set_zoom_at(self, zoom_level, pos=None):
        if self._original_pixmap is None:
            return

        old_zoom = self._zoom_level if not self._fit_to_window else self._get_fit_zoom()
        zoom_ratio = zoom_level / old_zoom

        self._zoom_level = max(0.25, min(4.0, zoom_level))

        fit_zoom = self._get_fit_zoom()
        if abs(self._zoom_level - fit_zoom) < 0.05:
            self._fit_to_window = True
        else:
            self._fit_to_window = False

        self._update_display()

        if pos is not None and zoom_ratio != 1.0:
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            old_h = hbar.value() + pos.x()
            old_v = vbar.value() + pos.y()
            new_h = int(old_h * zoom_ratio) - pos.x()
            new_v = int(old_v * zoom_ratio) - pos.y()
            hbar.setValue(new_h)
            vbar.setValue(new_v)

        self._update_cursor()

    def _step_zoom_at(self, direction, pos=None):
        if self._fit_to_window:
            current_zoom = self._get_fit_zoom()
            self._fit_to_window = False
        else:
            current_zoom = self._zoom_level

        if direction > 0:
            for level in self.ZOOM_LEVELS:
                if level > current_zoom * 1.1:
                    self._set_zoom_at(level, pos)
                    return
            self._set_zoom_at(4.0, pos)
        else:
            for level in reversed(self.ZOOM_LEVELS):
                if level < current_zoom * 0.9:
                    self._set_zoom_at(level, pos)
                    return
            self._set_zoom_at(self._get_fit_zoom(), pos)

    def _update_cursor(self):
        fit_zoom = self._get_fit_zoom()
        if self._zoom_level > fit_zoom * 1.3:
            self.image_label.setCursor(Qt.OpenHandCursor)
        else:
            self.image_label.setCursor(self._zoom_cursor)

    def mousePressEvent(self, event):
        if self._original_pixmap is None:
            return

        if event.button() == Qt.LeftButton:
            fit_zoom = self._get_fit_zoom()
            if self._zoom_level > fit_zoom * 1.3:
                self._panning = True
                self._last_mouse_pos = event.pos()
                self.image_label.setCursor(Qt.ClosedHandCursor)
            else:
                image_pos = self.image_label.mapFrom(self.viewport(), event.pos())
                self._set_zoom_at(1.25, image_pos)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._last_mouse_pos:
            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._panning:
            self._panning = False
            self._update_cursor()

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._fit_to_window = True
            self._update_display()
            self._update_cursor()

        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if self._original_pixmap is None:
            return

        delta = event.angleDelta().y()
        pos = self.image_label.mapFrom(self.viewport(), event.position().toPoint())

        if delta > 0:
            self._step_zoom_at(1, pos)
        else:
            self._step_zoom_at(-1, pos)

        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            event.ignore()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_to_window:
            self._update_display()
