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
import logging
import sys
import os
import time
import threading
import numpy as np
import cv2

log = logging.getLogger(__name__)

from PySide6.QtWidgets import (
    QWidget, QLabel, QScrollArea, QFrame, QVBoxLayout, QHBoxLayout,
    QApplication, QSizePolicy, QToolTip,
)
from PySide6.QtCore import (
    Qt, QTimer, QSize, Signal, QThread, QEvent, QPropertyAnimation, QEasingCurve,
    QPoint, QPointF,
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QCursor, QLinearGradient,
    QMovie, QPainterPath, QColorSpace, QFont,
)

from core import resource_path
from core.processor import FlashbackProcessor
from core.config import _timing_print
from ui.theme import C, qcolor, register_theme_listener, ui_font


# =============================================================================
# THUMBNAIL WORKER
# =============================================================================

class ThumbnailWorker(QThread):
    """
    Background worker thread for generating thumbnails.
    Loads each RAW file in fast mode (LINEAR demosaic) and emits
    the display image and the linear ACEScg intermediate for each.
    """

    progress = Signal(int, int)            # current, total
    thumbnail_ready = Signal(int, object, object)  # index, thumb_array, intermediate
    finished = Signal()
    error = Signal(int, str)               # index, error_message

    def __init__(self, image_files, processor_lut):
        super().__init__()
        self.image_files = image_files
        self.processor_lut = processor_lut
        self._is_running = True

    def run(self):
        """Generate thumbnails in background."""
        processor = FlashbackProcessor(None)
        if self.processor_lut is not None:
            processor.lut = self.processor_lut

        total = len(self.image_files)
        total_start = time.time()

        for i in range(total):
            if not self._is_running:
                break

            file_path_str = str(self.image_files[i])

            try:
                img_display = processor.load_image(file_path_str)

                if img_display is not None and self._is_running:
                    intermediate = processor.intermediate_acescg.copy()
                    h, w = img_display.shape[:2]
                    scale = 70 / h
                    new_w = int(w * scale)
                    thumb_array = cv2.resize(img_display, (new_w, 70), interpolation=cv2.INTER_LINEAR)

                    self.thumbnail_ready.emit(i, thumb_array, intermediate)
                    self.progress.emit(i + 1, total)

            except Exception as e:
                log.error("  ✗ Failed thumbnail %d: %s", i, e)
                self.error.emit(i, str(e))

        total_time = time.time() - total_start
        _timing_print(f"\n✓ Thumbnails complete! {total} images in {total_time:.1f}s (avg: {total_time/total*1000:.0f}ms per image)\n")

        del processor
        self.finished.emit()


# =============================================================================
# RENDER WORKER
# =============================================================================

class RenderWorker(QThread):
    """
    Latest-wins background renderer for interactive slider scrubbing.

    Call request(downscale) from the main thread whenever a new render is
    needed. If a render is already in flight, the new parameters replace any
    queued request. The running render is allowed to finish and emit its
    result (so the user keeps seeing frames during a fast scrub); the next
    request then fires immediately, eventually converging on the latest state.

    Call invalidate() when the processor's state (intermediate, LUT, settings)
    changes out from under the worker — image switch, rotate, paste, vibe
    change. Any in-flight render's result is dropped instead of emitted.

    render_done(img_array, was_downscaled) is emitted on the main thread.
    The was_downscaled flag lets the caller decide whether to bump the
    thumbnail / persist settings.
    """

    render_done = Signal(object, bool)   # img_array, was_downscaled

    def __init__(self, processor):
        super().__init__()
        self._processor = processor
        self._lock = threading.Condition()
        self._pending = None          # None | bool (downscale flag)
        self._epoch = 0               # bumped on invalidate(); in-flight result is dropped if epoch changed
        self._running = True

    def request(self, downscale: bool):
        with self._lock:
            self._pending = downscale
            self._lock.notify()

    def invalidate(self):
        """Drop any pending request and discard the in-flight render's result.
        Call this when the processor's intermediate, LUT, or settings change
        out from under the worker (image switch, rotate, paste, vibe change)."""
        with self._lock:
            self._pending = None
            self._epoch += 1

    def stop(self):
        with self._lock:
            self._running = False
            self._pending = None
            self._lock.notify()

    def run(self):
        while True:
            with self._lock:
                while self._pending is None and self._running:
                    self._lock.wait()
                if not self._running:
                    return
                downscale = self._pending
                self._pending = None
                start_epoch = self._epoch

            img = self._processor._render_fast(downscale=downscale)

            with self._lock:
                if self._epoch != start_epoch:
                    # Invalidated mid-render — the result is computed against
                    # state that no longer matches the UI. Drop it.
                    continue

            if img is not None:
                self.render_done.emit(img, downscale)


# =============================================================================
# VIBE REFRESH WORKER
# =============================================================================

class VibeRefreshWorker(QThread):
    """
    Background worker that re-renders thumbnails after a vibe switch.

    Takes a snapshot of the image cache at construction time (main thread)
    so the worker never races against cache mutations during initial load.
    """

    thumbnail_ready = Signal(int, object)  # index, thumb_array
    finished = Signal()

    def __init__(self, image_files, cache_snapshot, image_settings,
                 current_index, lut, grain_tiles, default_settings, vibe):
        super().__init__()
        self.image_files = image_files
        self.cache_snapshot = cache_snapshot      # {path_str: acescg_array}
        self.image_settings = image_settings      # {path_str: settings_dict}
        self.current_index = current_index
        self.lut = lut
        self.grain_tiles = grain_tiles
        self.default_settings = default_settings
        self.vibe = vibe                          # snapshot of the active VibeConfig
        self._is_running = True

    def run(self):
        processor = FlashbackProcessor(vibe=self.vibe)
        processor.lut = self.lut
        processor.grain_tiles = self.grain_tiles

        for idx, path in enumerate(self.image_files):
            if not self._is_running:
                break
            if idx == self.current_index:
                continue
            file_path = str(path)
            cached = self.cache_snapshot.get(file_path)
            if cached is None:
                continue
            settings = self.image_settings.get(file_path, self.default_settings)
            try:
                processor.intermediate_acescg = cached.copy()
                processor.current_file = file_path
                processor.set_settings(settings)
                img_display = processor._render_fast(downscale=True)
                if img_display is not None:
                    h, w = img_display.shape[:2]
                    new_w = int(w * 70 / h)
                    thumb = cv2.resize(img_display, (new_w, 70), interpolation=cv2.INTER_LINEAR)
                    self.thumbnail_ready.emit(idx, thumb)
            except Exception as e:
                log.error("  ✗ Vibe refresh thumbnail %d: %s", idx, e)

        self.finished.emit()

    def stop(self):
        self._is_running = False


# =============================================================================
# THUMBNAIL WIDGET
# =============================================================================

class ThumbnailWidget(QFrame):
    """
    Individual thumbnail widget with TWO independent selection states + status:
    1. Process selection (right-click): accent border glow
    2. Paste selection (shift/cmd + left click): white overlay + marker
    Plus an index label ("01") and a green "processed" dot per mockup.
    """

    clicked = Signal(int)        # left click
    right_clicked = Signal(int)  # right click

    THUMBNAIL_HEIGHT = 70  # Fixed height, variable width based on aspect ratio

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.is_process_selected = False
        self.is_paste_selected = False
        self.is_current = False
        self.is_processed = False
        self.pixmap = None
        self.setFixedSize(int(self.THUMBNAIL_HEIGHT * 1.5), self.THUMBNAIL_HEIGHT)
        self.setFrameStyle(QFrame.NoFrame)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        register_theme_listener(self.update)

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

    def set_current(self, current: bool):
        self.is_current = current
        self.update()

    def set_processed(self, processed: bool):
        self.is_processed = processed
        self.update()

    def paintEvent(self, event):
        """Custom paint: thumbnail + index label + processed/paste markers + selection ring."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        radius = 4
        rounded_rect = QPainterPath()
        rounded_rect.addRoundedRect(self.rect(), radius, radius)
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
            painter.fillRect(self.rect(), qcolor("bg_strip"))

        if self.is_paste_selected:
            marker = qcolor("paste_marker")
            marker.setAlpha(50)
            painter.fillRect(self.rect(), marker)

        # Index label "01" in the top-left corner
        idx_col = qcolor("text_primary")
        idx_col.setAlpha(200)
        painter.setPen(idx_col)
        f = QFont("JetBrains Mono", 9)
        f.setWeight(QFont.Medium)
        painter.setFont(f)
        painter.drawText(6, 14, f"{self.index + 1:02d}")

        # Processed dot (bottom-right)
        if self.is_processed:
            painter.setBrush(qcolor("processed"))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.width() - 12, self.height() - 12, 6, 6)

        # Paste-selected marker (top-right)
        if self.is_paste_selected:
            painter.setBrush(qcolor("paste_marker"))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.width() - 12, 6, 6, 6)

        painter.setClipping(False)

        if self.is_process_selected:
            pen = QPen(qcolor("accent"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, radius, radius)
        elif self.is_current:
            pen = QPen(qcolor("text_primary"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, radius, radius)
        else:
            pen = QPen(qcolor("border_soft"))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, radius, radius)

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
        register_theme_listener(self.update)

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
        fade_width = 40

        strip_bg = qcolor("bg_strip")
        transparent = QColor(strip_bg)
        transparent.setAlpha(0)

        left_gradient = QLinearGradient(0, 0, fade_width, 0)
        left_gradient.setColorAt(0, strip_bg)
        left_gradient.setColorAt(1, transparent)
        painter.fillRect(0, 0, fade_width, height, left_gradient)

        right_gradient = QLinearGradient(width - fade_width, 0, width, 0)
        right_gradient.setColorAt(0, transparent)
        right_gradient.setColorAt(1, strip_bg)
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
        self.setFixedHeight(72)

        self.container = QWidget()
        self._apply_strip_bg()
        register_theme_listener(self._apply_strip_bg)
        self.layout = QHBoxLayout(self.container)
        self.layout.setSpacing(5)
        self.layout.setContentsMargins(12, 0, 12, 0)
        self.layout.setAlignment(Qt.AlignLeft)
        self.setWidget(self.container)

    def _apply_strip_bg(self):
        from ui.theme import C
        bg = C['bg_strip']
        self.setStyleSheet(f"QScrollArea {{ border: none; background-color: {bg}; }}")
        self.container.setStyleSheet(f"background: {bg};")

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

    def set_current_index(self, index: int):
        """Highlight the thumbnail for the currently-displayed image."""
        for i, t in enumerate(self.thumbnails):
            t.set_current(i == index)

    def set_processed(self, index: int, processed: bool):
        """Mark a thumbnail as exported (green dot)."""
        if 0 <= index < len(self.thumbnails):
            self.thumbnails[index].set_processed(processed)

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
        register_theme_listener(self.update)

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
        painter.fillRect(self.rect(), qcolor("bg_window"))

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
    ZOOM_FACTOR = 1.18  # multiplicative step per scroll tick

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
        self._apply_viewer_theme()
        register_theme_listener(self._apply_viewer_theme)

        self.image_label = RoundedLabel(radius=2)
        self.image_label.setMouseTracking(True)

        self.placeholder_label = QLabel("Drag & drop DNG files here\nor use the Folder icon")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self._apply_viewer_theme()
        self.setWidget(self.placeholder_label)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self._update_cursor()

    def _apply_viewer_theme(self):
        from ui.theme import C
        bg = C['bg_window']
        self.setStyleSheet(
            f"QScrollArea {{ border: none; background-color: {bg}; border-radius: 2px; }}"
            f"QScrollArea > QWidget {{ background-color: {bg}; }}"
            f"QScrollArea > QWidget > QWidget {{ background-color: {bg}; }}"
        )
        vp = self.viewport()
        if vp is not None:
            pal = vp.palette()
            pal.setColor(vp.backgroundRole(), qcolor("bg_window"))
            vp.setPalette(pal)
            vp.setAutoFillBackground(True)
        if hasattr(self, "placeholder_label"):
            self.placeholder_label.setStyleSheet(
                f"QLabel {{ color: {C['text_dim']}; font-size: 13px; background: transparent; }}"
            )

    def _create_zoom_cursor(self):
        # Rasterise assets/icons/zoom.svg into a plain 32×32 cursor pixmap.
        # Uses a single 1× pixmap so every platform treats it as a standard
        # 32-pixel cursor (some WMs clip oversized pixmaps from the top-left).
        from PySide6.QtSvg import QSvgRenderer
        from PySide6.QtCore import QByteArray, QRectF

        pm = QPixmap(32, 32)
        pm.fill(Qt.transparent)

        try:
            with open(resource_path("assets/icons/zoom.svg"), "rb") as f:
                data = f.read()
        except OSError:
            return QCursor(Qt.ArrowCursor)

        shadow_data = data.replace(b"stroke:black", b"stroke:#000")
        main_data = data.replace(b"stroke:black", b"stroke:#f0f0f0")

        target = 24            # SVG render box
        ox, oy = 4, 4          # centres 24px inside the 32px canvas

        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)

        p.setOpacity(0.55)
        QSvgRenderer(QByteArray(shadow_data)).render(
            p, QRectF(ox + 1, oy + 1, target, target)
        )
        p.setOpacity(1.0)
        QSvgRenderer(QByteArray(main_data)).render(
            p, QRectF(ox, oy, target, target)
        )
        p.end()

        # Glass centre in the 14-unit SVG viewBox ≈ (5.2, 5.0); after scaling
        # to 24px and the (4, 4) offset this lands at ≈ (13, 13).
        return QCursor(pm, 13, 13)

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

    def set_scrub_image(self, img_array):
        """Display a downscaled scrub frame without replacing the full-res reference pixmap.

        When zoomed in, a slider-drag render is downscaled for speed.  Calling
        set_image() would overwrite _original_pixmap with the small frame, making
        _zoom_level * new_width < expected size and causing a visible jump.  This
        method renders the scrub frame at the correct on-screen size instead.
        """
        if self._original_pixmap is None or self._fit_to_window:
            self.set_image(img_array)
            return
        img_8bit = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        h, w, c = img_8bit.shape
        q_image = QImage(img_8bit.data, w, h, c * w, QImage.Format_RGB888)
        scrub_pix = QPixmap.fromImage(q_image)
        display_w = int(self._original_pixmap.width() * self._zoom_level)
        display_h = int(self._original_pixmap.height() * self._zoom_level)
        scaled = scrub_pix.scaled(display_w, display_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.setFixedSize(scaled.size())

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

    def _set_zoom_at(self, zoom_level, cursor_vp=None):
        """Zoom to zoom_level, keeping the point under cursor_vp (viewport coords) fixed."""
        if self._original_pixmap is None:
            return

        old_zoom = self._get_fit_zoom() if self._fit_to_window else self._zoom_level
        zoom_ratio = zoom_level / old_zoom

        # Sample label-space position under cursor BEFORE layout changes.
        # image_label.mapFrom(viewport, p) already accounts for both scroll
        # offset and centering alignment, giving true label coordinates.
        label_pos = (
            self.image_label.mapFrom(self.viewport(), cursor_vp)
            if cursor_vp is not None else None
        )

        self._zoom_level = max(0.1, min(4.0, zoom_level))
        fit_zoom = self._get_fit_zoom()
        self._fit_to_window = abs(self._zoom_level - fit_zoom) < 0.05

        self._update_display()

        # Anchor: label_pos * zoom_ratio must end up at cursor_vp in the viewport.
        # new_scroll = label_pos * zoom_ratio - cursor_vp
        if label_pos is not None and zoom_ratio != 1.0:
            new_h = round(label_pos.x() * zoom_ratio - cursor_vp.x())
            new_v = round(label_pos.y() * zoom_ratio - cursor_vp.y())
            self.horizontalScrollBar().setValue(max(0, new_h))
            self.verticalScrollBar().setValue(max(0, new_v))

        self._update_cursor()

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
                self._set_zoom_at(1.25, event.pos())

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
        if delta == 0:
            return

        current = self._get_fit_zoom() if self._fit_to_window else self._zoom_level
        factor = self.ZOOM_FACTOR if delta > 0 else (1.0 / self.ZOOM_FACTOR)
        self._set_zoom_at(current * factor, event.position().toPoint())
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


# =============================================================================
# VIBE PICKER
# =============================================================================

class VibePicker(QWidget):
    """Film-character preset selector — styled dropdown matching the HTML design."""

    vibe_changed = Signal(str)

    VIBES = [
        ('disposable',           'Disposable',          "So bad it's good",           '1'),
        ('point_shoot',          'Point & Shoot',       '90s photoalbum vibes',        '2'),
        ('rangefinder',          'Rangefinder',         'Like-a M6',                   '3'),
        ('monochrome',           'Monochrome',          'makes everything art',         '4'),
        ('flashback_classic_v1', 'Flashback Classic V1','Recreation of Flashback Classic V1', '5'),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = 'disposable'
        self._setup_ui()
        register_theme_listener(self._apply_theme)

    def current_vibe(self):
        return self._current

    def set_vibe(self, vibe_id: str):
        if vibe_id == self._current or not any(v[0] == vibe_id for v in self.VIBES):
            return
        self._current = vibe_id
        self._update_display()
        self.vibe_changed.emit(vibe_id)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._selector = QFrame()
        self._selector.setCursor(Qt.CursorShape.PointingHandCursor)
        self._selector.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        sel_layout = QHBoxLayout(self._selector)
        sel_layout.setContentsMargins(10, 8, 10, 8)
        sel_layout.setSpacing(8)

        text_col = QWidget()
        text_col.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        text_v = QVBoxLayout(text_col)
        text_v.setContentsMargins(0, 0, 0, 0)
        text_v.setSpacing(2)

        self._name_lbl = QLabel()
        self._sub_lbl = QLabel()
        text_v.addWidget(self._name_lbl)
        text_v.addWidget(self._sub_lbl)

        self._chevron = QLabel("›")
        self._chevron.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        sel_layout.addWidget(text_col, 1)
        sel_layout.addWidget(self._chevron)

        layout.addWidget(self._selector)

        self._selector.mousePressEvent = lambda e: self._show_popup()
        self._update_display()
        self._apply_theme()

    def _update_display(self):
        vibe = next(v for v in self.VIBES if v[0] == self._current)
        self._name_lbl.setText(vibe[1])
        self._sub_lbl.setText(vibe[2])

    def _apply_theme(self):
        self._selector.setStyleSheet(
            f"QFrame {{ background: {C['bg_input']}; border: 1px solid {C['border_input']}; border-radius: 4px; }}"
            f"QFrame:hover {{ border-color: {C['border_active']}; }}"
        )
        self._name_lbl.setFont(ui_font(12, QFont.Weight.Medium))
        self._name_lbl.setStyleSheet(f"color: {C['text_primary']}; background: transparent; border: none;")
        self._sub_lbl.setFont(ui_font(10, QFont.Weight.Normal))
        self._sub_lbl.setStyleSheet(f"color: {C['text_dim']}; background: transparent; border: none;")
        self._chevron.setFont(ui_font(16, QFont.Weight.Normal))
        self._chevron.setStyleSheet(f"color: {C['text_dim']}; background: transparent; border: none;")

    def _show_popup(self):
        popup = QFrame(None, Qt.WindowType.Popup)
        popup.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        popup.setStyleSheet(
            f"QFrame#VibePopup {{ background: {C['bg_rail']}; border: 1px solid {C['border_input']};"
            f"  border-radius: 4px; }}"
        )
        popup.setObjectName("VibePopup")

        pop_layout = QVBoxLayout(popup)
        pop_layout.setContentsMargins(0, 4, 0, 4)
        pop_layout.setSpacing(0)

        for vibe_id, name, sub, shortcut in self.VIBES:
            row = QFrame()
            row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            row.setCursor(Qt.CursorShape.PointingHandCursor)
            is_selected = vibe_id == self._current
            row.setStyleSheet(
                f"QFrame {{ background: {C['bg_input_active'] if is_selected else 'transparent'}; border: none; }}"
            )

            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 8, 10, 8)
            row_layout.setSpacing(8)

            check_lbl = QLabel("✓" if is_selected else "")
            check_lbl.setFixedWidth(14)
            check_lbl.setFont(ui_font(11, QFont.Weight.Medium))
            check_lbl.setStyleSheet(f"color: {C['accent']}; background: transparent; border: none;")

            text_col = QWidget()
            text_col.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            text_v = QVBoxLayout(text_col)
            text_v.setContentsMargins(0, 0, 0, 0)
            text_v.setSpacing(2)

            name_lbl = QLabel(name)
            name_lbl.setFont(ui_font(12, QFont.Weight.Medium))
            name_lbl.setStyleSheet(f"color: {C['text_primary']}; background: transparent; border: none;")
            sub_lbl = QLabel(sub)
            sub_lbl.setFont(ui_font(10, QFont.Weight.Normal))
            sub_lbl.setStyleSheet(f"color: {C['text_dim']}; background: transparent; border: none;")

            text_v.addWidget(name_lbl)
            text_v.addWidget(sub_lbl)

            shortcut_lbl = QLabel(shortcut)
            shortcut_lbl.setFixedWidth(18)
            shortcut_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            shortcut_lbl.setFont(ui_font(10, QFont.Weight.Medium))
            shortcut_lbl.setStyleSheet(
                f"color: {C['text_dim']}; background: {C['bg_input']};"
                f" border: 1px solid {C['border_input']}; border-radius: 3px;"
                f" padding: 1px 0;"
            )

            row_layout.addWidget(check_lbl)
            row_layout.addWidget(text_col, 1)
            row_layout.addWidget(shortcut_lbl)

            def make_handler(vid, p, r):
                def on_press(event):
                    self._current = vid
                    self._update_display()
                    p.close()
                    self.vibe_changed.emit(vid)
                def on_enter(event):
                    if vid != self._current:
                        r.setStyleSheet(f"QFrame {{ background: {C['bg_input_hover']}; border: none; }}")
                def on_leave(event):
                    if vid != self._current:
                        r.setStyleSheet("QFrame { background: transparent; border: none; }")
                return on_press, on_enter, on_leave

            press, enter, leave = make_handler(vibe_id, popup, row)
            row.mousePressEvent = press
            row.enterEvent = enter
            row.leaveEvent = leave

            pop_layout.addWidget(row)

        global_pos = self._selector.mapToGlobal(QPoint(0, self._selector.height() + 4))
        popup.move(global_pos)
        popup.resize(self._selector.width(), len(self.VIBES) * 56 + 8)
        popup.show()
