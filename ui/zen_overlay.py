"""
Frameless fullscreen "zen" overlay — distraction-free preview with gesture-
based exposure / WB / tint adjustments.

Communicates with the main editor via signals (closed / navigated / rotated)
and direct calls on `self.main_window` (set as the parent at construction).
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QLabel, QPushButton


class FullscreenZenOverlay(QWidget):
    closed = Signal()
    navigated = Signal(int)
    rotated = Signal(int)
    remove_requested = Signal()

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

        # Persist the gesture-adjusted values for the current image. The
        # slider_*Released handlers do this in normal mode; zen drives the
        # sliders via setValue() and bypasses those signals entirely.
        self.main_window.save_current_settings()

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
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.remove_requested.emit()
            # If that emptied the project, close ourselves here — closing from
            # within zen's own keyPressEvent is more reliable than scheduling
            # hide() from the main-window handler.
            if not self.main_window.image_files:
                self.close_zen()
        elif event.key() in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5):
            vibes = ('disposable', 'point_shoot', 'rangefinder', 'monochrome', 'flashback_classic_v1')
            self.main_window.vibe_picker.set_vibe(vibes[event.key() - Qt.Key_1])

    def close_zen(self):
        self.hide()
        self.closed.emit()
