"""Minimal precision slider styled to match the Flashback handoff mockup.

- 2px track, 14px circular thumb
- Dual (±) mode: fill grows from the center + a center tick
- Click-to-jump on the track (Lightroom-style)
- Drag halo around the thumb while pressed
"""

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QSlider

from ui.theme import qcolor, register_theme_listener


class ScrubSlider(QSlider):
    TRACK_H = 2
    THUMB_R = 7          # thumb radius (diameter 14)
    CENTER_TICK_H = 8
    HALO_R = 11          # drag halo radius

    def __init__(self, parent=None, dual: bool = False):
        super().__init__(Qt.Horizontal, parent)
        self._dual = dual
        self._dragging = False
        self.setMinimumHeight(18)
        self.setFixedHeight(18)
        # Neutralise the native QSlider subcontrols so they can't bleed through
        # our custom paintEvent (left-edge groove artefact on some platforms).
        self.setStyleSheet(
            "QSlider { background: transparent; border: none; }"
            "QSlider::groove:horizontal { background: transparent; border: none; height: 0px; }"
            "QSlider::sub-page:horizontal, QSlider::add-page:horizontal {"
            "  background: transparent; border: none;"
            "}"
            "QSlider::handle:horizontal { background: transparent; border: none; width: 0px; }"
        )
        self.setFocusPolicy(Qt.NoFocus)
        register_theme_listener(self.update)

    def setDual(self, dual: bool):
        self._dual = dual
        self.update()

    # ── mouse: click-to-jump, drag to scrub ──────────────────────────────
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._dragging = True
            self._set_from_x(e.position().x())
            self.sliderPressed.emit()
            e.accept()
            self.update()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._set_from_x(e.position().x())
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.sliderReleased.emit()
            e.accept()
            self.update()
            return
        super().mouseReleaseEvent(e)

    def _set_from_x(self, x: float):
        w = self.width()
        if w <= 0:
            return
        ratio = max(0.0, min(1.0, x / w))
        vmin, vmax = self.minimum(), self.maximum()
        val = round(vmin + ratio * (vmax - vmin))
        if val != self.value():
            self.setValue(val)

    # ── painting ─────────────────────────────────────────────────────────
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cy = h / 2
        track_y = cy - self.TRACK_H / 2

        vmin, vmax = self.minimum(), self.maximum()
        span = vmax - vmin if vmax != vmin else 1
        pct = (self.value() - vmin) / span
        thumb_x = pct * w

        # track
        track_col = qcolor("border_input")
        p.setPen(Qt.NoPen)
        p.setBrush(track_col)
        p.drawRoundedRect(
            QRect(0, int(track_y), w, self.TRACK_H), 1, 1
        )

        # fill
        fill_col = qcolor("text_label")
        if self._dual:
            center_x = w / 2
            x0 = min(center_x, thumb_x)
            x1 = max(center_x, thumb_x)
            p.setBrush(fill_col)
            p.drawRoundedRect(
                QRect(int(x0), int(track_y), max(1, int(x1 - x0)), self.TRACK_H),
                1, 1,
            )
            # center tick
            p.setBrush(qcolor("border_active"))
            p.drawRect(
                QRect(
                    int(center_x - 0.5),
                    int(cy - self.CENTER_TICK_H / 2),
                    1,
                    self.CENTER_TICK_H,
                )
            )
        else:
            p.setBrush(fill_col)
            p.drawRoundedRect(
                QRect(0, int(track_y), int(thumb_x), self.TRACK_H), 1, 1
            )

        # halo (on drag)
        if self._dragging:
            p.setBrush(qcolor("accent_soft"))
            p.drawEllipse(
                int(thumb_x - self.HALO_R),
                int(cy - self.HALO_R),
                self.HALO_R * 2,
                self.HALO_R * 2,
            )

        # thumb w/ soft shadow
        shadow = QColor(0, 0, 0, 90)
        p.setBrush(shadow)
        p.drawEllipse(
            int(thumb_x - self.THUMB_R),
            int(cy - self.THUMB_R + 1),
            self.THUMB_R * 2,
            self.THUMB_R * 2,
        )
        p.setBrush(qcolor("text_primary"))
        p.drawEllipse(
            int(thumb_x - self.THUMB_R),
            int(cy - self.THUMB_R),
            self.THUMB_R * 2,
            self.THUMB_R * 2,
        )
        p.end()
