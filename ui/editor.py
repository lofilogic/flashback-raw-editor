"""
Main application window.

FlashbackEditor — QMainWindow: file loading, sliders, thumbnail strip,
                  export, keyboard shortcuts, drag & drop.

The fullscreen Zen overlay lives in ui.zen_overlay.
"""
import logging
import sys
import os
import shutil
import time
import traceback
import platform
import re
from collections import OrderedDict
from pathlib import Path

# core must be imported before colour to apply the NumPy 2.0 compatibility shim
import core  # noqa: F401

log = logging.getLogger(__name__)

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
from core.gpu import gpu
from core.processor import FlashbackProcessor, export_image
from core.config import (
    _timing_print, VIBE_PRESETS, VIBE_EXPORT_SUFFIX,
    VibeConfig, ImageAdjustments, vibe_config_for, effective_lut_ref,
)
from core.export_naming import export_basename
from core.v1_negative import is_v1_negative
from core.camera_import import date_folder_name
from core import vibe_state

from .widgets import (
    ThumbnailWorker, ThumbnailWidget, ThumbnailStrip,
    FadeOverlayWidget, LoaderOverlay, ZoomableImageWidget, VibePicker,
    RenderWorker, VibeRefreshWorker,
)
from .debug_panel import DebugPanel
from .scrub_slider import ScrubSlider
from .zen_overlay import FullscreenZenOverlay
from . import theme
from .theme import (
    C, UI_FONT, MONO_FONT,
    icon_btn_qss, section_title_qss, section_reset_link_qss,
    process_btn_qss, format_pill_qss, svg_icon,
)


# =============================================================================
# MAIN EDITOR WINDOW
# =============================================================================

def _value_nbytes(value):
    """Bytes held by a cache value: a numpy array, or a tuple/list that
    contains arrays (preview_cache stores ``(key, img_array)``)."""
    nb = getattr(value, 'nbytes', None)
    if nb is not None:
        return nb
    if isinstance(value, (tuple, list)):
        return sum(_value_nbytes(v) for v in value)
    return 0


class _CacheBudget:
    """Shared, dynamically-sized memory budget for the array caches.

    All caches that register share one pool, so the cap is the total resident
    cache memory — not per-cache. The limit is recomputed live from system RAM
    (via psutil) as ``min(fraction * total, available - reserve)`` so the caches
    automatically back off when other software consumes memory, and grow again
    when it's freed. Falls back to a fixed limit if psutil is unavailable.
    """

    def __init__(self, fraction=0.5, reserve_bytes=2 * 1024 ** 3,
                 floor_bytes=512 * 1024 ** 2, fallback_bytes=2 * 1024 ** 3):
        self.fraction = fraction
        self.reserve = reserve_bytes
        self.floor = floor_bytes
        self.fallback = fallback_bytes
        self.used = 0
        self._caches = []

    def register(self, cache):
        self._caches.append(cache)

    def limit(self):
        try:
            import psutil
            vm = psutil.virtual_memory()
            return max(self.floor, min(int(self.fraction * vm.total),
                                       int(vm.available - self.reserve)))
        except Exception:
            return max(self.floor, self.fallback)

    def enforce(self, protect=()):
        """Evict least-recently-used entries across all registered caches until
        the shared total is within the current limit. ``protect`` names keys that
        must never be evicted (the active image)."""
        limit = self.limit()
        # Guard bounds the loop against pathological states; normal exit is the
        # budget condition or running out of evictable entries.
        for _ in range(1_000_000):
            if self.used <= limit:
                return
            victim = None
            for cache in self._caches:
                for key in cache:                # OrderedDict: oldest first
                    if key not in protect:
                        victim = (cache, key)
                        break
                if victim:
                    break
            if victim is None:
                return                            # everything left is protected
            cache, key = victim
            del cache[key]


class _ByteBudgetLRU(OrderedDict):
    """LRU cache of numpy arrays sharing a :class:`_CacheBudget`.

    get/set bump recency; inserts trigger a shared-budget enforcement that
    evicts the globally least-recently-used entries across all caches sharing
    the budget. Evicted intermediates are re-derived from disk on next visit,
    and the currently displayed image survives because the processor holds its
    own reference to that array (and it is passed as ``protect`` on insert).
    """

    def __init__(self, budget: _CacheBudget):
        super().__init__()
        self._budget = budget
        budget.register(self)

    def __getitem__(self, key):
        self.move_to_end(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        super().__setitem__(key, value)
        self._budget.used += _value_nbytes(value)
        self._budget.enforce(protect=(key,))

    def __delitem__(self, key):
        self._budget.used -= _value_nbytes(super().__getitem__(key))
        super().__delitem__(key)

    def pop(self, key, *default):
        # Delete through our own __delitem__ (which adjusts the budget) using
        # the raw OrderedDict accessor — NOT super().pop / self[key], both of
        # which re-enter the overridden __getitem__ and move_to_end, turning an
        # absent-key lookup into a KeyError that defeats the `default` arg.
        if key in self:
            value = OrderedDict.__getitem__(self, key)
            del self[key]
            return value
        if default:
            return default[0]
        raise KeyError(key)

    def clear(self):
        for v in self.values():
            self._budget.used -= _value_nbytes(v)
        super().clear()

    def prune_to(self, valid_keys):
        """Drop entries whose key is not in ``valid_keys`` (e.g. images no
        longer in the open project)."""
        for key in [k for k in self if k not in valid_keys]:
            del self[key]


class FlashbackEditor(QMainWindow):
    """Main application window for LoFi Logic image editing."""

    # Permissive: every raw extension libraw can plausibly decode. We let the
    # actual decode be the gate — an unsupported/corrupt file raises in
    # rawpy.imread, is caught at load, and surfaces as a clean miss (not a crash).
    # Exposure handling: DNGs read embedded BaselineExposure (Tier 1); measured
    # makes hit Tier 2; everything else lands on the Tier-3 default (+0.2).
    SUPPORTED_EXTENSIONS = (
        '.dng',                                  # Adobe / Leica / Ricoh / Pixel / iPhone
        '.cr2', '.cr3', '.crw',                  # Canon
        '.nef', '.nrw',                          # Nikon
        '.arw', '.srf', '.sr2',                  # Sony
        '.raf',                                  # Fujifilm
        '.orf',                                  # Olympus / OM
        '.rw2', '.raw',                          # Panasonic
        '.rwl',                                  # Leica
        '.pef', '.ptx',                          # Pentax
        '.3fr', '.fff',                          # Hasselblad / Imacon
        '.iiq', '.cap', '.eip',                  # Phase One
        '.mef',                                  # Mamiya
        '.mos',                                  # Leaf
        '.mrw',                                  # Minolta
        '.dcr', '.dcs', '.kdc', '.k25', '.drf',  # Kodak
        '.mdc',                                  # Minolta / Agfa
        '.erf',                                  # Epson
        '.srw',                                  # Samsung
        '.x3f',                                  # Sigma (Foveon)
        '.pxn',                                  # Logitech
        '.gpr',                                  # GoPro
        '.rwz',                                  # Rawzor
        '.bay',                                  # Casio
        '.ari',                                  # ARRI
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
        # All array caches share one dynamic, RAM-relative memory budget so the
        # combined resident cache never exceeds it; entries are LRU-evicted
        # across caches and re-derived from disk on next visit.
        self.cache_budget = _CacheBudget()
        self.image_cache = _ByteBudgetLRU(self.cache_budget)
        self.preview_cache = _ByteBudgetLRU(self.cache_budget)
        self.export_mode = 'jpeg'  # 'jpeg' | 'tiff' | 'dng'
        self.thumbnail_cache = _ByteBudgetLRU(self.cache_budget)
        self.thumbnail_settings = {}
        self._file_is_flashback: dict = {}  # path_str -> bool
        # Cumulative rotation in degrees (0/90/180/270) per image path. The
        # processor *consumes* its rotation field by burning it into the
        # intermediate, so we keep our own running tally that survives reloads
        # and gets persisted in project files.
        self.image_rotations: dict = {}
        # Path of the currently-open project file (.lofi), or None if the
        # current image set didn't come from a project. Save reuses this;
        # Save As always prompts.
        self.current_project_path = None  # type: ignore[assignment]

        self.app_settings = QSettings("LoFi Logic", "Editor")

        # Two folders, two responsibilities:
        #   camera_import_dir — where the camera-import worker archives rolls
        #                       (creates YYYY-MM-DD/_RAW/<name>.dng subfolders)
        #   output_dir        — where regular exports go (DNG / JPG)
        # Each has a *default* persisted in QSettings (set from the advanced
        # panel). The runtime value is initialized from that default on every
        # launch; mid-session changes to output_dir are intentionally not
        # written back, so a one-off export to elsewhere can't nest the next
        # camera import inside it.
        pictures_loc = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        base_dir = pictures_loc if pictures_loc else str(Path.home())
        fallback = os.path.join(base_dir, "LoFi_Logic")

        def _resolve(key: str) -> str:
            v = self.app_settings.value(key, fallback)
            return v if isinstance(v, str) and v else fallback

        self.camera_import_dir = _resolve("default_camera_import_dir")
        self.output_dir = _resolve("default_export_dir")
        os.makedirs(self.camera_import_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.pending_file_path = None

        # The active vibe — replaces the old global DebugConfig. Initialized
        # to factory disposable here; the real vibe is loaded in
        # _on_vibe_selected() once the picker exists.
        self.current_vibe = VibeConfig()
        self.current_vibe.dng_profile_name = self.app_settings.value(
            "dng_profile_name", "Flashback Standard"
        )

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
        self._vibe_refresh_worker = None
        self._thumbnails_dirty = set()
        self._lut_cache: dict = {}
        # The LUT ref currently uploaded to the GPU / set on the processor —
        # may be a transient V1 override of current_vibe.lut_ref (see
        # _apply_effective_lut), so it's tracked separately to avoid redundant
        # re-uploads when scrubbing between frames.
        self._active_lut_ref = None

        self._tint_manual_offset = 0.0  # user's manual tint correction on top of WB coupling

        self.pending_render = False

        # Run pre-1.5 → 1.5.0 vibe-state migration once. The report (if
        # any) is stashed for the post-window-shown notice; vibes loaded
        # here are not directly used (the editor reads via _vibe_for) but
        # calling migrate_and_load is what triggers the on-disk rewrite.
        _, self._migration_report = vibe_state.migrate_and_load()

        # LUT is loaded by FlashbackProcessor from current_vibe.lut_ref
        # (factory:<id> or user:<path>). No path argument anymore — the
        # processor never reads filesystem paths from the editor.
        self.processor = FlashbackProcessor(
            vibe=self.current_vibe,
            adjustments=ImageAdjustments(),
        )

        self._render_worker = RenderWorker(self.processor)
        self._render_worker.render_done.connect(self._on_render_done)
        self._render_worker.start()
        self._render_needs_commit = False  # True after slider release

        self.init_ui()

        self.debug_panel = DebugPanel(self.processor, self)
        self._on_vibe_selected(self.vibe_picker._current)
        self.debug_panel.hide()

        screen = QApplication.primaryScreen().geometry()
        main_geo = self.geometry()
        debug_x = main_geo.right() + 20
        if debug_x + 400 > screen.width():
            debug_x = main_geo.left() - 420
        self.debug_panel.move(max(0, debug_x), main_geo.y())

        QTimer.singleShot(500, self.detect_camera)

        QApplication.instance().installEventFilter(self)

        self.zen_overlay = FullscreenZenOverlay(self)
        self.zen_overlay.closed.connect(self.on_zen_closed)
        self.zen_overlay.navigated.connect(self.on_zen_navigate)
        self.zen_overlay.rotated.connect(self.on_zen_rotate)
        self.zen_overlay.remove_requested.connect(self.remove_current_from_project)

        # Defer the post-migration notice until the main window has had a
        # chance to render; singleShot(0) puts it at the back of the next
        # event-loop tick, after show().
        if self._migration_report is not None:
            QTimer.singleShot(0, self._show_migration_notice)

        # Probe how the GPU resolved once the window is up. Init is lazy and a
        # little slow, so defer it off the constructor; if we landed on a
        # software adapter or the CPU fallback, surface it instead of letting
        # the user wonder why renders crawl on capable hardware.
        QTimer.singleShot(0, self._check_gpu_health)

    def _check_gpu_health(self):
        try:
            status = gpu.status()
        except Exception:
            log.exception("[gpu] status probe failed")
            return
        if status['mode'] == 'gpu':
            return
        if status['mode'] == 'software':
            msg = (f"GPU not in use — running on a software renderer "
                   f"({status['summary']}). Renders will be slow; update your "
                   f"graphics drivers.")
        elif not status['available']:
            msg = ("GPU acceleration off — the 'wgpu' library is not installed. "
                   "Renders will be slow. Install dependencies: "
                   "pip install -r requirements.txt")
        else:
            msg = ("GPU not in use — running on the CPU fallback. Renders will "
                   "be slow; check that GPU drivers are installed and current.")
        log.warning("[gpu] %s", msg)
        if hasattr(self, 'mode_label'):
            self.mode_label.setText("⚠ " + msg)
            self.mode_label.setStyleSheet(f"color: {C['accent']};")

    def _show_migration_notice(self):
        """Show the one-shot post-migration summary dialog (step 6).

        Dismissal persists in the v2 envelope via
        vibe_state.mark_migration_acknowledged so the dialog never fires
        twice for the same migration."""
        from .migration_notice import MigrationNoticeDialog
        dlg = MigrationNoticeDialog(self._migration_report, parent=self)
        dlg.show()  # non-modal — user can keep working with the editor
        self._migration_notice_dialog = dlg  # keep a ref so it isn't GC'd

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
        if 0 <= new_index < len(self.image_files) and new_index != self.current_index:
            self.current_index = new_index
            self.load_current_image()

    def on_zen_rotate(self, angle):
        if angle == 90:
            self.rotate_clockwise()
        else:
            self.rotate_counterclockwise()

    def on_zen_closed(self):
        self.refresh_from_debug()

    # ===================================================================
    # ROTATION
    # ===================================================================

    def rotate_clockwise(self):
        if not self.image_files:
            return
        if hasattr(self, '_render_worker'):
            self._render_worker.invalidate()
        img_array = self.processor.rotate_clockwise()
        self.display_image(img_array)
        self.update_current_thumbnail(img_array)
        file_path = str(self.image_files[self.current_index])
        self.image_cache[file_path] = self.processor.intermediate_acescg.copy()
        self.image_rotations[file_path] = (self.image_rotations.get(file_path, 0) + 90) % 360

    def rotate_counterclockwise(self):
        if not self.image_files:
            return
        if hasattr(self, '_render_worker'):
            self._render_worker.invalidate()
        img_array = self.processor.rotate_counterclockwise()
        self.display_image(img_array)
        self.update_current_thumbnail(img_array)
        file_path = str(self.image_files[self.current_index])
        self.image_cache[file_path] = self.processor.intermediate_acescg.copy()
        self.image_rotations[file_path] = (self.image_rotations.get(file_path, 0) - 90) % 360

    # ===================================================================
    # LUT LOADING
    # ===================================================================

    def _load_custom_lut(self):
        """Prompt for a .cube file. Stored on the current vibe as a
        `user:<absolute path>` ref so it becomes part of the active vibe's
        session state and can be saved with it."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select LUT", "", "LUT Files (*.cube)"
        )
        if not file_path:
            return
        try:
            custom_lut = colour.io.read_LUT(file_path)
            self._lut_cache[file_path] = custom_lut
            self.processor.lut = custom_lut
            gpu.upload_lut(custom_lut.table)
            self.current_vibe.lut_ref = f"user:{file_path}"
            self._active_lut_ref = self.current_vibe.lut_ref
            # User just imported a fresh LUT — any preserved pre-1.5 path
            # is no longer the active choice, so drop the breadcrumb.
            self.current_vibe.legacy_user_lut = ''
            self.debug_panel.refresh_lut_label()
            self.debug_panel.update_modified_indicator()
            self.refresh_from_debug()
        except Exception as e:
            QMessageBox.warning(self, "LUT Load Error", f"Failed to parse LUT file:\n{e}")

    # ===================================================================
    # EVENT FILTER (arrow key navigation + double-click sliders to reset)
    # ===================================================================

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress:
            # Always route arrow keys to image navigation/rotation, regardless
            # of which widget has focus.  Guard against modal dialogs (file
            # picker, message boxes) and unrelated windows.
            active = QApplication.activeWindow()
            if active in (self, getattr(self, 'zen_overlay', None)):
                key = event.key()
                if key == Qt.Key_Left:
                    if self.image_files and self.current_index > 0:
                        self.current_index -= 1
                        self.load_current_image()
                    return True
                elif key == Qt.Key_Right:
                    if self.image_files and self.current_index < len(self.image_files) - 1:
                        self.current_index += 1
                        self.load_current_image()
                    return True
                elif key == Qt.Key_Up:
                    if self.image_files:
                        self.rotate_clockwise()
                    return True
                elif key == Qt.Key_Down:
                    if self.image_files:
                        self.rotate_counterclockwise()
                    return True

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
        self.processor.adjustments.exposure_ev = 0.0
        self.save_current_settings()
        self._render_needs_commit = True
        self._render_worker.request(downscale=False)

    def reset_wb_slider(self):
        self.slider_wb.blockSignals(True)
        self.slider_wb.setValue(0)
        self.slider_wb.blockSignals(False)
        self.label_wb.setText("5600 K")
        self.processor.adjustments.wb_temp = 0.0
        if self.chk_wb_link.isChecked():
            self._tint_manual_offset = 0.0
            self.slider_tint.blockSignals(True)
            self.slider_tint.setValue(0)
            self.slider_tint.blockSignals(False)
            self.label_tint.setText("+0")
            self.processor.adjustments.tint = 0.0
        self.save_current_settings()
        self._render_needs_commit = True
        self._render_worker.request(downscale=False)

    def reset_tint_slider(self):
        self._tint_manual_offset = 0.0
        self.slider_tint.blockSignals(True)
        self.slider_tint.setValue(0)
        self.slider_tint.blockSignals(False)
        self.label_tint.setText("+0")
        self.processor.adjustments.tint = 0.0
        self.save_current_settings()
        self._render_needs_commit = True
        self._render_worker.request(downscale=False)

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
        """Re-run every registered stylesheet and icon with the current palette.

        Each refresh step is guarded so one bad widget can't abort the whole
        palette swap; failures are logged at debug (off by default) so a
        systematic breakage is still diagnosable instead of fully silent.
        """
        for widget, builder in self._themed_styles:
            try:
                widget.setStyleSheet(builder())
            except Exception:
                log.debug("theme: stylesheet refresh failed for %r", widget, exc_info=True)
        for button, rel_path, color_token, size in self._themed_icons:
            try:
                button.setIcon(svg_icon(rel_path, color_token, size))
            except Exception:
                log.debug("theme: icon refresh failed for %s", rel_path, exc_info=True)
        # Regenerate drag-overlay strings (they hold cached accent/text colours)
        if hasattr(self, "_rebuild_drag_styles"):
            self._rebuild_drag_styles()
        # Dynamic styles (format pills, mode label, process-button-done, etc.)
        # aren't registered — they're reapplied by their owners on the next
        # state change. Trigger that here so the palette swap is immediate.
        if hasattr(self, "btn_export_jpeg") and hasattr(self, "export_mode"):
            try:
                self.set_export_mode(self.export_mode)
            except Exception:
                log.debug("theme: export-mode restyle failed", exc_info=True)
        if hasattr(self, "mode_label"):
            try:
                self.update_mode_label()
            except Exception:
                log.debug("theme: mode-label restyle failed", exc_info=True)
        # Force a repaint on widgets that read the palette inside paintEvent
        for w in self._themed_repaint:
            try:
                w.update()
            except Exception:
                log.debug("theme: repaint failed for %r", w, exc_info=True)
        # Apple/Windows native chrome needs to follow the theme too
        if getattr(self, "_native_chrome_applied", False):
            try:
                from ui import native_chrome
                native_chrome.apply(self, theme.current_theme())
            except Exception:
                log.debug("theme: native chrome refresh failed", exc_info=True)

    def set_dng_profile_name(self, name: str):
        self.current_vibe.dng_profile_name = name
        self.app_settings.setValue("dng_profile_name", name)

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
        self.setWindowTitle("LoFi Logic")
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
        self.thumbnail_strip.thumbnail_remove_requested.connect(self.remove_from_project)
        fl.addWidget(self.thumbnail_strip)

        self.fade_overlay = FadeOverlayWidget(self.thumbnail_strip)
        root.addWidget(filmstrip)

        # ─────────────── STATUS BAR ───────────────
        root.addWidget(self._build_status_bar())

        self.loader_overlay = LoaderOverlay(self.centralWidget())
        self.settings_clipboard = None

        # ⌘R resets all sliders
        reset_sc = QAction(self)
        reset_sc.setShortcut(QKeySequence("Ctrl+R"))
        reset_sc.triggered.connect(self.reset_all_sliders)
        self.addAction(reset_sc)

        # 1–4 select vibe preset
        for key, vibe_id in (('1', 'disposable'), ('2', 'point_shoot'),
                             ('3', 'rangefinder'), ('4', 'monochrome'),
                             ('5', 'flashback_classic_v1')):
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
        """Vibe changed: load saved VibeConfig if any, else factory; apply."""
        vibe = self._vibe_for(vibe_id)
        self._apply_vibe(vibe_id, vibe, refresh_thumbnails=True)

    def _vibe_for(self, vibe_id: str) -> VibeConfig:
        """Saved VibeConfig if present for `vibe_id`, otherwise the factory recipe."""
        saved = vibe_state.load_all().get(vibe_id)
        if saved is not None:
            return saved
        return vibe_config_for(vibe_id)

    def _apply_vibe(self, vibe_id: str, vibe: VibeConfig, refresh_thumbnails: bool = False):
        """Install `vibe` as the active vibe: bind it to the processor, load
        its LUT, sync the debug panel, and refresh the preview."""
        if hasattr(self, '_render_worker'):
            self._render_worker.invalidate()
        # Preserve the persistent profile name across vibe swaps — it's an
        # app-wide preference, not a per-vibe field.
        profile_name = self.current_vibe.dng_profile_name
        self.current_vibe = vibe
        self.current_vibe.dng_profile_name = profile_name
        self.processor.vibe = self.current_vibe
        # Tag the per-image record with the new vibe id (for future Save Project).
        self.processor.adjustments.active_vibe_id = vibe_id
        self._apply_effective_lut()
        if hasattr(self, 'debug_panel'):
            self.debug_panel.sync_from_config()
            self.debug_panel.update_modified_indicator()
        self.refresh_from_debug()
        if refresh_thumbnails:
            self._refresh_all_thumbnails()

    def _apply_effective_lut(self, file_path: str = None):
        """Load the LUT actually used to render `file_path` (the current image
        if omitted), applying any V1 per-file override of the active vibe's LUT.

        The override is transient: it's pushed to the processor/GPU but never
        written back into current_vibe.lut_ref, so saving the vibe keeps its
        canonical (V2) LUT. No-ops when the effective ref is already loaded."""
        if file_path is None and self.image_files:
            file_path = str(self.image_files[self.current_index])
        is_v1 = bool(file_path) and is_v1_negative(file_path)
        base = self.current_vibe.lut_ref
        eff = effective_lut_ref(base, is_v1)
        if eff == self._active_lut_ref:
            return
        # Persist only when no override is in play, preserving the existing
        # user-LUT fallback bookkeeping for the canonical case.
        self._load_lut_from_ref(eff, persist=(eff == base))

    def _lut_obj(self, ref: str):
        """Resolve a tagged LUT ref to a cached `colour` LUT object, or None."""
        from core.config import resolve_lut_ref
        if not ref:
            return None
        resolved, _ = resolve_lut_ref(ref)
        if not resolved:
            return None
        if resolved not in self._lut_cache:
            try:
                self._lut_cache[resolved] = colour.io.read_LUT(resolved)
            except Exception as e:
                log.warning("⚠ Could not load LUT '%s': %s", resolved, e)
                return None
        return self._lut_cache[resolved]

    def _v1_variant_lut(self):
        """The V1-tuned LUT object for the active vibe (e.g. disposable_V1), or
        None when the vibe's LUT has no V1 variant. Passed to thumbnail workers
        so V1 negatives in a mixed roll render with the right LUT."""
        base = self.current_vibe.lut_ref
        v1_ref = effective_lut_ref(base, True)
        return self._lut_obj(v1_ref) if v1_ref != base else None

    def _load_lut_from_ref(self, lut_ref: str, persist: bool = True):
        """Resolve a tagged LUT ref (`factory:<id>` or `user:<path>`) to an
        absolute path via core.config.resolve_lut_ref, load + cache, push
        into processor + GPU. Empty ref clears the LUT so the tone-curve
        fallback renders.

        A `user:` ref whose file no longer exists falls back to the LUT
        the vibe normally ships with (whichever factory id matches the
        active vibe), so a missing custom LUT doesn't degrade further than
        the factory look. The post-migration / startup summary handles
        surfacing this to the user.
        """
        from core.config import resolve_lut_ref, vibe_config_for, LUT_REF_FACTORY
        if not lut_ref:
            self.processor.lut = None
            self._active_lut_ref = None
            return
        resolved, origin = resolve_lut_ref(lut_ref)
        if resolved is None and origin == 'user':
            log.warning("⚠ Custom LUT missing: %s. Falling back to factory LUT.", lut_ref)
            try:
                vibe_id = self.current_vibe_id()
                fallback_ref = vibe_config_for(vibe_id).lut_ref
            except (KeyError, AttributeError):
                fallback_ref = ''
            if fallback_ref and fallback_ref != lut_ref:
                self._load_lut_from_ref(fallback_ref, persist=persist)
            else:
                self.processor.lut = None
                self._active_lut_ref = None
            return
        if resolved is None:
            log.warning("⚠ Could not resolve LUT ref %r", lut_ref)
            self.processor.lut = None
            self._active_lut_ref = None
            return
        try:
            if resolved not in self._lut_cache:
                self._lut_cache[resolved] = colour.io.read_LUT(resolved)
            lut = self._lut_cache[resolved]
            self.processor.lut = lut
            gpu.upload_lut(lut.table)
            self._active_lut_ref = lut_ref
            if persist:
                self.current_vibe.lut_ref = lut_ref
        except Exception as e:
            log.warning("⚠ Could not load LUT '%s': %s", resolved, e)

    # -------------------------------------------------------------------
    # Per-vibe save / reset
    # -------------------------------------------------------------------

    def current_vibe_id(self) -> str:
        return self.vibe_picker.current_vibe()

    def save_current_vibe_defaults(self):
        """Promote the live VibeConfig to saved defaults for the active vibe."""
        vibe_id = self.current_vibe_id()
        vibe_state.save_one(vibe_id, self.current_vibe)
        if hasattr(self, 'debug_panel'):
            self.debug_panel.update_modified_indicator()
            self.debug_panel.status_label.setText(f"Saved defaults for {vibe_id}.")

    def reset_current_vibe_to_saved(self):
        """Discard session edits, reload saved defaults (or factory if no saved)."""
        vibe_id = self.current_vibe_id()
        self._apply_vibe(vibe_id, self._vibe_for(vibe_id), refresh_thumbnails=True)
        if hasattr(self, 'debug_panel'):
            label = "saved" if vibe_state.has_saved(vibe_id) else "factory (no saved defaults)"
            self.debug_panel.status_label.setText(f"Reset {vibe_id} to {label}.")

    def reset_current_vibe_to_factory(self):
        """Wipe saved defaults for the active vibe and apply factory state."""
        vibe_id = self.current_vibe_id()
        vibe_state.clear_one(vibe_id)
        self._apply_vibe(vibe_id, vibe_config_for(vibe_id), refresh_thumbnails=True)
        if hasattr(self, 'debug_panel'):
            self.debug_panel.status_label.setText(f"Reset {vibe_id} to factory defaults.")

    _DEFAULT_USER_SETTINGS = {'exposure_ev': 0.0, 'wb_temp': 0, 'tint': 0.0, 'push_pull_ev': 0.0}

    def _refresh_all_thumbnails(self):
        """Re-render every cached thumbnail in the background after a vibe change."""
        if not self.image_files:
            return

        # Stop any in-flight refresh before starting a new one.
        self._stop_vibe_refresh_worker()

        # Snapshot cache keys on the main thread so the worker never iterates the
        # live dict. Share the array references (shallow) rather than deep-copying
        # every intermediate — that copy duplicated the entire cache (gigabytes)
        # into the worker. Safe because cache values are never mutated in place
        # (always replaced) and the worker copies each array before rendering.
        cache_snapshot = dict(self.image_cache.items())

        self._vibe_refresh_worker = VibeRefreshWorker(
            image_files=self.image_files,
            cache_snapshot=cache_snapshot,
            image_settings=self.image_settings.copy(),
            current_index=self.current_index,
            lut=self._lut_obj(self.current_vibe.lut_ref),
            grain_tiles=self.processor.grain_tiles,
            default_settings=self._DEFAULT_USER_SETTINGS.copy(),
            vibe=self.current_vibe.copy(),
            lut_v1=self._v1_variant_lut(),
        )
        self._vibe_refresh_worker.thumbnail_ready.connect(self._on_vibe_refresh_thumbnail)
        self._vibe_refresh_worker.start()

    def _on_vibe_refresh_thumbnail(self, index, thumb_array):
        file_path = str(self.image_files[index])
        self.thumbnail_cache[file_path] = thumb_array
        self.thumbnail_strip.update_thumbnail(index, thumb_array)

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

        # Format pills (JPEG / DNG)
        pills_row = QHBoxLayout()
        pills_row.setSpacing(6)
        self.btn_export_jpeg = QPushButton("JPEG")
        self.btn_export_dng  = QPushButton("DNG")
        for b in (self.btn_export_jpeg, self.btn_export_dng):
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(28)
        self.btn_export_jpeg.setToolTip("Final film look (JPEG)")
        self.btn_export_dng.setToolTip("Clean RAW DNG for Camera Raw / Lightroom")
        self.btn_export_jpeg.clicked.connect(lambda: self.set_export_mode('jpeg'))
        self.btn_export_dng.clicked.connect(lambda: self.set_export_mode('dng'))
        pills_row.addWidget(self.btn_export_jpeg, 1)
        pills_row.addWidget(self.btn_export_dng,  1)
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
        self.set_export_mode('jpeg')
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

        # ── About / Help ──────────────────────────────────────────────
        # "About" with AboutRole moves to the app menu automatically on macOS.
        # We put it in a Help menu so it appears somewhere on Windows/Linux too.
        help_menu = mb.addMenu("Help")

        act_about = QAction("About LoFi Logic", self)
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

        act_open_project = QAction("Open Project…", self)
        act_open_project.setShortcut(QKeySequence("Ctrl+Shift+O"))
        act_open_project.triggered.connect(lambda: self.open_project())
        file_menu.addAction(act_open_project)

        self.recent_projects_menu = file_menu.addMenu("Open Recent Project")
        self.recent_projects_menu.aboutToShow.connect(self._rebuild_recent_projects_menu)
        # Initial population so the menu isn't empty before first show.
        self._rebuild_recent_projects_menu()

        act_save_project = QAction("Save Project", self)
        act_save_project.setShortcut(QKeySequence.StandardKey.Save)  # Cmd+S / Ctrl+S
        act_save_project.triggered.connect(self.save_project)
        file_menu.addAction(act_save_project)

        act_save_project_as = QAction("Save Project As…", self)
        act_save_project_as.setShortcut(QKeySequence.StandardKey.SaveAs)  # Cmd+Shift+S
        act_save_project_as.triggered.connect(self.save_project_as)
        file_menu.addAction(act_save_project_as)

        file_menu.addSeparator()

        act_export_jpg = QAction("Export JPGs", self)
        act_export_jpg.triggered.connect(self.export_as_jpeg)
        file_menu.addAction(act_export_jpg)

        act_export_dng = QAction("Export DNGs", self)
        act_export_dng.triggered.connect(self.export_as_dng)
        file_menu.addAction(act_export_dng)

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

        edit_menu.addSeparator()

        act_remove = QAction("Remove from Project", self)
        act_remove.setShortcut(QKeySequence(Qt.Key_Delete))
        act_remove.triggered.connect(self.remove_current_from_project)
        edit_menu.addAction(act_remove)

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
            "About LoFi Logic",
            f"<b>LoFi Logic</b><br>"
            f"Version {__version__}<br><br>"
            "A RAW editor for the Flashback One35 cameras.<br><br>"
            "© 2026 LoFi Logic"
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
        self.set_export_mode('jpeg')
        self.process_all_images()

    def export_as_dng(self):
        self.set_export_mode('dng')
        self.process_all_images()

    def export_lut_tiffs(self, output_dir, reverse_ae=False):
        """Export ACEScct TIFFs for all selected (or all) images.

        reverse_ae=False (default) exports at the app's standard exposure — the
        right input for previewing a hand-built LUT. reverse_ae=True normalises
        each frame by its EXIF shutter speed for real film-stock profiling.
        """
        if not self.image_files:
            return 0, 0

        selected_indices = self.thumbnail_strip.get_process_selected_indices()
        indices_to_process = sorted(selected_indices) if selected_indices else list(range(len(self.image_files)))
        total = len(indices_to_process)

        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_process_all.setEnabled(False)

        success_count = 0
        for i, idx in enumerate(indices_to_process):
            file_path = str(self.image_files[idx])
            try:
                self.progress_bar.setValue(i)
                self.mode_label.setText(f"Exporting TIFF {i+1}/{total}…")
                QApplication.processEvents()

                # Always re-load from DNG: needed so _rev_gain_unconditional is
                # computed from this file's EXIF, not a previously cached image.
                self.processor.load_image(file_path)

                if file_path in self.image_settings:
                    self.processor.set_settings(self.image_settings[file_path])

                base_name = export_basename(file_path)
                suffix = "_lut_profile" if reverse_ae else "_standard"
                output_path = os.path.join(output_dir, f"{base_name}{suffix}.tif")
                if export_image(self.processor, output_path, as_tiff=True,
                                lut_profiling=True, reverse_ae=reverse_ae):
                    success_count += 1
            except Exception as e:
                log.error("Error exporting TIFF for %s: %s", file_path, e)
                traceback.print_exc()
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

        self.progress_bar.setVisible(False)
        self.btn_process_all.setEnabled(True)
        self.load_current_image()
        self.update_mode_label()
        return success_count, total

    def center_window(self):
        frame_geo = self.frameGeometry()
        screen_geo = QApplication.primaryScreen().availableGeometry()
        frame_geo.moveCenter(screen_geo.center())
        self.move(frame_geo.topLeft())

    # ===================================================================
    # FILE MANAGEMENT
    # ===================================================================

    # Volume labels that identify a Flashback camera (fast-path; the camera
    # is recognised by content even if the user renames the SD card).
    CAMERA_VOLUME_NAMES = {'ONE35 V2', 'ONE35'}

    # The camera is a disposable-style device with a fixed frame budget; any
    # volume holding more DNGs than this can't be a Flashback camera and is
    # almost certainly someone's photo library backup.
    CAMERA_MAX_FRAMES = 27

    # Camera-issued filename shape: SN<serial>_<frame>.dng
    _CAMERA_DNG_PATTERN = re.compile(r'^SN\d+_\d+\.dng$', re.IGNORECASE)

    @classmethod
    def _looks_like_flashback_camera(cls, mount: Path, vol_name: str, dng_files: list) -> bool:
        """Heuristic check: does this mount look like a Flashback camera?

        Rejects anything over the camera's frame capacity, then accepts on
        any of: known volume label, presence of the camera's UNPROCESSED_JPG
        sibling folder, or all DNG filenames matching the SN-pattern.
        """
        if len(dng_files) > cls.CAMERA_MAX_FRAMES:
            return False
        # Known label: trust even if the card is empty (freshly erased camera).
        if vol_name in cls.CAMERA_VOLUME_NAMES:
            return True
        if not dng_files:
            return False
        if (mount / 'UNPROCESSED_JPG').is_dir():
            return True
        if all(cls._CAMERA_DNG_PATTERN.match(p.name) for p in dng_files):
            return True
        return False

    @staticmethod
    def _get_volume_name(path: Path) -> str:
        """Return the volume label for a mount point (cross-platform)."""
        if sys.platform == 'win32':
            import ctypes
            buf = ctypes.create_unicode_buffer(1024)
            try:
                ctypes.windll.kernel32.GetVolumeInformationW(
                    str(path), buf, len(buf), None, None, None, None, 0
                )
                return buf.value
            except Exception:
                return ''
        return path.name

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
            # Modern desktop distros (Ubuntu, Fedora) auto-mount removable
            # media at /media/<username>/<label> or /run/media/<username>/
            # <label>. Walk one level deeper than /media so we see actual
            # volumes, not per-user buckets. /mnt is typically used for
            # manual mounts and lives at the top level.
            for base in (Path("/media"), Path("/run/media")):
                if not base.exists():
                    continue
                for child in base.iterdir():
                    if not child.is_dir():
                        continue
                    try:
                        mount_points.extend(child.iterdir())
                    except (PermissionError, OSError):
                        continue
            if Path("/mnt").exists():
                mount_points.extend(Path("/mnt").iterdir())

        for mount in mount_points:
            if not mount.is_dir():
                continue
            vol_name = self._get_volume_name(mount)
            try:
                dng_files = list(set(mount.glob("*.dng")) | set(mount.glob("*.DNG")))
            except (PermissionError, OSError):
                continue
            if not self._looks_like_flashback_camera(mount, vol_name, dng_files):
                continue
            if dng_files:
                from core.camera_import import plan_imports
                to_import, skipped = plan_imports(dng_files, Path(self.camera_import_dir))
                new_n = len(to_import)
                skip_n = len(skipped)
                if new_n == 0:
                    reply = QMessageBox.question(
                        self, "Camera Detected",
                        f"{vol_name}: all {skip_n} DNGs already imported.\n\n"
                        f"Open the existing copies from {self.camera_import_dir}?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        self.load_image_files(skipped)
                    return
                msg = f"Found {len(dng_files)} DNG files on {vol_name}.\n\n"
                msg += f"Import {new_n} new file(s) into {self.camera_import_dir}?"
                if skip_n:
                    msg += f"\n({skip_n} already imported and will be skipped.)"
                reply = QMessageBox.question(
                    self, "Camera Detected", msg,
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    target_files = [tgt for _src, tgt in to_import]
                    export_sources = {str(tgt): str(src) for src, tgt in to_import}
                    # Include the already-imported copies so the strip shows
                    # the whole roll, not just the freshly-exported subset.
                    self.load_image_files(
                        target_files + skipped,
                        export_sources=export_sources,
                    )
            else:
                QMessageBox.information(
                    self, "Camera Connected",
                    f"{vol_name} is connected but contains no DNG files."
                )
            return

    RECENT_PROJECTS_MAX = 8

    def _recent_projects(self):
        raw = self.app_settings.value("recent_projects", []) or []
        if isinstance(raw, str):  # QSettings sometimes returns a single string
            raw = [raw]
        return [str(p) for p in raw]

    def _remember_recent_project(self, path):
        path = str(path)
        items = [p for p in self._recent_projects() if p != path]
        items.insert(0, path)
        items = items[: self.RECENT_PROJECTS_MAX]
        self.app_settings.setValue("recent_projects", items)
        if hasattr(self, 'recent_projects_menu'):
            self._rebuild_recent_projects_menu()

    def _remove_recent_project(self, path):
        path = str(path)
        items = [p for p in self._recent_projects() if p != path]
        self.app_settings.setValue("recent_projects", items)
        if hasattr(self, 'recent_projects_menu'):
            self._rebuild_recent_projects_menu()

    def _rebuild_recent_projects_menu(self):
        menu = self.recent_projects_menu
        menu.clear()
        items = [p for p in self._recent_projects() if Path(p).exists()]
        if not items:
            act = QAction("(No recent projects)", self)
            act.setEnabled(False)
            menu.addAction(act)
            return
        for p in items:
            label = Path(p).name
            act = QAction(label, self)
            act.setToolTip(p)
            act.triggered.connect(lambda _checked=False, pth=p: self.open_project(pth))
            menu.addAction(act)
        menu.addSeparator()
        act_clear = QAction("Clear Menu", self)
        act_clear.triggered.connect(lambda: (
            self.app_settings.setValue("recent_projects", []),
            self._rebuild_recent_projects_menu(),
        ))
        menu.addAction(act_clear)

    def save_project(self):
        """Save to the currently-open project file, or prompt if none."""
        if self.current_project_path is None:
            return self.save_project_as()
        self._write_project_to(self.current_project_path)

    def save_project_as(self):
        from core.project import PROJECT_EXT
        if not self.image_files:
            QMessageBox.information(self, "Save Project", "No images are open.")
            return
        default_dir = self.app_settings.value("last_project_dir", self.output_dir)
        suggested = (str(self.current_project_path)
                     if self.current_project_path
                     else str(Path(default_dir) / f"Untitled{PROJECT_EXT}"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", suggested,
            f"LoFi Logic Project (*{PROJECT_EXT})"
        )
        if not path:
            return
        self._write_project_to(Path(path))

    def _write_project_to(self, path):
        from core.project import save_project
        if not self.image_files:
            return
        # Commit any in-flight edits so the saved project reflects the UI.
        cur = str(self.image_files[self.current_index])
        self.image_settings[cur] = self.processor.get_settings()
        try:
            written = save_project(
                path, self.image_files, self.image_settings,
                image_rotations=self.image_rotations,
                current_index=self.current_index,
            )
            self.current_project_path = written
            self.app_settings.setValue("last_project_dir", str(written.parent))
            self._remember_recent_project(written)
        except Exception as e:
            log.error("Save project failed: %s", e)
            QMessageBox.critical(self, "Save Project", f"Failed to save project:\n{e}")

    def open_project(self, path=None):
        from core.project import load_project, PROJECT_EXT, LEGACY_PROJECT_EXT
        if not path:
            default_dir = self.app_settings.value("last_project_dir", self.output_dir)
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Project", default_dir,
                f"LoFi Logic Project (*{PROJECT_EXT} *{LEGACY_PROJECT_EXT})"
            )
            if not path:
                return
        try:
            image_files, image_settings, image_rotations, current_index = load_project(path)
        except Exception as e:
            log.error("Open project failed: %s", e)
            QMessageBox.critical(self, "Open Project", f"Failed to open project:\n{e}")
            self._remove_recent_project(path)
            return
        if not image_files:
            QMessageBox.warning(self, "Open Project",
                                "None of the project's images could be found on disk.")
            return
        self.app_settings.setValue("last_project_dir", str(Path(path).parent))
        self._remember_recent_project(path)
        self.image_settings = dict(image_settings)
        # Pick the original active-file path so the alphabetical re-sort
        # inside load_image_files doesn't drift the selection.
        active_path = (str(image_files[current_index])
                       if 0 <= current_index < len(image_files) else None)
        self.load_image_files(
            image_files,
            image_rotations=image_rotations,
            current_path=active_path,
        )
        # load_image_files cleared current_project_path; re-attach so Save
        # writes back to this file.
        self.current_project_path = Path(path)

    def open_os_path(self, path):
        """Open a path the OS handed us (file association double-click, "Open
        With", or a command-line argument). Routes project files to
        open_project and image/zip/folder inputs through the normal load path,
        so double-clicking a .lofi behaves like File → Open Project."""
        from core.project import PROJECT_EXT, LEGACY_PROJECT_EXT
        if not path or not os.path.exists(path):
            return
        if path.lower().endswith((PROJECT_EXT, LEGACY_PROJECT_EXT)):
            self.open_project(path)
            return
        resolved = self._resolve_input_paths([path])
        if resolved:
            self.load_image_files(resolved)

    def open_files(self):
        default_dir = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation) or str(Path.home())
        start_dir = self.app_settings.value("last_open_dir", default_dir)

        lower_exts = [f"*{ext}" for ext in self.SUPPORTED_EXTENSIONS]
        upper_exts = [f"*{ext.upper()}" for ext in self.SUPPORTED_EXTENSIONS]
        filter_string = (f"Supported Images ({' '.join(lower_exts + upper_exts + ['*.zip'])});;"
                         f"Flashback V1 Roll (*.zip)")

        files, _ = QFileDialog.getOpenFileNames(self, "Select Image Files", start_dir, filter_string)

        if files:
            new_dir = str(Path(files[0]).parent)
            self.app_settings.setValue("last_open_dir", new_dir)
            self.load_image_files(self._resolve_input_paths(files))

    def _stop_vibe_refresh_worker(self):
        """Stop, join, and release the vibe-refresh worker.

        Releasing the reference (not just stopping) is what frees memory: the
        worker holds a snapshot of the whole image cache, so a lingering
        ``self._vibe_refresh_worker`` keeps that entire snapshot alive — the
        cause of RAM not dropping when a project is replaced.
        """
        w = self._vibe_refresh_worker
        if w is None:
            return
        w.blockSignals(True)
        w.stop()
        if w.isRunning():
            w.wait()
        self._vibe_refresh_worker = None

    def _stop_thumbnail_workers(self):
        """Stop and join any running thumbnail workers so their QThreads never
        outlive the Python owner — that aborts with "QThread: Destroyed while
        thread is still running". signals are blocked first so a stale
        ``finished`` emission can't fire its slot against a replacement worker.
        V1 negatives develop slowly, widening the window where a worker is
        mid-flight on close/reload."""
        for attr in ('thumbnail_worker', 'add_thumbnail_worker'):
            w = getattr(self, attr, None)
            if w is None:
                continue
            w.blockSignals(True)
            w._is_running = False
            if w.isRunning():
                w.wait()
            setattr(self, attr, None)

    def load_image_files(self, image_files, export_sources=None,
                          image_rotations=None, current_path=None):
        if not image_files:
            return

        self._stop_vibe_refresh_worker()
        # Retire any in-flight thumbnail pass before we replace the worker
        # reference below, or the old QThread is orphaned while still running.
        self._stop_thumbnail_workers()

        # Always present in alphabetical order (by filename, case-insensitive)
        # regardless of source ordering or OS settings.
        image_files = sorted(image_files, key=lambda p: Path(str(p)).name.lower())

        # Any fresh image load detaches us from the previously-open project,
        # so subsequent Save uses Save-As semantics. open_project re-assigns
        # current_project_path after calling load_image_files.
        self.current_project_path = None

        self.image_files = image_files
        self.image_cache.clear()
        self.preview_cache.clear()
        self.thumbnail_cache.clear()
        self._file_is_flashback.clear()
        self.image_rotations = dict(image_rotations) if image_rotations else {}
        self.thumbnail_strip.clear()

        # Pick starting index: caller-provided path wins, else first image.
        self.current_index = 0
        if current_path:
            target = str(current_path)
            for i, p in enumerate(self.image_files):
                if str(p) == target:
                    self.current_index = i
                    break

        self.btn_process_all.setEnabled(True)
        self.update_process_button_text()

        expected_thumb_width = 105
        layout_spacing = 5
        final_width = len(self.image_files) * (expected_thumb_width + layout_spacing)
        self.thumbnail_strip.container.setMinimumWidth(final_width)

        # When importing from a camera, the first target file does not yet
        # exist on disk — export it synchronously so load_current_image() has
        # something to read. Pre-fill the image cache with what the export
        # helper already loaded so load_current_image avoids a second read.
        # The background worker skips this file (target now exists) and moves
        # on to the next.
        if export_sources:
            first = str(image_files[0])
            src = export_sources.get(first)
            if src and not os.path.exists(first):
                try:
                    from core.camera_import import export_camera_dng
                    preloaded = export_camera_dng(src, first, self.processor)
                    if preloaded is not None:
                        self.image_cache[first] = self.processor.intermediate_acescg.copy()
                        self._file_is_flashback[first] = bool(self.processor.is_flashback_file)
                except Exception as e:
                    log.error("First-image camera export failed: %s", e)

        self.load_current_image()

        if hasattr(self, 'loader_overlay'):
            self.loader_overlay.fade_in()
            self.loader_overlay.update_progress(0, len(self.image_files))

        self.thumbnail_worker = ThumbnailWorker(
            self.image_files,
            self._lut_obj(self.current_vibe.lut_ref),
            export_sources=export_sources,
            rotations=self.image_rotations,
            lut_v1=self._v1_variant_lut(),
        )

        self.thumbnail_worker.progress.connect(self.loader_overlay.update_progress)
        self.thumbnail_worker.thumbnail_ready.connect(self._add_thumbnail_to_ui)

        if hasattr(self, '_on_thumbnail_error'):
            self.thumbnail_worker.error.connect(self._on_thumbnail_error)
        if hasattr(self, '_on_thumbnails_finished'):
            self.thumbnail_worker.finished.connect(self._on_thumbnails_finished)

        self.thumbnail_worker.start()

    def _on_thumbnail_error(self, index, error_message):
        log.error("  ✗ Failed thumbnail %d: %s", index, error_message)
        try:
            if hasattr(self, 'loader_overlay') and self.loader_overlay.isVisible():
                self.loader_overlay.progress_label.setText(f"Error at {index}: {error_message}")
                QTimer.singleShot(1500, lambda: self.loader_overlay.update_progress(index + 1, len(self.image_files)))
        except Exception:
            log.debug("loader overlay error display failed", exc_info=True)

    def _on_thumbnails_finished(self):
        self.thumbnails_loading = False
        log.info("✓ Thumbnail generation complete!")
        self.thumbnail_strip.container.setUpdatesEnabled(True)
        if self.thumbnail_worker:
            # ThumbnailWorker emits its own `finished` as the LAST line of run(),
            # i.e. while the QThread is still technically running. wait() blocks
            # until run() has actually returned (instant here) so the queued
            # deleteLater can't destroy a still-running QThread -> qFatal/abort.
            self.thumbnail_worker.wait()
            self.thumbnail_worker.deleteLater()
            self.thumbnail_worker = None
        try:
            if hasattr(self, 'loader_overlay'):
                self.loader_overlay.clear_and_hide()
        except Exception:
            log.debug("loader overlay hide failed", exc_info=True)

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
            self._lut_obj(self.current_vibe.lut_ref),
            lut_v1=self._v1_variant_lut(),
        )
        self.add_thumbnail_worker.progress.connect(self.loader_overlay.update_progress)
        self.add_thumbnail_worker.thumbnail_ready.connect(
            lambda i, t, mid, isfb, off=offset: self._add_thumbnail_to_ui(i + off, t, mid, isfb)
        )
        self.add_thumbnail_worker.finished.connect(self._on_add_thumbnails_finished)
        self.add_thumbnail_worker.start()

    def _on_add_thumbnails_finished(self):
        log.info("✓ Add-images thumbnail generation complete!")
        if hasattr(self, 'add_thumbnail_worker') and self.add_thumbnail_worker:
            # See _on_thumbnails_finished: join the thread before deleteLater so
            # the DeferredDelete can't hit a QThread that's still running.
            self.add_thumbnail_worker.wait()
            self.add_thumbnail_worker.deleteLater()
            self.add_thumbnail_worker = None
        try:
            if hasattr(self, 'loader_overlay'):
                self.loader_overlay.clear_and_hide()
        except Exception:
            log.debug("loader overlay hide failed", exc_info=True)
        self.update_mode_label()

    # ===================================================================
    # THUMBNAIL MANAGEMENT
    # ===================================================================

    def update_thumbnail_for_settings(self, index, settings, _processor=None):
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
                log.error("  ✗ Failed to update current thumbnail: %s", e)
        else:
            try:
                if file_path in self.image_cache:
                    temp_processor = _processor or FlashbackProcessor(vibe=self.current_vibe)
                    restore_lut = None
                    if _processor is None:
                        # Pick this file's LUT (V1 negatives may differ from the
                        # active image). The GPU LUT is thread-local and holds the
                        # active image's LUT, so swap in, render, then restore.
                        base = self._lut_obj(self.current_vibe.lut_ref)
                        v1 = self._v1_variant_lut()
                        chosen = v1 if (v1 is not None and is_v1_negative(file_path)) else base
                        temp_processor.lut = chosen
                        if chosen is not self.processor.lut:
                            if chosen is not None:
                                gpu.upload_lut(chosen.table)
                            restore_lut = self.processor.lut
                    temp_processor.intermediate_acescg = self.image_cache[file_path].copy()
                    temp_processor.current_file = file_path
                    temp_processor.set_settings(settings)
                    img_display = temp_processor._render_fast(downscale=True)
                    if restore_lut is not None:
                        gpu.upload_lut(restore_lut.table)
                    if img_display is not None:
                        h, w = img_display.shape[:2]
                        scale = 70 / h
                        new_w = int(w * scale)
                        thumb_array = cv2.resize(img_display, (new_w, 70), interpolation=cv2.INTER_LINEAR)
                        self.thumbnail_cache[file_path] = thumb_array
                        self.thumbnail_strip.update_thumbnail(index, thumb_array)
                        return
            except Exception as e:
                log.error("  ✗ Failed to update thumbnail %d: %s", index, e)

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

    def _add_thumbnail_to_ui(self, index, thumb_array, intermediate=None,
                             is_flashback=None):
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

        if intermediate is not None and self.image_files and index < len(self.image_files):
            file_path = str(self.image_files[index])
            self.image_cache[file_path] = intermediate
            if is_flashback is not None:
                # Persist the Flashback flag alongside the cached intermediate,
                # otherwise navigating to a worker-cached image leaves the DNG
                # export button greyed out (it only sees False from .get()).
                self._file_is_flashback[file_path] = bool(is_flashback)
                if index == self.current_index:
                    self._update_dng_button_state()

    def on_thumbnail_click(self, index):
        if 0 <= index < len(self.image_files):
            self.current_index = index
            self.load_current_image()

    def remove_current_from_project(self):
        self.remove_from_project(self.current_index)

    def remove_from_project(self, index):
        """Drop the image at `index` from the open set (no file deletion).
        Used to curate before saving a project."""
        if not self.image_files:
            return
        if not (0 <= index < len(self.image_files)):
            return
        file_path = str(self.image_files[index])

        self.image_files.pop(index)
        self.image_settings.pop(file_path, None)
        self.image_rotations.pop(file_path, None)
        self.image_cache.pop(file_path, None)
        self.preview_cache.pop(file_path, None)
        self.thumbnail_cache.pop(file_path, None)
        self._file_is_flashback.pop(file_path, None)
        self.thumbnail_strip.remove_at(index)

        if not self.image_files:
            self.current_index = 0
            # Drop the cached intermediate so refresh_from_debug() (triggered
            # by close_zen → on_zen_closed) doesn't re-render the removed
            # image back into the view.
            self.processor.intermediate_acescg = None
            self.processor.current_file = None
            # And cancel any in-flight background render — otherwise its
            # render_done would fire after the clear and re-set
            # image_label._original_pixmap, making the image reappear on the
            # next zoom/scroll.
            if hasattr(self, '_render_worker'):
                self._render_worker.invalidate()
            self._render_needs_commit = False
            self.image_label.clear()
            self.label_filename.setText("")
            self.label_counter.setText("0 / 0")
            self.btn_process_all.setEnabled(False)
            self.update_process_button_text()
            self.update_mode_label()
            # Zen mode shows the same image; with no images left, exit zen.
            # Defer the close so it happens after the current key-press handler
            # in the zen overlay unwinds (calling hide() from inside a child's
            # keyPressEvent can race with Qt re-asserting focus on the overlay).
            if hasattr(self, 'zen_overlay'):
                QTimer.singleShot(0, self.zen_overlay.close_zen)
            return

        # Stay on the same slot when possible; if we deleted the tail, fall
        # back to what is now the last image.
        self.current_index = min(index, len(self.image_files) - 1)
        self.update_process_button_text()
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

    def _preview_key(self):
        """Identity of the downscaled preview for the current processor state.

        A cached preview is reusable only if every input that the downscale
        render depends on is unchanged: the source intermediate (id changes on
        rotate/reload), the per-image sliders, the active vibe, and the LUT
        (V1 negatives swap in a variant). Keying on these makes a stale preview
        structurally impossible — any change yields a new key and a cache miss,
        so no manual invalidation is needed when settings or vibe change."""
        a = self.processor.adjustments
        return (
            id(self.processor.intermediate_acescg),
            round(a.exposure_ev, 4), round(a.wb_temp, 4), round(a.tint, 4),
            round(getattr(a, 'push_pull_ev', 0.0), 4),
            getattr(a, 'rotation', 0),
            self.current_vibe_id(),
            id(self.processor.lut),
        )

    def load_current_image(self):
        if not self.image_files:
            self.label_filename.setText("")
            self.label_counter.setText("0 / 0")
            return

        # Cancel any in-flight scrub render and clear the commit flag — the
        # processor state is about to change out from under the worker.
        if hasattr(self, '_render_worker'):
            self._render_worker.invalidate()
        self._render_needs_commit = False

        file_path = str(self.image_files[self.current_index])
        self.label_filename.setText(Path(file_path).name)
        self.label_counter.setText(f"{self.current_index + 1} / {len(self.image_files)}")
        if hasattr(self, 'thumbnail_strip'):
            self.thumbnail_strip.set_current_index(self.current_index)

        # Swap in the V1-tuned LUT when this frame is a negative and the active
        # vibe has a V1 variant (e.g. disposable) — before any render below.
        self._apply_effective_lut(file_path)

        if file_path in self.image_settings:
            settings = self.image_settings[file_path]
            self.processor.set_settings(settings)
            self.chk_wb_link.blockSignals(True)
            self.chk_wb_link.setChecked(settings.get('auto_tint', False))
            self.chk_wb_link.blockSignals(False)
            self.update_sliders_from_processor()
        else:
            self.processor.adjustments = ImageAdjustments(active_vibe_id=self.current_vibe_id())
            self.chk_wb_link.blockSignals(True)
            self.chk_wb_link.setChecked(False)
            self.chk_wb_link.blockSignals(False)
            self.update_sliders_from_processor()


        if file_path in self.image_cache:
            self.processor.intermediate_acescg = self.image_cache[file_path]
            self.processor.current_file = file_path
            # Restore Flashback status so DNG button reflects the correct state
            self.processor.is_flashback_file = self._file_is_flashback.get(file_path, False)
            self._update_dng_button_state()
            # Revisiting an image must be instant and must not block the UI
            # thread on a GPU readback (that readback serialises behind any
            # in-flight full-res render on the shared device — the freeze that
            # made switching show the previous image for seconds). Reuse the
            # cached downscaled preview when it still matches the current state;
            # only render synchronously on a genuine miss (first visit / changed
            # settings or vibe).
            key = self._preview_key()
            cached = self.preview_cache.get(file_path)
            if cached is not None and cached[0] == key:
                img_array = cached[1]
            else:
                img_array = self.processor.render_preview(downscale=True)
                self.preview_cache[file_path] = (key, img_array)
            self.display_image(img_array, is_scrub=True)
            self.update_current_thumbnail(img_array)
            self.update_mode_label()
            self._render_worker.request(downscale=False)
        else:
            if Path(file_path).suffix.lower() in ('.tif', '.tiff'):
                QMessageBox.information(
                    self, "TIFF Not Supported",
                    "TIFF intermediates cannot be imported in this version.\n\n"
                    "Open the original DNG instead, or use an older 1.1.x build "
                    "to continue working with existing TIFF intermediates."
                )
                return
            log.info("[Load] Image not in cache, loading from disk...")
            img_array = self.processor.load_image(file_path)
            if img_array is not None:
                self._file_is_flashback[file_path] = self.processor.is_flashback_file
                self._update_dng_button_state()
                # Re-apply any rotation persisted from a previous session.
                stored_rot = self.image_rotations.get(file_path, 0)
                if stored_rot:
                    self.processor.adjustments.rotation = stored_rot
                    img_array = self.processor._apply_rotation_and_render()
                self.image_cache[file_path] = self.processor.intermediate_acescg.copy()
                self.preview_cache[file_path] = (self._preview_key(), img_array)
                self.update_current_thumbnail(img_array)
                self.display_image(img_array, is_scrub=True)
                self.save_current_settings()
                self.update_mode_label()
                self._render_worker.request(downscale=False)
            else:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{file_path}")

    def display_image(self, img_array, is_scrub=False):
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
            if is_scrub:
                # Scale the scrub frame up to the full render's pixel dimensions so
                # the zen overlay layout is stable throughout the nav→render lifecycle.
                # Using processor.intermediate_acescg (not _original_pixmap) as the
                # target means rotated images and navigated images always match what
                # the background render will produce — no snap when it arrives.
                if (self.processor is not None and
                        self.processor.intermediate_acescg is not None):
                    full_h, full_w = self.processor.intermediate_acescg.shape[:2]
                    pixmap = pixmap.scaled(full_w, full_h,
                                           Qt.KeepAspectRatio, Qt.SmoothTransformation)
                else:
                    ref = self.image_label._original_pixmap
                    if ref and not ref.isNull():
                        pixmap = pixmap.scaled(ref.width(), ref.height(),
                                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                self.image_label._original_pixmap = pixmap  # full-res — update reference
            self.zen_overlay.update_preview(pixmap)
        else:
            if is_scrub:
                self.image_label.set_scrub_image(img_array)
            else:
                self.image_label.set_image(img_array)

    def update_sliders_from_processor(self):
        a = self.processor.adjustments

        self.slider_exposure.blockSignals(True)
        self.slider_wb.blockSignals(True)
        self.slider_tint.blockSignals(True)

        self.slider_exposure.setValue(int(a.exposure_ev * 10))
        self.slider_wb.setValue(int(a.wb_temp))
        self.slider_tint.setValue(int(round(a.tint * 5)))

        self.label_exposure.setText(f"{a.exposure_ev:.1f} EV")
        temp_absolute = 5600 + int(a.wb_temp)
        self.label_wb.setText(f"{temp_absolute} K")
        self.label_tint.setText(f"{int(round(a.tint * 5)):+d}")

        self.slider_exposure.blockSignals(False)
        self.slider_wb.blockSignals(False)
        self.slider_tint.blockSignals(False)

        # Keep the manual tint offset coherent with the newly loaded settings
        if self.chk_wb_link.isChecked():
            self._tint_manual_offset = a.tint - self._coupled_tint(a.wb_temp)

    # ===================================================================
    # RENDER WORKER CALLBACK
    # ===================================================================

    def _on_render_done(self, img_array, was_downscaled):
        """Receive a completed render from the background RenderWorker."""
        self.display_image(img_array, is_scrub=was_downscaled)
        # Keep the downscaled-preview cache warm with the latest look so a later
        # revisit is instant. Safe even mid-scrub: the worker drops post-switch
        # renders (epoch), so this only ever fires for the current image, and
        # the key captures the live settings used to produce this frame.
        if was_downscaled and self.image_files:
            file_path = str(self.image_files[self.current_index])
            self.preview_cache[file_path] = (self._preview_key(), img_array)
        if not was_downscaled and self._render_needs_commit:
            self._render_needs_commit = False
            self.update_current_thumbnail(img_array)
            self.update_mode_label()
            # save_current_settings is called synchronously at slider release —
            # see on_*_released / reset_*_slider — so persistence survives an
            # image switch that invalidates this render.

    # ===================================================================
    # SLIDER HANDLERS
    # ===================================================================

    def on_exposure_slider_moved(self, value):
        ev = value / 10.0
        self.label_exposure.setText(f"{ev:.1f} EV")
        self.processor.adjustments.exposure_ev = ev
        self._render_worker.request(downscale=True)

    def on_exposure_released(self):
        self.save_current_settings()
        self._render_needs_commit = True
        self._render_worker.request(downscale=False)

    def _on_wb_link_toggled(self, checked):
        self.save_current_settings()

    def _coupled_tint(self, wb_offset):
        """Coupled tint value for a given WB offset from neutral (5600K).
        Linear ±6: 0 → 0,  -2000 → +6,  +2000 → -6.
        Returns tint in actual units (same as processor.adjustments.tint).
        """
        return wb_offset / 2000.0 * -6.0

    def _apply_wb_tint_link(self, wb_value):
        """When the link is active: compute coupled tint + manual offset, update
        the tint slider/label without triggering on_tint_slider_moved, and update
        the processor setting.  Returns the new tint value."""
        coupled = self._coupled_tint(wb_value)
        new_tint = max(-10.0, min(10.0, coupled + self._tint_manual_offset))
        self.processor.adjustments.tint = new_tint
        self.slider_tint.blockSignals(True)
        self.slider_tint.setValue(int(round(new_tint * 5)))
        self.label_tint.setText(f"{int(round(new_tint * 5)):+d}")
        self.slider_tint.blockSignals(False)
        return new_tint

    def on_wb_slider_moved(self, value):
        temp_absolute = 5600 + value
        self.label_wb.setText(f"{temp_absolute} K")

        if self.chk_wb_link.isChecked():
            self._apply_wb_tint_link(value)
        self.processor.adjustments.wb_temp = value
        self._render_worker.request(downscale=True)

    def on_wb_released(self):
        self.save_current_settings()
        self._render_needs_commit = True
        self._render_worker.request(downscale=False)

    def on_tint_slider_moved(self, value):
        tint = value / 5.0
        self.label_tint.setText(f"{value:+d}")
        if self.chk_wb_link.isChecked():
            self._tint_manual_offset = tint - self._coupled_tint(self.slider_wb.value())
        self.processor.adjustments.tint = tint
        self._render_worker.request(downscale=True)

    def on_tint_released(self):
        self.save_current_settings()
        self._render_needs_commit = True
        self._render_worker.request(downscale=False)

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
        self.processor.adjustments = ImageAdjustments(active_vibe_id=self.current_vibe_id())
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
            base = export_basename(file_path)
            candidates = ["_clean.dng", "_edit.jpg"]
            candidates += [f"_{s}.jpg" for s in VIBE_EXPORT_SUFFIX.values()]
            for suffix in candidates:
                if os.path.exists(os.path.join(self.output_dir, base + suffix)):
                    return True
        except Exception:
            log.debug("export-exists check failed for %s", file_path, exc_info=True)
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

        if hasattr(self, '_render_worker'):
            self._render_worker.invalidate()
        self._render_needs_commit = False

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
                self.processor.set_settings(self.settings_clipboard)
                img_array = self.processor.render_preview()
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
            self.processor.set_settings(self.settings_clipboard)
            img_array = self.processor.render_preview()
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
            # Session-only override. The default lives in QSettings under
            # "default_export_dir" and is changed from the advanced panel.
            self.output_dir = directory
            self.label_output.setText(directory)
            self.label_output.setToolTip(directory)

    def _update_dng_button_state(self):
        """Enable or disable the DNG export pill based on the current file's type.

        DNG export uses Flashback-specific color science and is only meaningful
        for files shot on a Flashback camera. For all other raws the button is
        visible but greyed out, and the mode is silently redirected to JPEG.
        """
        if not hasattr(self, 'btn_export_dng'):
            return
        file_path = str(self.image_files[self.current_index]) if self.image_files else None
        is_flashback = self._file_is_flashback.get(file_path, False) if file_path else False

        self.btn_export_dng.setEnabled(is_flashback)
        if is_flashback:
            self.btn_export_dng.setToolTip("Clean RAW DNG for Camera Raw / Lightroom")
        else:
            self.btn_export_dng.setToolTip("DNG export is only available for Flashback camera files")
            if self.export_mode == 'dng':
                self.set_export_mode('jpeg')

    def set_export_mode(self, mode):
        """Select JPEG / DNG export; sync pills."""
        if mode in (True, 'tiff'):  # legacy: redirect old TIFF mode to JPEG
            mode = 'jpeg'
        elif mode is False:
            mode = 'jpeg'
        self.export_mode = mode
        self.btn_export_jpeg.setChecked(mode == 'jpeg')
        self.btn_export_dng.setChecked(mode == 'dng')
        self.btn_export_jpeg.setStyleSheet(format_pill_qss(mode == 'jpeg'))
        self.btn_export_dng.setStyleSheet(format_pill_qss(mode == 'dng'))
        if hasattr(self, "btn_process_all") and hasattr(self, "thumbnail_strip"):
            self.update_process_button_text()

    def process_all_images(self):
        if not self.image_files:
            return

        if hasattr(self, '_render_worker'):
            self._render_worker.invalidate()
        self._render_needs_commit = False

        selected_indices = self.thumbnail_strip.get_process_selected_indices()
        if selected_indices:
            indices_to_process = sorted(selected_indices)
        else:
            indices_to_process = list(range(len(self.image_files)))

        # Low disk space: inline warning in the status bar, no modal.
        mb_per_image = {'jpeg': 5, 'dng': 17}.get(self.export_mode, 5)
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
        skip_count = 0
        total = len(indices_to_process)

        for i, idx in enumerate(indices_to_process):
            file_path = str(self.image_files[idx])

            try:
                self.progress_bar.setValue(i)
                self.btn_process_all.setText(f"Processing {i + 1} / {total}")
                self.mode_label.setText(f"Processing {i+1}/{total}...")
                QApplication.processEvents()

                if self.export_mode == 'dng':
                    is_flashback = self._file_is_flashback.get(file_path)
                    if is_flashback is None:
                        from core.processor import _read_dng_exif
                        is_flashback, _ = _read_dng_exif(file_path)
                        self._file_is_flashback[file_path] = is_flashback
                    if not is_flashback:
                        skip_count += 1
                        continue

                if self.export_mode != 'dng':
                    if file_path in self.image_cache:
                        self.processor.intermediate_acescg = self.image_cache[file_path].copy()
                        self.processor.current_file = file_path
                    else:
                        self.processor.load_image(file_path)
                        self.image_cache[file_path] = self.processor.intermediate_acescg.copy()

                    if file_path in self.image_settings:
                        self.processor.set_settings(self.image_settings[file_path])

                base_name = export_basename(file_path)
                if self.export_mode == 'dng':
                    output_path = os.path.join(self.output_dir, f"{base_name}_clean.dng")
                else:
                    # Per-file LUT (V1 negatives get the V1-tuned variant).
                    self._apply_effective_lut(file_path)
                    vibe_id = self.processor.adjustments.active_vibe_id
                    suffix = VIBE_EXPORT_SUFFIX.get(vibe_id, 'edit')
                    output_path = os.path.join(self.output_dir, f"{base_name}_{suffix}.jpg")

                if self.export_mode == 'dng':
                    from core.dng_export import export_dng
                    thumb = None
                    strip_pixmap = None
                    if 0 <= idx < len(self.thumbnail_strip.thumbnails):
                        strip_pixmap = self.thumbnail_strip.thumbnails[idx].pixmap
                    if strip_pixmap and not strip_pixmap.isNull():
                        img = strip_pixmap.toImage().convertToFormat(QImage.Format_RGB888)
                        w, h = img.width(), img.height()
                        stride = img.bytesPerLine()
                        arr = np.frombuffer(img.bits(), dtype=np.uint8).reshape((h, stride))[:, :w * 3].reshape((h, w, 3)).copy()
                        tw, th = 512, max(1, int(h * 512 / w))
                        thumb = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_LINEAR)
                    if thumb is None:
                        import rawpy
                        with rawpy.imread(file_path) as raw:
                            thumb = raw.postprocess(
                                half_size=True, use_camera_wb=True,
                                no_auto_bright=True, output_bps=8,
                            )
                        tw = 512
                        th = max(1, int(thumb.shape[0] * tw / thumb.shape[1]))
                        thumb = cv2.resize(thumb, (tw, th), interpolation=cv2.INTER_LINEAR)
                    ok = export_dng(file_path, output_path, thumb, self.current_vibe.dng_profile_name)
                elif export_image(self.processor, output_path):
                    ok = True
                else:
                    ok = False

                if ok:
                    success_count += 1
                    self.thumbnail_strip.set_processed(idx, True)

            except Exception as e:
                log.error("Error processing %s: %s", file_path, e)
                traceback.print_exc()

            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

        self.progress_bar.setVisible(False)
        self.btn_process_all.setEnabled(True)
        self.load_current_image()
        self.update_mode_label()

        processed_total = total - skip_count
        if skip_count > 0:
            self.mode_label.setText(
                f"✓ {success_count} processed · {skip_count} skipped (non-Flashback, DNG only)"
            )
            self.mode_label.setStyleSheet(f"color: {C['text_dim']};")
            QTimer.singleShot(4000, self.update_mode_label)

        if success_count == processed_total and processed_total > 0:
            self._set_process_button_done(success_count)
        else:
            self.update_process_button_text()

    # ===================================================================
    # DEBUG / REFRESH
    # ===================================================================

    def refresh_from_debug(self):
        log.info("Refreshing...")
        if self.processor and self.processor.intermediate_acescg is not None:
            img_array = self.processor.render_preview()
            self.display_image(img_array)
            self.update_current_thumbnail(img_array)

    def reload_current_image(self):
        if not self.image_files:
            return
        file_path = str(self.image_files[self.current_index])
        if file_path in self.image_cache:
            del self.image_cache[file_path]
        self.preview_cache.pop(file_path, None)
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
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.image_files:
                self.remove_current_from_project()
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
                    log.warning("[native_chrome] apply failed: %s", e)

            QTimer.singleShot(0, _do_apply)

    def closeEvent(self, event):
        # Join every background QThread before teardown — any still running when
        # its Python owner is destroyed aborts the process ("QThread: Destroyed
        # while thread is still running"). Slow V1 thumbnail passes make this
        # easy to hit on close.
        self._stop_vibe_refresh_worker()
        self._stop_thumbnail_workers()
        self._render_worker.stop()
        self._render_worker.wait()
        super().closeEvent(event)

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

    def _negatives_from_zip(self, zip_path):
        """Extract a Flashback V1 roll zip to paired negative files under the
        output dir and return the raw paths (sorted). Empty list on failure."""
        from core.v1_negative import extract_negatives_from_zip, roll_capture_date
        from datetime import datetime
        try:
            # Mirror V2's date-foldered import layout: <output>/<date>/_v1_imports/<roll>.
            # The date comes from the negatives' own timestamps inside the zip
            # (old rolls can be exported anytime), falling back to import time.
            roll_dt = roll_capture_date(zip_path) or datetime.now()
            dest = (Path(self.output_dir) / date_folder_name(roll_dt)
                    / '_v1_imports' / Path(zip_path).stem
                    if self.output_dir else None)
            raws = extract_negatives_from_zip(zip_path, dest)
            if not raws:
                log.warning("[editor] no V1 negatives found in zip: %s", zip_path)
            return raws
        except Exception:
            log.exception("[editor] failed to read V1 roll zip: %s", zip_path)
            return []

    def _images_from_folder(self, folder):
        """Collect loadable images from an already-imported folder: supported
        raws plus V1 negatives (extensionless raw + .json sidecar). Loaded in
        place — no copy, so re-dragging a folder never overwrites it.
        Non-recursive; sorted by name."""
        found = []
        try:
            entries = sorted(Path(folder).iterdir(), key=lambda x: x.name.lower())
        except OSError:
            return found
        for entry in entries:
            if not entry.is_file() or entry.name.lower().endswith('.json'):
                continue  # .json is a V1 sidecar, picked up with its raw
            if entry.name.lower().endswith(self.SUPPORTED_EXTENSIONS) \
                    or is_v1_negative(str(entry)):
                found.append(entry)
        if not found:
            log.warning("[editor] no loadable images in folder: %s", folder)
        return found

    def _resolve_input_paths(self, paths):
        """Expand dropped/opened inputs into loadable image paths:
          - .zip rolls   -> extracted V1 negatives (idempotent; existing reused)
          - directories  -> the supported raws / V1 negatives they contain
          - everything else passes through unchanged."""
        resolved = []
        for p in paths:
            sp = str(p)
            if sp.lower().endswith('.zip'):
                resolved.extend(self._negatives_from_zip(sp))
            elif os.path.isdir(sp):
                resolved.extend(self._images_from_folder(sp))
            else:
                resolved.append(Path(p))
        return resolved

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and (
                        url.toLocalFile().lower().endswith(self.SUPPORTED_EXTENSIONS + ('.zip',))
                        or os.path.isdir(url.toLocalFile())):
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
        dropped = []
        for url in urls:
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(self.SUPPORTED_EXTENSIONS + ('.zip',)) \
                        or os.path.isdir(file_path):
                    dropped.append(file_path)
        image_files = self._resolve_input_paths(dropped)

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
