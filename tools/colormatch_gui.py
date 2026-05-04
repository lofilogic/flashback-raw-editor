#!/usr/bin/env python3
"""ColorMatch GUI — generate a Flashback look-match LUT from any camera raw.

Workflow (two steps in one window):
  Step 1  Pick a raw / DNG from your target camera → Generate HALD
            Adobe DNG Converter → linear + uncompressed DNG → HALD-injected
            Output: apply-look-to-me.dng (next to the source file).
  Step 2  Open apply-look-to-me.dng in Camera Raw / Lightroom with
            **manual 5000K WB, Tint 0**, apply your look, export 16-bit sRGB
            Gamma 2.4 TIFF. Pick that TIFF → Generate LUT.
            Output: <tif_name>_match.cube (next to the TIFF).

The Flashback side is baked in (tools/fb_hald_samples.npz). No FB DNG needed.

Run:  python tools/colormatch_gui.py
"""
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import core  # noqa: F401
sys.path.insert(0, str(HERE))

from PySide6 import QtCore, QtWidgets, QtGui

from inject_hald_dng import inject_hald
from build_lut_from_hald import build_lut_from_target_only
from dng_converter import (find_dng_converter, set_saved_path,
                            normalise_input_to_linear_dng, RAW_EXTENSIONS)

FB_SAMPLES = HERE / 'fb_hald_samples.npz'
HALD_OUTPUT_NAME = 'apply-look-to-me.dng'


# ─── Worker base class ─────────────────────────────────────────────────────

class Worker(QtCore.QObject):
    """Generic worker that runs `fn` on a background thread, emits log lines
    via `log` signal, and `done` (success: bool) when finished."""
    log = QtCore.Signal(str)
    done = QtCore.Signal(bool)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    @QtCore.Slot()
    def run(self):
        try:
            self._fn(self.log.emit)
            self.done.emit(True)
        except Exception:
            self.log.emit('\n✗ Failed:')
            self.log.emit(traceback.format_exc())
            self.done.emit(False)


# ─── Main window ───────────────────────────────────────────────────────────

class ColorMatchWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Flashback ColorMatch')
        self.resize(760, 560)

        self.dng_path = ''
        self.tif_path = ''
        self.meta_path = None  # set after Generate HALD

        self._thread = None
        self._worker = None

        self._build_ui()
        self._check_fb_samples()

    # ─── UI construction ────────────────────────────────────────────────

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        intro = QtWidgets.QLabel(
            'Generate a Flashback colour-match LUT from any camera raw.\n'
            '   Step 1 — pick a raw or DNG; Adobe DNG Converter normalises it\n'
            '             (skipped automatically if it\'s already a linear DNG),\n'
            '             then a HALD is injected. Output: apply-look-to-me.dng.\n'
            '   Step 2 — open apply-look-to-me.dng in your raw editor\n'
            '             and apply your look, export 16-bit sRGB TIFF,\n'
            '             then pick it here to build the LUT.'
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # Adobe DNG Converter status row
        cv_row = QtWidgets.QHBoxLayout()
        cv_row.addWidget(QtWidgets.QLabel('Adobe DNG Converter:'))
        self.lbl_converter = QtWidgets.QLabel('(searching…)')
        self.lbl_converter.setStyleSheet('color: #666;')
        cv_row.addWidget(self.lbl_converter, stretch=1)
        self.btn_locate_converter = QtWidgets.QPushButton('Locate…')
        self.btn_locate_converter.clicked.connect(self.locate_converter)
        cv_row.addWidget(self.btn_locate_converter)
        outer.addLayout(cv_row)

        # Step 1
        s1 = QtWidgets.QGroupBox('Step 1 — Pick raw / DNG → inject HALD')
        l1 = QtWidgets.QVBoxLayout(s1)
        row1 = QtWidgets.QHBoxLayout()
        self.btn_pick_dng = QtWidgets.QPushButton('Browse raw / DNG…')
        self.btn_pick_dng.clicked.connect(self.pick_dng)
        row1.addWidget(self.btn_pick_dng)
        self.lbl_dng = QtWidgets.QLabel('(no file selected)')
        self.lbl_dng.setStyleSheet('color: #666;')
        row1.addWidget(self.lbl_dng, stretch=1)
        l1.addLayout(row1)
        self.btn_inject = QtWidgets.QPushButton('Generate HALD')
        self.btn_inject.setEnabled(False)
        self.btn_inject.clicked.connect(self.run_inject)
        l1.addWidget(self.btn_inject, alignment=QtCore.Qt.AlignLeft)
        outer.addWidget(s1)

        # Step 2
        s2 = QtWidgets.QGroupBox('Step 2 — Build LUT from look-applied TIFF')
        l2 = QtWidgets.QVBoxLayout(s2)
        row2 = QtWidgets.QHBoxLayout()
        self.btn_pick_tif = QtWidgets.QPushButton('Browse TIFF…')
        self.btn_pick_tif.clicked.connect(self.pick_tif)
        row2.addWidget(self.btn_pick_tif)
        self.lbl_tif = QtWidgets.QLabel('(no file selected)')
        self.lbl_tif.setStyleSheet('color: #666;')
        row2.addWidget(self.lbl_tif, stretch=1)
        l2.addLayout(row2)
        self.btn_build = QtWidgets.QPushButton('Generate LUT')
        self.btn_build.setEnabled(False)
        self.btn_build.clicked.connect(self.run_build)
        l2.addWidget(self.btn_build, alignment=QtCore.Qt.AlignLeft)
        outer.addWidget(s2)

        # Log pane
        log_box = QtWidgets.QGroupBox('Log')
        lb = QtWidgets.QVBoxLayout(log_box)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        font = QtGui.QFont('Menlo', 11)
        self.log_view.setFont(font)
        lb.addWidget(self.log_view)
        outer.addWidget(log_box, stretch=1)

    # ─── Helpers ────────────────────────────────────────────────────────

    def _log(self, msg):
        self.log_view.appendPlainText(str(msg).rstrip())

    def _check_fb_samples(self):
        if not FB_SAMPLES.exists():
            self._log(f'⚠ Flashback samples missing: {FB_SAMPLES}')
            self._log('  Run tools/calibrate_fb_baseline.py once before using this tool.')
        else:
            self._log(f'Flashback samples loaded: {FB_SAMPLES.name}')
        self._refresh_converter_status()

    def _refresh_converter_status(self):
        path = find_dng_converter()
        if path:
            self.lbl_converter.setText(path)
            self.lbl_converter.setStyleSheet('color: #2a7;')
            self.btn_locate_converter.setText('Change…')
            self._log(f'Adobe DNG Converter: {path}')
        else:
            self.lbl_converter.setText('Not found — pre-converted linear DNGs only')
            self.lbl_converter.setStyleSheet('color: #b40;')
            self.btn_locate_converter.setText('Locate…')
            self._log('⚠ Adobe DNG Converter not found.')
            self._log('  Free download: https://helpx.adobe.com/camera-raw/digital-negative.html')
            self._log('  Or click Locate… if installed elsewhere.')
            self._log('  Without it, only LinearRaw + uncompressed DNGs can be processed.')

    def locate_converter(self):
        # Pick the executable. macOS: inside the .app bundle. Windows: .exe.
        if sys.platform == 'darwin':
            title = 'Pick Adobe DNG Converter'
            filt  = 'Executable (Adobe DNG Converter);;All files (*.*)'
            start = '/Applications/Adobe DNG Converter.app/Contents/MacOS'
        elif sys.platform.startswith('win'):
            title = 'Pick Adobe DNG Converter.exe'
            filt  = 'Adobe DNG Converter (Adobe DNG Converter.exe);;Executables (*.exe);;All files (*.*)'
            start = r'C:\Program Files\Adobe\Adobe DNG Converter'
        else:
            title = 'Pick Adobe DNG Converter executable'
            filt  = 'All files (*.*)'
            start = ''
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, start, filt)
        if not path:
            return
        set_saved_path(path)
        self._refresh_converter_status()

    def _set_busy(self, busy):
        for b in (self.btn_pick_dng, self.btn_pick_tif, self.btn_inject, self.btn_build):
            b.setEnabled(not busy)
        if not busy:
            self.btn_inject.setEnabled(bool(self.dng_path))
            self.btn_build.setEnabled(bool(self.tif_path))

    def _start_worker(self, fn, on_success=None):
        """Run `fn(log_emit)` on a Qt thread, log everything, call on_success on success."""
        self._set_busy(True)
        self._thread = QtCore.QThread(self)
        self._worker = Worker(fn)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log)

        def cleanup(success):
            self._thread.quit()
            self._thread.wait()
            self._set_busy(False)
            if success and on_success is not None:
                try:
                    on_success()
                except Exception:
                    self._log(traceback.format_exc())

        self._worker.done.connect(cleanup)
        self._thread.start()

    # ─── File pickers ───────────────────────────────────────────────────

    def pick_dng(self):
        # Build raw-extension filter (case-insensitive on macOS via *.ext + *.EXT)
        raw_globs = []
        for ext in RAW_EXTENSIONS:
            raw_globs.append(f'*{ext}')
            raw_globs.append(f'*{ext.upper()}')
        raw_filter = 'Camera raw / DNG (' + ' '.join(raw_globs) + ')'
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Pick a raw or DNG from your target camera', '',
            f'{raw_filter};;All files (*.*)',
        )
        if not path:
            return
        self.dng_path = path
        self.lbl_dng.setText(Path(path).name)
        self.btn_inject.setEnabled(True)
        self._log(f'Source raw: {path}')

    def pick_tif(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Pick look-applied TIFF', '',
            'TIFF (*.tif *.tiff *.TIF *.TIFF);;All files (*.*)',
        )
        if not path:
            return
        self.tif_path = path
        self.lbl_tif.setText(Path(path).name)
        self.btn_build.setEnabled(True)
        self._log(f'Look TIFF: {path}')

    # ─── Actions ────────────────────────────────────────────────────────

    def run_inject(self):
        if not self.dng_path:
            return
        out_dng = str(Path(self.dng_path).with_name(HALD_OUTPUT_NAME))
        self._log(f'\n=== Step 1: convert + HALD inject ===')

        captured_meta = {}

        def fn(log_emit):
            # 1. Normalise to linear + uncompressed DNG (skip if already that)
            with tempfile.TemporaryDirectory() as tmp:
                tentative = str(Path(tmp) / 'normalised.dng')
                normalised = normalise_input_to_linear_dng(
                    self.dng_path, tentative, log=log_emit)
                # 2. Inject HALD into the normalised DNG, save next to the source
                res = inject_hald(normalised, out_dng, log=log_emit)
                captured_meta['path'] = res['meta_path']

        def on_success():
            self.meta_path = captured_meta.get('path')
            self._log(f'\n→ Open {HALD_OUTPUT_NAME} in your raw editor:')
            self._log('  • WB: Custom, 5000 K, Tint 0')
            self._log('  • Apply your look (no spatial effects: no NR, no sharpening)')
            self._log('  • Export 16-bit sRGB Gamma 2.4 TIFF')
            self._log('Then pick that TIFF in Step 2.')

        self._start_worker(fn, on_success=on_success)

    def run_build(self):
        if not self.tif_path:
            return
        meta_path = self.meta_path
        if meta_path is None or not Path(meta_path).exists():
            tif_dir = Path(self.tif_path).parent
            candidates = list(tif_dir.glob('*.hald_meta.json'))
            if len(candidates) == 1:
                meta_path = str(candidates[0])
            elif len(candidates) > 1:
                meta_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
            else:
                QtWidgets.QMessageBox.critical(
                    self, 'Meta JSON not found',
                    f'No *.hald_meta.json next to:\n{self.tif_path}\n\n'
                    'Run Step 1 first (Generate HALD).'
                )
                return

        if not FB_SAMPLES.exists():
            QtWidgets.QMessageBox.critical(
                self, 'Missing FB samples',
                f'{FB_SAMPLES} not found.\n'
                'Run tools/calibrate_fb_baseline.py first.'
            )
            return

        out_lut = str(Path(self.tif_path).with_name(Path(self.tif_path).stem + '_match.cube'))
        self._log(f'\n=== Step 2: build LUT ===')
        self._log(f'  meta: {meta_path}')

        def fn(log_emit):
            build_lut_from_target_only(
                target_tif=self.tif_path,
                target_meta_path=meta_path,
                fb_samples_npz=str(FB_SAMPLES),
                output_lut_path=out_lut,
                log=log_emit,
            )

        def on_success():
            self._log(f'\n→ Load {Path(out_lut).name} in Flashback Editor.')

        self._start_worker(fn, on_success=on_success)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ColorMatchWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
