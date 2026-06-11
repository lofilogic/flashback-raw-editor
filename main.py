"""
LoFi Logic — entry point.
"""
import logging
import os
import shutil
import sys
import platform

from PySide6.QtCore import Qt, QEvent, QSettings, QStandardPaths
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QSurfaceFormat, QPalette, QColor

from ui.editor import FlashbackEditor
from _version import __version__

log = logging.getLogger(__name__)


class _LoFiApplication(QApplication):
    """QApplication that routes OS 'open document' events to the editor.

    macOS delivers a file-association double-click / "Open With" as a
    QFileOpenEvent (not via argv). On a cold launch that event can arrive before
    the window exists, so it's buffered and flushed once the editor registers.
    (Windows/Linux pass the path on argv instead — handled in main().)
    """

    def __init__(self, argv):
        super().__init__(argv)
        self._editor = None
        self._pending = []

    def event(self, e):
        if e.type() == QEvent.Type.FileOpen:
            path = e.file()
            if path:
                if self._editor is not None:
                    self._editor.open_os_path(path)
                else:
                    self._pending.append(path)
                return True
        return super().event(e)

    def register_editor(self, editor):
        self._editor = editor
        for p in self._pending:
            editor.open_os_path(p)
        self._pending.clear()

# Identity used by builds before the LoFi Logic rename. Kept only so a one-time
# migration can carry a beta user's settings + saved vibes across the rename.
_LEGACY_ORG, _LEGACY_APP = "Flashback", "Flashback One35 v2"
_LEGACY_SETTINGS = ("Flashback", "Editor")
ORG_NAME, APP_NAME = "LoFi Logic", "LoFi Logic"
SETTINGS_SCOPE = ("LoFi Logic", "Editor")


def _migrate_app_identity():
    """Best-effort one-time carry-over of pre-rename user data.

    Renaming the app/org name moves both the QSettings store and Qt's
    AppDataLocation (saved vibes). Copy the old data into the new locations once,
    only when the new ones are still empty, so an existing beta install keeps its
    settings and tuned vibes instead of silently resetting. Never clobbers data
    the user already created under the new name. Must run after the new
    application/organization names are set (so AppDataLocation resolves to the
    new dir) and before the editor reads anything.
    """
    # 1) QSettings (default folders, DNG profile, window state, last project).
    new_qs = QSettings(*SETTINGS_SCOPE)
    if not new_qs.allKeys():
        old_qs = QSettings(*_LEGACY_SETTINGS)
        keys = old_qs.allKeys()
        if keys:
            for k in keys:
                new_qs.setValue(k, old_qs.value(k))
            new_qs.sync()
            log.info("Migrated %d app setting(s) from the previous app name.", len(keys))

    # 2) AppDataLocation (saved vibes). Derive the old dir from the new one by
    #    swapping the org/app path segment — robust across platforms since Qt
    #    builds the path as <base>/<org>/<app>.
    new_dir = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    if not new_dir:
        return
    old_dir = new_dir.replace(os.path.join(ORG_NAME, APP_NAME),
                              os.path.join(_LEGACY_ORG, _LEGACY_APP))
    if old_dir == new_dir or not os.path.isdir(old_dir):
        return
    os.makedirs(new_dir, exist_ok=True)
    if any(f.startswith("vibe_state") for f in os.listdir(new_dir)):
        return  # user already has state under the new name — don't overwrite
    for name in os.listdir(old_dir):
        src, dst = os.path.join(old_dir, name), os.path.join(new_dir, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    log.info("Migrated saved vibes from the previous app-data directory.")


def main():
    """Main entry point."""
    # Plain-message format preserves the look of the existing log lines
    # (which already carry their own [module] prefixes and ✓ / ⚠ / ✗ glyphs).
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Allow fractional DPI scaling (e.g. 125%, 150%) — must be set before QApplication
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    if platform.system() == 'Darwin':
        # Force sRGB color space to prevent P3 display oversaturation
        fmt = QSurfaceFormat.defaultFormat()
        fmt.setColorSpace(QSurfaceFormat.ColorSpace.sRGBColorSpace)
        QSurfaceFormat.setDefaultFormat(fmt)

        # Patch CFBundleName so the macOS app menu shows the correct name
        # when running directly as `python main.py` (bundled builds use Info.plist).
        try:
            from Foundation import NSBundle
            bundle_info = NSBundle.mainBundle().infoDictionary()
            bundle_info['CFBundleName'] = 'LoFi Logic'
        except Exception:
            pass  # pyobjc not available — bundled app uses Info.plist instead

    app = _LoFiApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    _migrate_app_identity()

    # Fusion style with an explicit dark palette — ensures consistent appearance
    # regardless of the OS light/dark mode setting (important for Windows VMs).
    app.setStyle("Fusion")
    dark = QPalette()
    dark.setColor(QPalette.ColorRole.Window,          QColor(49,  49,  49))
    dark.setColor(QPalette.ColorRole.WindowText,      QColor(208, 208, 208))
    dark.setColor(QPalette.ColorRole.Base,            QColor(35,  35,  35))
    dark.setColor(QPalette.ColorRole.AlternateBase,   QColor(53,  53,  53))
    dark.setColor(QPalette.ColorRole.ToolTipBase,     QColor(49,  49,  49))
    dark.setColor(QPalette.ColorRole.ToolTipText,     QColor(208, 208, 208))
    dark.setColor(QPalette.ColorRole.Text,            QColor(208, 208, 208))
    dark.setColor(QPalette.ColorRole.Button,          QColor(61,  61,  61))
    dark.setColor(QPalette.ColorRole.ButtonText,      QColor(208, 208, 208))
    dark.setColor(QPalette.ColorRole.BrightText,      QColor(255, 255, 255))
    dark.setColor(QPalette.ColorRole.Link,            QColor(255, 138, 53))
    dark.setColor(QPalette.ColorRole.Highlight,       QColor(255, 138, 53))
    dark.setColor(QPalette.ColorRole.HighlightedText, QColor(30,  30,  30))
    dark.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(80, 80, 80))
    dark.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor(80, 80, 80))
    dark.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(80, 80, 80))
    app.setPalette(dark)

    window = FlashbackEditor()
    window.setWindowTitle(f"LoFi Logic ({__version__})")
    window.show()
    app.register_editor(window)

    # Windows/Linux hand a double-clicked / "Open with" file on the command line
    # (macOS uses the QFileOpenEvent path above). Open the first real path.
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            window.open_os_path(arg)
            break

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
