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
    """Passes macOS file-open events to the editor.

    These can arrive before the window exists, so they're buffered.
    Windows and Linux use argv instead.
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

# Names before the LoFi Logic rename, for migrating settings.
_LEGACY_ORG, _LEGACY_APP = "Flashback", "Flashback One35 v2"
_LEGACY_SETTINGS = ("Flashback", "Editor")
ORG_NAME, APP_NAME = "LoFi Logic", "LoFi Logic"
SETTINGS_SCOPE = ("LoFi Logic", "Editor")


def _migrate_app_identity():
    """Copy settings and saved vibes from the pre-rename locations, if the new
    ones are empty. Run after setting the app names, before the editor starts.
    """
    # QSettings
    new_qs = QSettings(*SETTINGS_SCOPE)
    if not new_qs.allKeys():
        old_qs = QSettings(*_LEGACY_SETTINGS)
        keys = old_qs.allKeys()
        if keys:
            for k in keys:
                new_qs.setValue(k, old_qs.value(k))
            new_qs.sync()
            log.info("Migrated %d app setting(s) from the previous app name.", len(keys))

    # Saved vibes. Qt builds the path as <base>/<org>/<app>.
    new_dir = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    if not new_dir:
        return
    old_dir = new_dir.replace(os.path.join(ORG_NAME, APP_NAME),
                              os.path.join(_LEGACY_ORG, _LEGACY_APP))
    if old_dir == new_dir or not os.path.isdir(old_dir):
        return
    os.makedirs(new_dir, exist_ok=True)
    if any(f.startswith("vibe_state") for f in os.listdir(new_dir)):
        return
    for name in os.listdir(old_dir):
        src, dst = os.path.join(old_dir, name), os.path.join(new_dir, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    log.info("Migrated saved vibes from the previous app-data directory.")


def main():
    # Messages carry their own [module] prefixes.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Fractional DPI scaling; must be set before QApplication
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    if platform.system() == 'Darwin':
        # sRGB, or P3 displays oversaturate
        fmt = QSurfaceFormat.defaultFormat()
        fmt.setColorSpace(QSurfaceFormat.ColorSpace.sRGBColorSpace)
        QSurfaceFormat.setDefaultFormat(fmt)

        # App menu name when run from source; builds use Info.plist.
        try:
            from Foundation import NSBundle
            bundle_info = NSBundle.mainBundle().infoDictionary()
            bundle_info['CFBundleName'] = 'LoFi Logic'
        except Exception:
            pass

    app = _LoFiApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    _migrate_app_identity()

    # Fusion with our own palette, independent of the OS theme.
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

    # Windows/Linux pass opened files on the command line.
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            window.open_os_path(arg)
            break

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
