"""
Flashback One35 — entry point.
"""
import logging
import sys
import platform

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QSurfaceFormat, QPalette, QColor

from ui.editor import FlashbackEditor
from _version import __version__


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
            bundle_info['CFBundleName'] = 'Flashback One35 v2'
        except Exception:
            pass  # pyobjc not available — bundled app uses Info.plist instead

    app = QApplication(sys.argv)
    app.setApplicationName("Flashback One35 v2")
    app.setOrganizationName("Flashback")

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
    window.setWindowTitle(f"Flashback One35 v2 ({__version__})")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
