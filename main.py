"""
Flashback One35 — entry point.
"""
import sys
import platform

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QSurfaceFormat

from ui.editor import FlashbackEditor
from _version import __version__


def main():
    """Main entry point."""
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

    # Fusion style provides a consistent dark theme across platforms
    app.setStyle("Fusion")

    window = FlashbackEditor()
    window.setWindowTitle(f"Flashback One35 v2 ({__version__})")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
