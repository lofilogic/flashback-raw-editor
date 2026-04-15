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
    # macOS: Force sRGB color space to prevent P3 display oversaturation
    if platform.system() == 'Darwin':
        fmt = QSurfaceFormat.defaultFormat()
        fmt.setColorSpace(QSurfaceFormat.ColorSpace.sRGBColorSpace)
        QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)

    # Fusion style provides a consistent dark theme across platforms
    app.setStyle("Fusion")

    window = FlashbackEditor()
    window.setWindowTitle(f"Flashback One35 v2 ({__version__})")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
