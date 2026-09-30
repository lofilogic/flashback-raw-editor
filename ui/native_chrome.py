"""Native title bar.

macOS: transparent title bar, content up to the top edge.
Windows: dark title bar in dark mode. Linux: nothing.
"""

from __future__ import annotations

import ctypes
import platform

# Space for the traffic lights, for toolbar padding.
MAC_TRAFFIC_LIGHT_WIDTH = 76

_applied_windows: set[int] = set()


def apply(qwindow, theme_name: str = "light") -> None:
    """Style the native title bar for ``qwindow`` to match ``theme_name``."""
    sysname = platform.system()
    if sysname == "Darwin":
        _apply_mac(qwindow, theme_name)
    elif sysname == "Windows":
        _apply_windows(qwindow, theme_name)


# ── macOS ──────────────────────────────────────────────────────────────────
def _apply_mac(qwindow, theme_name: str) -> None:
    try:
        from AppKit import (
            NSWindowStyleMaskFullSizeContentView,
            NSWindowTitleHidden, NSAppearance, NSColor,
        )
        import objc  # noqa: F401
    except ImportError:
        return

    ns_window = _ns_window_for(qwindow)
    if ns_window is None:
        return

    ns_window.setStyleMask_(
        ns_window.styleMask() | NSWindowStyleMaskFullSizeContentView
    )
    ns_window.setTitlebarAppearsTransparent_(True)
    ns_window.setTitleVisibility_(NSWindowTitleHidden)
    # Not movableByWindowBackground: dragging sliders would move the window.

    # Hide the separator below the title bar.
    try:
        from AppKit import NSTitlebarSeparatorStyleNone
        ns_window.setTitlebarSeparatorStyle_(NSTitlebarSeparatorStyleNone)
    except Exception:
        pass

    from ui.theme import C, qcolor as _qcolor
    bg_qc = _qcolor("bg_toolbar")
    ns_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(
        bg_qc.redF(), bg_qc.greenF(), bg_qc.blueF(), 1.0
    )
    ns_window.setBackgroundColor_(ns_color)

    name = "NSAppearanceNameDarkAqua" if theme_name == "dark" else "NSAppearanceNameAqua"
    appearance = NSAppearance.appearanceNamed_(name)
    if appearance is not None:
        ns_window.setAppearance_(appearance)


def _ns_window_for(qwindow):
    try:
        import objc
        ns_view_addr = int(qwindow.winId())
        if not ns_view_addr:
            return None
        return objc.objc_object(c_void_p=ns_view_addr).window()
    except Exception:
        return None


# ── Windows ────────────────────────────────────────────────────────────────
def _apply_windows(qwindow, theme_name: str) -> None:
    try:
        dwm = ctypes.WinDLL("dwmapi")
    except OSError:
        return

    hwnd = int(qwindow.winId())
    use_dark = ctypes.c_int(1 if theme_name == "dark" else 0)
    for attr in (20, 19):
        try:
            dwm.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(use_dark), ctypes.sizeof(use_dark)
            )
            break
        except OSError:
            continue
