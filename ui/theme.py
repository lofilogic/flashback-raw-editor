"""Design tokens for the LoFi Logic editor UI — one place for colors, fonts,
spacing, and QSS snippets so the editor and widget modules stay consistent.

The active palette lives in ``C`` (mutable dict). Switch themes at runtime via
``set_theme("light" | "dark")``; registered listeners fire afterwards so the
editor can re-apply stylesheets and force repaints without tearing down the UI.
"""

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPixmap

from core import resource_path


# ── Palettes ────────────────────────────────────────────────────────────
DARK_PALETTE = {
    "bg_window":      "#17161a",
    "bg_toolbar":     "#17161a",
    "bg_rail":        "#1b1a1e",
    "bg_strip":       "#131214",
    "bg_input":       "rgba(255, 255, 255, 0.03)",
    "bg_input_hover": "rgba(255, 255, 255, 0.05)",
    "bg_input_active":"rgba(255, 255, 255, 0.08)",
    "border_soft":    "rgba(255, 255, 255, 0.06)",
    "border_input":   "rgba(255, 255, 255, 0.08)",
    "border_active":  "rgba(255, 255, 255, 0.15)",
    "text_primary":   "#e6e4de",
    "text_secondary": "rgba(230, 228, 222, 0.7)",
    "text_label":     "rgba(230, 228, 222, 0.55)",
    "text_dim":       "rgba(230, 228, 222, 0.45)",
    "text_disabled":  "rgba(230, 228, 222, 0.25)",
    "accent":         "#FF8A35",
    "accent_hover":   "#ffa05c",
    "accent_press":   "#e67a25",
    "accent_soft":    "rgba(255, 138, 53, 0.15)",
    "processed":      "#6bb56a",
    "paste_marker":   "#ffffff",
}

LIGHT_PALETTE = {
    "bg_window":       "#f4ede1",
    "bg_toolbar":      "#e0d6c2",
    "bg_rail":         "#ebe3d3",
    "bg_strip":        "#e0d6c2",
    "bg_input":        "rgba(42, 38, 32, 0.04)",
    "bg_input_hover":  "rgba(42, 38, 32, 0.06)",
    "bg_input_active": "rgba(42, 38, 32, 0.10)",
    "border_soft":     "rgba(42, 38, 32, 0.08)",
    "border_input":    "rgba(42, 38, 32, 0.10)",
    "border_active":   "rgba(42, 38, 32, 0.18)",
    "text_primary":    "#2a2620",
    "text_secondary":  "rgba(42, 38, 32, 0.70)",
    "text_label":      "rgba(42, 38, 32, 0.55)",
    "text_dim":        "rgba(42, 38, 32, 0.45)",
    "text_disabled":   "rgba(42, 38, 32, 0.25)",
    "accent":          "#d08a4a",
    "accent_hover":    "#e09a5a",
    "accent_press":    "#b87a3e",
    "accent_soft":     "rgba(208, 138, 74, 0.18)",
    "processed":       "#4d9a4c",
    "paste_marker":    "#2a2620",
}

PALETTES = {"light": LIGHT_PALETTE, "dark": DARK_PALETTE}
_active_name = "light"

# ``C`` is the live palette — modules import it once and continue to read the
# same dict after a theme change because we mutate in place.
C: dict = {}
C.update(LIGHT_PALETTE)


_listeners: list = []


def current_theme() -> str:
    return _active_name


def set_theme(name: str) -> None:
    """Swap the active palette in-place and fire listeners."""
    global _active_name
    if name not in PALETTES:
        return
    _active_name = name
    C.clear()
    C.update(PALETTES[name])
    for cb in list(_listeners):
        try:
            cb()
        except Exception:
            pass


def toggle_theme() -> str:
    set_theme("dark" if _active_name == "light" else "light")
    return _active_name


def register_theme_listener(callback) -> None:
    if callback not in _listeners:
        _listeners.append(callback)


def unregister_theme_listener(callback) -> None:
    if callback in _listeners:
        _listeners.remove(callback)


def qcolor(token: str) -> QColor:
    """Convert a token (or raw hex/rgba string) to QColor."""
    val = C.get(token, token)
    if val.startswith("rgba"):
        nums = val[val.index("(") + 1 : val.index(")")].split(",")
        r, g, b = (int(n) for n in nums[:3])
        a = int(float(nums[3]) * 255)
        return QColor(r, g, b, a)
    return QColor(val)


# ── Fonts ───────────────────────────────────────────────────────────────
UI_FONT = "Inter"
MONO_FONT = "JetBrains Mono"


def load_app_fonts() -> None:
    """Register bundled Inter + JetBrains Mono with QFontDatabase."""
    for fname in (
        "Inter-Regular.ttf",
        "Inter-Medium.ttf",
        "Inter-SemiBold.ttf",
        "JetBrainsMono-Medium.ttf",
    ):
        QFontDatabase.addApplicationFont(resource_path(f"assets/fonts/{fname}"))


def ui_font(size: int = 11, weight: QFont.Weight = QFont.Medium) -> QFont:
    f = QFont(UI_FONT, size)
    f.setWeight(weight)
    return f


def mono_font(size: int = 10, weight: QFont.Weight = QFont.Medium) -> QFont:
    f = QFont(MONO_FONT, size)
    f.setWeight(weight)
    return f


# ── Icons ───────────────────────────────────────────────────────────────
def svg_icon(rel_path: str, color_token: str = "text_label", size: int = 16) -> QIcon:
    """Load an SVG from assets, tint it to a palette color, and return a QIcon."""
    from PySide6.QtSvg import QSvgRenderer

    abs_path = resource_path(rel_path)
    try:
        with open(abs_path, "rb") as f:
            data = f.read()
    except OSError:
        return QIcon()

    color = qcolor(color_token)
    hex_color = color.name(QColor.HexRgb)
    data = data.replace(b"currentColor", hex_color.encode("ascii"))

    scale = 2
    pm = QPixmap(size * scale, size * scale)
    pm.fill(Qt.transparent)
    renderer = QSvgRenderer(QByteArray(data))
    painter = QPainter(pm)
    renderer.render(painter)
    painter.end()
    pm.setDevicePixelRatio(scale)
    return QIcon(pm)


# ── Reusable QSS snippets ───────────────────────────────────────────────
def icon_btn_qss(size: int = 28, radius: int = 4) -> str:
    return f"""
        QPushButton {{
            background: transparent;
            border: none;
            border-radius: {radius}px;
            color: {C['text_label']};
            padding: 0;
            min-width: {size}px;
            max-width: {size}px;
            min-height: {size}px;
            max-height: {size}px;
        }}
        QPushButton:hover {{
            background: {C['bg_input_hover']};
            color: {C['text_primary']};
        }}
        QPushButton:pressed {{
            background: {C['bg_input_active']};
        }}
        QPushButton:disabled {{
            color: {C['text_disabled']};
        }}
        QPushButton:checked {{
            background: {C['bg_input_active']};
            color: {C['text_primary']};
        }}
    """


def section_title_qss() -> str:
    return f"""
        QLabel {{
            color: {C['text_dim']};
            font-family: "{UI_FONT}";
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 1.2px;
            text-transform: uppercase;
        }}
    """


def section_reset_link_qss() -> str:
    return f"""
        QPushButton {{
            background: transparent;
            border: none;
            color: {C['text_dim']};
            font-family: "{UI_FONT}";
            font-size: 10px;
            font-weight: 500;
            padding: 0;
        }}
        QPushButton:hover {{ color: {C['text_primary']}; }}
    """


def process_btn_qss() -> str:
    # Button text colour for readability on the accent fill: near-black on the
    # dark theme's brighter orange, white on the light theme's muted orange.
    btn_text = "#1a1410" if current_theme() == "dark" else "#ffffff"
    return f"""
        QPushButton {{
            background: {C['accent']};
            color: {btn_text};
            border: none;
            border-radius: 3px;
            font-family: "{UI_FONT}";
            font-size: 12px;
            font-weight: 600;
            padding: 10px 12px;
        }}
        QPushButton:hover {{ background: {C['accent_hover']}; }}
        QPushButton:pressed {{ background: {C['accent_press']}; }}
        QPushButton:disabled {{
            background: {C['bg_input']};
            color: {C['text_disabled']};
        }}
    """


def format_pill_qss(active: bool) -> str:
    if active:
        border = C['border_active']
        bg = C['bg_input_active']
        color = C['text_primary']
    else:
        border = C['border_input']
        bg = C['bg_input']
        color = C['text_label']
    return f"""
        QPushButton {{
            background: {bg};
            border: 1px solid {border};
            border-radius: 3px;
            color: {color};
            font-family: "{UI_FONT}";
            font-size: 11px;
            font-weight: 500;
            padding: 6px 8px;
        }}
        QPushButton:hover {{ color: {C['text_primary']}; }}
    """
