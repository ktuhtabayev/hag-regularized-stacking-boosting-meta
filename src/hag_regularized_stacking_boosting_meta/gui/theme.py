"""Single source of truth for GUI colors, shared by the Qt stylesheet and matplotlib."""

from __future__ import annotations


THEME: dict[str, str] = {
    "window_bg": "#f5f6f8",
    "surface": "#ffffff",
    "surface_alt": "#f3f5f8",
    "border": "#d0d4da",
    "border_strong": "#b6bcc6",
    "text": "#1f2430",
    "text_muted": "#5b6472",
    "accent": "#2563eb",
    "accent_hover": "#1d4ed8",
    "accent_pressed": "#1e40af",
    "accent_soft": "#dbeafe",
    "header_bg": "#e8ebf0",
    "hover_bg": "#eef2f7",
    "pressed_bg": "#e2e8f0",
    "disabled_bg": "#e9ecf0",
    "disabled_text": "#9aa2ae",
    "figure_bg": "#ffffff",
    "axes_bg": "#ffffff",
    "plot_note": "#5b6472",
    "plot_grid_line": "#e8ebf0",
}

# Plot colors. Classes use categorical slots 1-2 (validated colorblind-safe on white);
# class overlap is a state, so it uses the reserved "critical" status color + a text label.
PLOT_COLORS: dict[str, str] = {
    "class_k1": "#2a78d6",
    "class_k2": "#eb6834",
    "series": "#2a78d6",
    "margin_band": "#f0efec",
    "overlap": "#d03b3b",
    "ink": THEME["text"],
    "ink_muted": THEME["text_muted"],
}


def build_stylesheet(theme: dict[str, str] = THEME) -> str:
    t = theme
    return f"""
QMainWindow,
QWidget {{
    background-color: {t["window_bg"]};
    color: {t["text"]};
    font-size: 13px;
}}

QLabel {{
    background: transparent;
}}

QLabel#predictionLabel {{
    background-color: {t["accent_soft"]};
    color: {t["text"]};
    border: 1px solid {t["accent"]};
    border-radius: 6px;
    padding: 5px 12px;
    font-weight: 600;
}}

QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QTextEdit {{
    background-color: {t["surface"]};
    color: {t["text"]};
    border: 1px solid {t["border"]};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {t["accent"]};
    selection-color: {t["surface"]};
}}

QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QTextEdit:focus {{
    border: 1px solid {t["accent"]};
}}

/* Read-only statistics (e.g. the organizer found by HAG) */
QLineEdit:read-only {{
    background-color: {t["surface_alt"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox QAbstractItemView {{
    background-color: {t["surface"]};
    border: 1px solid {t["border"]};
    selection-background-color: {t["accent_soft"]};
    selection-color: {t["text"]};
}}

QPushButton,
QToolButton {{
    background-color: {t["surface"]};
    color: {t["text"]};
    border: 1px solid {t["border"]};
    border-radius: 6px;
    padding: 6px 14px;
}}

QPushButton:hover,
QToolButton:hover {{
    background-color: {t["hover_bg"]};
    border-color: {t["border_strong"]};
}}

QPushButton:pressed,
QToolButton:pressed {{
    background-color: {t["pressed_bg"]};
}}

QPushButton:disabled,
QToolButton:disabled {{
    background-color: {t["disabled_bg"]};
    color: {t["disabled_text"]};
    border-color: {t["border"]};
}}

QPushButton#primaryButton {{
    background-color: {t["accent"]};
    color: {t["surface"]};
    border: 1px solid {t["accent"]};
    font-weight: 600;
}}

QPushButton#primaryButton:hover {{
    background-color: {t["accent_hover"]};
    border-color: {t["accent_hover"]};
}}

QPushButton#primaryButton:pressed {{
    background-color: {t["accent_pressed"]};
}}

QPushButton#primaryButton:disabled {{
    background-color: {t["disabled_bg"]};
    color: {t["disabled_text"]};
    border-color: {t["border"]};
}}

QTabWidget::pane {{
    background-color: {t["surface"]};
    border: 1px solid {t["border"]};
    border-radius: 6px;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {t["text_muted"]};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 7px 14px;
    margin-right: 2px;
}}

QTabBar::tab:hover {{
    color: {t["text"]};
}}

QTabBar::tab:selected {{
    color: {t["accent"]};
    border-bottom: 2px solid {t["accent"]};
    font-weight: 600;
}}

QTableView {{
    background-color: {t["surface"]};
    alternate-background-color: {t["surface_alt"]};
    color: {t["text"]};
    border: 1px solid {t["border"]};
    gridline-color: {t["border"]};
    selection-background-color: {t["accent"]};
    selection-color: {t["surface"]};
}}

QHeaderView::section {{
    background-color: {t["header_bg"]};
    color: {t["text"]};
    border: none;
    border-right: 1px solid {t["border"]};
    border-bottom: 1px solid {t["border"]};
    padding: 5px 8px;
    font-weight: 600;
}}

QTableCornerButton::section {{
    background-color: {t["header_bg"]};
    border: none;
    border-right: 1px solid {t["border"]};
    border-bottom: 1px solid {t["border"]};
}}

QProgressBar {{
    background-color: {t["surface_alt"]};
    border: 1px solid {t["border"]};
    border-radius: 6px;
    text-align: center;
    max-height: 14px;
}}

QProgressBar::chunk {{
    background-color: {t["accent"]};
    border-radius: 5px;
}}

QSlider::groove:horizontal {{
    background: {t["border"]};
    height: 4px;
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {t["accent"]};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {t["surface"]};
    border: 2px solid {t["accent"]};
    width: 12px;
    margin: -6px 0;
    border-radius: 8px;
}}

QScrollBar:vertical {{
    background: {t["window_bg"]};
    width: 12px;
    margin: 0;
}}

QScrollBar:horizontal {{
    background: {t["window_bg"]};
    height: 12px;
    margin: 0;
}}

QScrollBar::handle:vertical,
QScrollBar::handle:horizontal {{
    background: {t["border_strong"]};
    border-radius: 5px;
    min-height: 24px;
    min-width: 24px;
}}

QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover {{
    background: {t["text_muted"]};
}}

QScrollBar::add-line,
QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}

QMenu {{
    background-color: {t["surface"]};
    color: {t["text"]};
    border: 1px solid {t["border"]};
    padding: 4px;
}}

QMenu::item {{
    padding: 5px 22px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {t["accent_soft"]};
    color: {t["text"]};
}}
"""
