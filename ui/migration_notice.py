"""
Post-migration summary dialog.

Shown once on first launch after the 1.5.0 schema migration runs. The
dialog is non-blocking (a Qt.Tool window the user can dismiss) and
folds the per-vibe detail into a collapsible text area so the top-level
message stays a one-paragraph headline.

Dismissal persists via core.vibe_state.mark_migration_acknowledged().
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit,
)

from core import vibe_state


def _format_details(report) -> str:
    """Render the MigrationReport into a readable multi-section text body."""
    lines = []
    lines.append(f"Source file: {report.legacy_file}")
    lines.append(f"Vibes migrated: {len(report.migrated_vibe_ids)}")
    if report.migrated_vibe_ids:
        lines.append("  " + ", ".join(sorted(report.migrated_vibe_ids)))

    if report.rescaled_fields:
        lines.append("")
        lines.append(f"Rescaled to new units ({len(report.rescaled_fields)}):")
        for vibe_id, legacy_name, new_name in report.rescaled_fields:
            lines.append(f"  {vibe_id}: {legacy_name} → {new_name}")

    if report.reset_fields:
        lines.append("")
        lines.append(f"Reset to factory defaults ({len(report.reset_fields)}):")
        for vibe_id, new_name, reason in report.reset_fields:
            lines.append(f"  {vibe_id}: {new_name} — {reason}")

    if report.custom_luts_reset:
        lines.append("")
        lines.append(f"Custom LUTs reset ({len(report.custom_luts_reset)}):")
        lines.append("  Pre-1.5 .cube files were built against the old colour")
        lines.append("  pipeline and would look ~2 stops over + colour-shifted")
        lines.append("  under the new ACEScg pipeline. Files are untouched on")
        lines.append("  disk; re-import once you have regenerated them.")
        for vibe_id, legacy_path in report.custom_luts_reset:
            lines.append(f"    {vibe_id}: {legacy_path}")

    return "\n".join(lines)


class MigrationNoticeDialog(QDialog):
    """Single-shot summary shown after a pre-1.5 → 1.5 migration.

    Dismissal (close, OK, escape) calls mark_migration_acknowledged() so
    the dialog never reappears for the same migration.
    """

    def __init__(self, report, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Settings migrated from a previous version")
        self.setMinimumWidth(560)
        self._acknowledged = False

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        n_vibes = len(report.migrated_vibe_ids)
        n_rescaled = len(report.rescaled_fields)
        n_reset = len(report.reset_fields)
        n_lut_reset = len(report.custom_luts_reset)

        headline = QLabel(
            f"<b>{n_vibes} vibe{'s' if n_vibes != 1 else ''} migrated</b> "
            f"from a pre-1.5 install. "
            f"{n_rescaled} parameter{'s' if n_rescaled != 1 else ''} were rescaled "
            f"to the new unit system; "
            f"{n_reset} were reset because the underlying effect changed."
        )
        headline.setWordWrap(True)
        layout.addWidget(headline)

        if n_lut_reset:
            lut_warning = QLabel(
                f"<b>{n_lut_reset} custom LUT{'s' if n_lut_reset != 1 else ''} "
                f"reset.</b> Pre-1.5 .cube files would look "
                f"~2 stops overexposed and colour-shifted under the new colour "
                f"pipeline. Your original files are unchanged on disk — see "
                f"details below."
            )
            lut_warning.setWordWrap(True)
            layout.addWidget(lut_warning)

        sub = QLabel(
            "Your previous settings file is preserved unchanged, so an "
            "older version of the app remains usable on this machine."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #888;")
        layout.addWidget(sub)

        self.details = QPlainTextEdit(_format_details(report))
        self.details.setReadOnly(True)
        self.details.setVisible(False)
        layout.addWidget(self.details, 1)

        btn_row = QHBoxLayout()
        self.btn_details = QPushButton("Show details")
        self.btn_details.clicked.connect(self._toggle_details)
        btn_row.addWidget(self.btn_details)
        btn_row.addStretch()
        self.btn_ok = QPushButton("OK")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_ok)
        layout.addLayout(btn_row)

    def _toggle_details(self):
        showing = not self.details.isVisible()
        self.details.setVisible(showing)
        self.btn_details.setText("Hide details" if showing else "Show details")
        if showing:
            self.resize(self.width(), max(self.height(), 520))

    def _acknowledge_once(self):
        if not self._acknowledged:
            vibe_state.mark_migration_acknowledged()
            self._acknowledged = True

    def accept(self):
        self._acknowledge_once()
        super().accept()

    def reject(self):
        self._acknowledge_once()
        super().reject()

    def closeEvent(self, event):
        self._acknowledge_once()
        super().closeEvent(event)
