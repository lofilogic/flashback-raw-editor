"""Helpers for importing DNGs from a connected Flashback camera into
date-named subfolders of the app output directory."""
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import exifread

from .dng_export import export_dng

log = logging.getLogger(__name__)


def date_folder_name(dt: datetime) -> str:
    """ISO 8601 (YYYY-MM-DD) — unambiguous across regions and sorts
    chronologically when listed alphabetically."""
    return dt.strftime('%Y-%m-%d')


def read_capture_date(source_path: Path) -> datetime:
    """Best-effort capture date for a DNG. Falls back to file mtime."""
    try:
        with open(source_path, 'rb') as f:
            tags = exifread.process_file(f, details=False, stop_tag='EXIF DateTimeOriginal')
        for key in ('EXIF DateTimeOriginal', 'Image DateTime', 'EXIF DateTimeDigitized'):
            if key in tags:
                raw = str(tags[key]).strip()
                # Standard EXIF format: 'YYYY:MM:DD HH:MM:SS'
                for fmt in ('%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S'):
                    try:
                        return datetime.strptime(raw, fmt)
                    except ValueError:
                        continue
    except Exception as exc:
        log.debug("EXIF read failed for %s: %s", source_path, exc)
    return datetime.fromtimestamp(source_path.stat().st_mtime)


def target_path_for(source: Path, output_root: Path) -> Path:
    """Resolve <output_root>/<YYYY-MM-DD>/_RAW/<source.name> for a camera DNG."""
    dt = read_capture_date(source)
    folder = output_root / date_folder_name(dt) / '_RAW'
    return folder / source.name


def _embed_thumb_from_display(display_img):
    """Downscale a processor display image to a small RGB embed thumbnail."""
    if display_img is None or display_img.size == 0:
        return None
    h, w = display_img.shape[:2]
    if h <= 0 or w <= 0:
        return None
    target_h = 120
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(display_img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def export_camera_dng(source_path, target_path, processor):
    """Load source via `processor`, derive a small embedded thumbnail, and
    export the DNG to `target_path`.

    Falls back to a plain file copy if the DNG rewrite fails. Returns the
    display image the processor produced (or None) so callers can reuse it
    instead of reading the file a second time.
    """
    source_str = str(source_path)
    target_str = str(target_path)
    img_display = processor.load_image(source_str)
    embed_thumb = _embed_thumb_from_display(img_display)
    os.makedirs(os.path.dirname(target_str), exist_ok=True)
    if not export_dng(source_str, target_str, embed_thumb):
        log.warning("DNG rewrite failed; falling back to copy: %s", source_str)
        shutil.copy2(source_str, target_str)
    return img_display


def plan_imports(sources, output_root: Path):
    """Return (to_import, skipped).

    to_import: list[(source_path, target_path)] for files that need exporting.
    skipped: list[target_path] for files whose target already exists, in the
        order they appear among `sources` — useful when the caller wants to
        still surface those already-imported files in the thumbnail strip.
    """
    to_import = []
    skipped = []
    for src in sources:
        src = Path(src)
        tgt = target_path_for(src, Path(output_root))
        if tgt.exists():
            skipped.append(tgt)
        else:
            to_import.append((src, tgt))
    return to_import, skipped
