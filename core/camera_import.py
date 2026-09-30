"""Import DNGs from a connected camera into <root>/<YYYY-MM-DD>/_RAW/."""
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
    return dt.strftime('%Y-%m-%d')


def read_capture_date(source_path: Path) -> datetime:
    """Capture date from EXIF, else the file mtime."""
    try:
        with open(source_path, 'rb') as f:
            tags = exifread.process_file(f, details=False, stop_tag='EXIF DateTimeOriginal')
        for key in ('EXIF DateTimeOriginal', 'Image DateTime', 'EXIF DateTimeDigitized'):
            if key in tags:
                raw = str(tags[key]).strip()
                for fmt in ('%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S'):
                    try:
                        return datetime.strptime(raw, fmt)
                    except ValueError:
                        continue
    except Exception as exc:
        log.debug("EXIF read failed for %s: %s", source_path, exc)
    return datetime.fromtimestamp(source_path.stat().st_mtime)


def target_path_for(source: Path, output_root: Path) -> Path:
    """<output_root>/<YYYY-MM-DD>/_RAW/<source.name>"""
    dt = read_capture_date(source)
    folder = output_root / date_folder_name(dt) / '_RAW'
    return folder / source.name


def _embed_thumb_from_display(display_img):
    """120 px high thumbnail for the DNG."""
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
    """Load the source, write it as our DNG, or copy it if that fails.

    Returns the rendered image so the caller doesn't have to load it again.
    """
    source_str = str(source_path)
    target_str = str(target_path)
    img_display = processor.load_image(source_str)
    embed_thumb = _embed_thumb_from_display(img_display)
    os.makedirs(os.path.dirname(target_str), exist_ok=True)
    profile_name = processor.vibe.dng_profile_name
    if not export_dng(source_str, target_str, embed_thumb, profile_name):
        log.warning("DNG rewrite failed; falling back to copy: %s", source_str)
        shutil.copy2(source_str, target_str)
    return img_display


def plan_imports(sources, output_root: Path):
    """Returns (to_import, skipped): (source, target) pairs to import, and
    targets that already exist, in source order."""
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
