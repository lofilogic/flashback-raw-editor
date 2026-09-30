"""Flashback ONE35 V1 negatives.

The V1 can't write DNGs. Its "negative" export is a headerless 8-bit RGGB
mosaic (``<uuid>`` or ``<uuid>.raw``) plus a ``<uuid>.json`` with the
geometry. This develops it to the same linear ACEScg intermediate as the DNG
path:

    uint8 mosaic -> black subtract -> dither -> demosaic
      -> downscale to V2 size -> highlight recovery -> exposure trim
      -> ForwardMatrix -> ACEScg -> V2 WB match

The matrices come from tools/generate_matrices_v1.py.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# =============================================================================
# V1 SENSOR / DEVELOP CONSTANTS
# =============================================================================

# From the histograms of sample negatives. A lens-cap dark frame would pin it
# down exactly.
V1_BLACK_LEVEL = 3.0
V1_WHITE_LEVEL = 255.0

# Same long edge as a V2 develop, so pixel-sized effects match.
V1_TARGET_LONG_EDGE = 2072

# Greens are at [0,1] and [1,0]. OpenCV's RG code gives natural colour, BG
# comes out cyan. The profiling tool must use the same code.
V1_BAYER_CODE = cv2.COLOR_BayerRG2RGB
# Edge-aware for developing; the plain one is enough for chart patches.
V1_BAYER_CODE_EA = cv2.COLOR_BayerRG2RGB_EA

# TPDF dither in LSB, so 8-bit deep shadows don't band after the LUT. The
# sensor noise already dithers everything brighter.
V1_DITHER_LSB = 0.5

# Exposure trim (EV) before the matrix. Brightness is anchored by the chart
# exposure used for calibration, so a large trim means the chart should be
# reshot instead.
V1_EXPOSURE_EV = -1.2

# Same recovery as the V2 path. Runs before the exposure trim, which would
# push clipped values below the threshold.
V1_HIGHLIGHT_RECOVERY = True
V1_HIGHLIGHT_THRESHOLD = 0.97

# WB correction to match V2, in slider units, applied after the matrix like
# the sliders are. Doing it in ASN instead gets much weaker through the matrix.
# Negative tint is greener.
V1_WB_TEMP = -75.0
V1_TINT = -4.0

# ---- colour, from tools/generate_matrices_v1.py ----------------------------
# ASN: neutral response under D50 from the chart's grey patches.
# ForwardMatrix: (raw / ASN) -> XYZ_D50.
V1_ASN_D50 = np.array([0.876826, 1.0, 0.862259], dtype=np.float32)
V1_FORWARD_MATRIX = np.array([
    [ 0.752684,  0.212928, -0.001412],
    [-0.006849,  1.438282, -0.431433],
    [ 0.246380, -1.381695,  1.960215],
], dtype=np.float32)  # d50 fit_rms=0.0627 n=24 (reds OK; deep-blue luminance slightly low)


# =============================================================================
# FILE HANDLING
# =============================================================================

def _stem_paths(path: str):
    """Any of a negative's files -> (raw_path, json_path)."""
    p = Path(path)
    stem_dir = p.parent
    stem = p.stem if p.suffix in ('.raw', '.json') else p.name
    json_path = stem_dir / f"{stem}.json"
    raw_candidates = [stem_dir / f"{stem}.raw", stem_dir / stem]
    raw_path = next((c for c in raw_candidates if c.exists()), None)
    return raw_path, (json_path if json_path.exists() else None)


@lru_cache(maxsize=4096)
def is_v1_negative(path: str) -> bool:
    """True if there's a sidecar .json and the raw is exactly width*height
    bytes.

    Cached because thumbnails ask on every vibe change.
    extract_negatives_from_zip clears it."""
    raw_path, json_path = _stem_paths(path)
    if raw_path is None or json_path is None:
        return False
    try:
        meta = json.loads(json_path.read_text())
        w, h = int(meta['width']), int(meta['height'])
        return raw_path.stat().st_size == w * h
    except Exception:
        return False


def read_negative(path: str):
    """Return (mosaic_uint8 HxW, meta dict). Raises on malformed input."""
    raw_path, json_path = _stem_paths(path)
    if raw_path is None or json_path is None:
        raise FileNotFoundError(f"Not a V1 negative: {path}")
    meta = json.loads(json_path.read_text())
    w, h = int(meta['width']), int(meta['height'])
    data = np.fromfile(raw_path, dtype=np.uint8)
    if data.size != w * h:
        raise ValueError(f"V1 raw size {data.size} != {w}*{h}={w*h} for {raw_path}")
    return data.reshape(h, w), meta


def extract_negatives_from_zip(zip_path, dest_dir=None) -> list:
    """Extract a roll zip into raw + .json pairs and return the raw paths.

    Only base names from the archive are used, so there are no nested
    folders or path traversal.
    """
    import tempfile
    import zipfile

    zip_path = Path(zip_path)
    dest_dir = (Path(tempfile.mkdtemp(prefix='fb_v1_')) if dest_dir is None
                else Path(dest_dir))
    dest_dir.mkdir(parents=True, exist_ok=True)

    raws = []
    with zipfile.ZipFile(zip_path) as zf:
        entries = [n for n in zf.namelist() if not n.endswith('/') and os.path.basename(n)]
        by_base = {os.path.basename(n): n for n in entries}
        json_stems = {b[:-5]: n for b, n in by_base.items() if b.lower().endswith('.json')}
        for stem, jn in json_stems.items():
            rn = by_base.get(stem) or by_base.get(f"{stem}.raw")
            if rn is None:
                continue
            try:
                meta = json.loads(zf.read(jn))
                w, h = int(meta['width']), int(meta['height'])
            except Exception:
                continue
            data = zf.read(rn)
            if len(data) != w * h:
                log.warning("[v1] zip entry %s: size %d != %dx%d, skipping",
                            stem, len(data), w, h)
                continue
            raw_out = dest_dir / stem
            json_out = dest_dir / f"{stem}.json"
            # Already extracted: re-importing a roll is a no-op.
            if raw_out.exists() and raw_out.stat().st_size == w * h and json_out.exists():
                raws.append(raw_out)
                continue
            raw_out.write_bytes(data)
            json_out.write_bytes(zf.read(jn))
            raws.append(raw_out)
    is_v1_negative.cache_clear()
    return sorted(raws, key=lambda p: p.name.lower())


def roll_capture_date(zip_path):
    """Earliest timestamp of the entries in the zip, or None.

    Not the zip's own mtime: a roll can be exported long after it was shot.
    """
    import zipfile
    from datetime import datetime

    dates = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                dt = info.date_time
                # 1980 is the DOS epoch, i.e. no real timestamp.
                if dt and dt[0] > 1980:
                    try:
                        dates.append(datetime(*dt))
                    except ValueError:
                        pass
    except Exception:
        return None
    return min(dates) if dates else None


# =============================================================================
# DEVELOP
# =============================================================================

_WB_MATCH_GAIN = None


def _wb_match_gain() -> np.ndarray:
    """ACEScg gain for V1_WB_TEMP / V1_TINT, same as the sliders would apply."""
    global _WB_MATCH_GAIN
    if _WB_MATCH_GAIN is None:
        from .processor import _kelvin_to_acescg_gain, _tint_to_acescg_gain, BASE_KELVIN
        _WB_MATCH_GAIN = (_kelvin_to_acescg_gain(BASE_KELVIN + V1_WB_TEMP)
                          * _tint_to_acescg_gain(V1_TINT)).astype(np.float32)
    return _WB_MATCH_GAIN


def _to_linear(mosaic: np.ndarray, black: float, dither_lsb: float,
               rng: np.random.Generator) -> np.ndarray:
    """Black subtract + optional dither -> [0, 1] float32."""
    x = mosaic.astype(np.float32)
    if dither_lsb > 0.0:
        n = (rng.random(x.shape, dtype=np.float32)
             - rng.random(x.shape, dtype=np.float32)) * dither_lsb
        x = x + n
    x = (x - black) / (V1_WHITE_LEVEL - black)
    return np.clip(x, 0.0, 1.0)


def linear_rgb(path: str, *, black: float = V1_BLACK_LEVEL,
               dither_lsb: float = 0.0, target_long_edge: int | None = None,
               edge_aware: bool = True, seed: int = 0) -> np.ndarray:
    """Demosaiced linear camera RGB, before WB and matrix. Also used by the
    profiling tool."""
    mosaic, _ = read_negative(path)
    rng = np.random.default_rng(seed)

    # cv2 demosaics integers only; 16-bit keeps the dither.
    lin = _to_linear(mosaic, black, dither_lsb, rng)
    lin16 = np.clip(lin * 65535.0, 0, 65535).astype(np.uint16)

    code = V1_BAYER_CODE_EA if edge_aware else V1_BAYER_CODE
    rgb = cv2.cvtColor(lin16, code).astype(np.float32) / 65535.0

    h, w = rgb.shape[:2]
    if target_long_edge and max(h, w) > target_long_edge:
        s = target_long_edge / max(h, w)
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)),
                         interpolation=cv2.INTER_AREA)
    return rgb


def develop_v1(path: str, *, black: float = V1_BLACK_LEVEL,
               dither_lsb: float = V1_DITHER_LSB,
               target_long_edge: int = V1_TARGET_LONG_EDGE,
               seed: int = 0) -> np.ndarray:
    """Develop a V1 negative to linear ACEScg (H, W, 3) float32."""
    rgb = linear_rgb(path, black=black, dither_lsb=dither_lsb,
                     target_long_edge=target_long_edge, edge_aware=True, seed=seed)

    exp_gain = 2.0 ** V1_EXPOSURE_EV
    from .processor import XYZ_D50_TO_ACESCG, color_transform, _recover_highlights
    if V1_HIGHLIGHT_RECOVERY:
        rgb_wb = _recover_highlights(rgb, V1_ASN_D50, threshold=V1_HIGHLIGHT_THRESHOLD)
    else:
        rgb_wb = rgb / V1_ASN_D50[np.newaxis, np.newaxis, :]
    if exp_gain != 1.0:
        rgb_wb = rgb_wb * exp_gain
    fwd_to_acescg = (XYZ_D50_TO_ACESCG @ V1_FORWARD_MATRIX).astype(np.float32)
    acescg = color_transform(rgb_wb, fwd_to_acescg)
    acescg = acescg * _wb_match_gain()

    return np.ascontiguousarray(acescg, dtype=np.float32)
