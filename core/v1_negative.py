"""Flashback ONE35 **V1** "negative" reader + developer.

The V1 camera cannot record DNGs. Instead it exports a "negative": a
*headerless* single-channel 8-bit RGGB Bayer dump (``<uuid>.raw``, or just
``<uuid>`` with no extension) plus a sidecar ``<uuid>.json`` carrying the
geometry and capture metadata.

This module turns one of those negatives into the SAME linear-ACEScg
intermediate the DNG path produces (see core/processor.py), so everything
downstream — sliders, halation, LUT, grain — is shared and the V1 frames
look like the V2 frames. The only V1-specific work lives here:

    read int8 mosaic  ->  black-subtract  ->  (decode dither)
      ->  demosaic RGGB  ->  downscale to V2 pixel-scale
      ->  white-balance (ASN)  ->  ForwardMatrix (raw_wb -> XYZ_D50)
      ->  XYZ_D50 -> ACEScg

The ForwardMatrix and ASN are produced by tools/generate_matrices_v1.py
from a ColorChecker + flat-field shot, exactly mirroring how the V2 matrices
were derived (tools/generate_matrices_from_colorchart.py).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# =============================================================================
# V1 SENSOR / DEVELOP CONSTANTS
# =============================================================================

# Black pedestal in raw code values. Measured from the sample negatives the
# histogram rises from ~3; lock this exactly with a lens-cap dark frame.
V1_BLACK_LEVEL = 3.0
V1_WHITE_LEVEL = 255.0

# Match the V2 working pixel-scale: V2 develops half_size to a 2072 px long
# edge, and every px-denominated effect (grain tile, halation/softness/sharpen
# radii) is calibrated against that. We demosaic V1 at full res (2560) for
# detail, then downscale so the long edge lands here — preserving V1's native
# 3:2 aspect while making the look transfer 1:1.
V1_TARGET_LONG_EDGE = 2072

# OpenCV Bayer code for the V1 CFA. The bright (green) sites sit at [0,1] and
# [1,0] in the mosaic, so the corners are R/B and the pattern is the BG/RG pair;
# the *RG* code (corner [0,0] = blue) is the one that yields natural warm colour
# on a daylight scene (BG comes out cyan). R/B is also self-correcting once
# profiled — the 3x3 matrix can absorb the swap — but the profiling tool MUST
# use this same code so any residual labelling is consistent.
V1_BAYER_CODE = cv2.COLOR_BayerRG2RGB
# Edge-aware variant for the final develop; the plain code above is fine for
# profiling patch means (flat patches, demosaic quality irrelevant there).
V1_BAYER_CODE_EA = cv2.COLOR_BayerRG2RGB_EA

# Decode dither amplitude in LSB. The sensor self-dithers (~>=1 LSB noise)
# everywhere except the deepest shadows; a touch of triangular (TPDF) noise
# breaks the quantization lattice there so the downstream tone/LUT stretch
# can't band. Grain re-dithers at the output, so this stays subtle.
V1_DITHER_LSB = 0.5

# Global exposure trim (EV) applied in develop, before colour. The develop's
# absolute brightness is anchored to the calibration chart's exposure level
# (the ForwardMatrix maps the chart's grey patch to its reference XYZ), so the
# right fix for a consistent over/under is to re-shoot the chart at the metering
# users actually use (Ultra-low). This knob is the residual fine-tune; keep it
# near 0 and let calibration do the work.
V1_EXPOSURE_EV = -1.2

# Highlight recovery (mirrors the V2 DNG path's "inpaint opposed"): reconstruct
# a clipped channel from the unclipped two so blown highlights keep plausible
# colour instead of skewing. Runs on the pre-WB demosaiced RGB, BEFORE the
# exposure trim — the trim scales clipped 1.0 values below the detection point.
V1_HIGHLIGHT_RECOVERY = True
V1_HIGHLIGHT_THRESHOLD = 0.97

# White-balance match to V2, baked as a post-matrix ACEScg gain (NOT in ASN: a
# camera-space WB nudge gets remapped by the colour matrix and lands far weaker
# than the slider, which acts post-matrix). These reproduce the manual slider
# correction exactly, so V1 frames open already-matched at slider zero. Units
# match ImageAdjustments; tune these to taste (more negative tint = greener):
#   V1_WB_TEMP  -50.0  == WB slider label 5550 K  (5600 + wb_temp)
#   V1_TINT      -3.0  == tint slider label -15   (label = tint * 5)
V1_WB_TEMP = -75.0
V1_TINT = -4.0

# ---- COLOR (calibrated via tools/generate_matrices_v1.py) --------------------
# ASN_V1: camera response to a neutral under D50, from the ColorChecker's
# neutral patches (no separate flat field needed), normalised to green.
# ForwardMatrix: WB raw (raw/ASN) -> XYZ_D50. If V1_FORWARD_MATRIX is None the
# develop path falls back to a linear-sRGB-ish approximation for geometry/scale
# validation. Re-derive both from a frame-filling, evenly-lit, unclipped chart.
#
# (The V2 WB match lives in V1_WB_TEMP/V1_TINT above, applied post-matrix.)
V1_ASN_D50 = np.array([0.876826, 1.0, 0.862259], dtype=np.float32)
V1_FORWARD_MATRIX = np.array([
    [ 0.752684,  0.212928, -0.001412],
    [-0.006849,  1.438282, -0.431433],
    [ 0.246380, -1.381695,  1.960215],
], dtype=np.float32)  # d50 fit_rms=0.0627 n=24 (reds OK; deep-blue luminance slightly low)
_CALIBRATED = V1_FORWARD_MATRIX is not None


# =============================================================================
# FILE HANDLING
# =============================================================================

def _stem_paths(path: str):
    """Given any of the negative's files (.raw, .json, or extensionless raw),
    return (raw_path, json_path)."""
    p = Path(path)
    stem_dir = p.parent
    stem = p.stem if p.suffix in ('.raw', '.json') else p.name
    json_path = stem_dir / f"{stem}.json"
    raw_candidates = [stem_dir / f"{stem}.raw", stem_dir / stem]
    raw_path = next((c for c in raw_candidates if c.exists()), None)
    return raw_path, (json_path if json_path.exists() else None)


def is_v1_negative(path: str) -> bool:
    """True if `path` belongs to a V1 negative: a sidecar .json exists and the
    raw payload is exactly width*height bytes (headerless, 8-bit, 1 channel)."""
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
    """Extract a camera "roll" zip of V1 negatives to a folder of paired files.

    The camera exports each frame as an *extensionless* raw plus a same-named
    ``.json`` sidecar, bundled in a zip. This pairs them by stem, validates the
    raw payload against the JSON geometry, and writes both out flattened (the
    archive's internal paths are ignored, so there's no path-traversal risk and
    no nested folders). Returns the extracted raw Paths, sorted by name — feed
    them straight to the loader; develop_v1 finds each sidecar beside its raw.
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
            # raw entry is the extensionless twin (tolerate a .raw twin too)
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
            # Idempotent like the V2 camera import: if this frame was already
            # extracted here (right-sized raw + sidecar present), reuse it
            # instead of overwriting, so re-importing a roll is a no-op.
            if raw_out.exists() and raw_out.stat().st_size == w * h and json_out.exists():
                raws.append(raw_out)
                continue
            raw_out.write_bytes(data)
            json_out.write_bytes(zf.read(jn))
            raws.append(raw_out)
    return sorted(raws, key=lambda p: p.name.lower())


def roll_capture_date(zip_path):
    """Best-effort capture date for a V1 roll, read from the timestamps the
    camera stamped on the negatives *inside* the zip — not the zip's own date.

    Old rolls can be exported from the Flashback app at any later time, so the
    archive's mtime is unreliable; the per-frame entry timestamps reflect when
    the roll was actually shot. Returns the earliest valid entry datetime, or
    None if the zip carries no usable timestamps (caller falls back).
    """
    import zipfile
    from datetime import datetime

    dates = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                dt = info.date_time  # (year, month, day, hour, min, sec)
                # ZIP timestamps are DOS-epoch based (>= 1980); a 1980-01-01
                # default usually means "no real timestamp", so skip it.
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
    """Cached ACEScg diagonal gain reproducing the V2-match WB+tint slider
    correction. Computed via the renderer's own gain functions so it's exactly
    equivalent; imported lazily to avoid the processor<->v1_negative import cycle."""
    global _WB_MATCH_GAIN
    if _WB_MATCH_GAIN is None:
        from .processor import _kelvin_to_acescg_gain, _tint_to_acescg_gain, BASE_KELVIN
        _WB_MATCH_GAIN = (_kelvin_to_acescg_gain(BASE_KELVIN + V1_WB_TEMP)
                          * _tint_to_acescg_gain(V1_TINT)).astype(np.float32)
    return _WB_MATCH_GAIN


def _to_linear(mosaic: np.ndarray, black: float, dither_lsb: float,
               rng: np.random.Generator) -> np.ndarray:
    """Black-subtract + optional TPDF decode-dither -> linear [0,1] float32."""
    x = mosaic.astype(np.float32)
    if dither_lsb > 0.0:
        # Triangular PDF (sum of two uniforms): zero-mean, ~dither_lsb peak.
        n = (rng.random(x.shape, dtype=np.float32)
             - rng.random(x.shape, dtype=np.float32)) * dither_lsb
        x = x + n
    x = (x - black) / (V1_WHITE_LEVEL - black)
    return np.clip(x, 0.0, 1.0)


def linear_rgb(path: str, *, black: float = V1_BLACK_LEVEL,
               dither_lsb: float = 0.0, target_long_edge: int | None = None,
               edge_aware: bool = True, seed: int = 0) -> np.ndarray:
    """Read + black-subtract + demosaic a V1 negative to **pre-colour** linear
    RGB float32 [0,1] (no WB, no matrix). Shared by develop_v1 and the
    profiling tool so the CFA/black handling is byte-identical on both paths.
    """
    mosaic, _ = read_negative(path)
    rng = np.random.default_rng(seed)

    # linearise (black + optional dither). cv2 demosaic needs an integer image,
    # so we promote to 16-bit AFTER dither/black-subtract to carry the lattice
    # break through the demosaic.
    lin = _to_linear(mosaic, black, dither_lsb, rng)            # HxW float[0,1]
    lin16 = np.clip(lin * 65535.0, 0, 65535).astype(np.uint16)

    code = V1_BAYER_CODE_EA if edge_aware else V1_BAYER_CODE
    rgb = cv2.cvtColor(lin16, code).astype(np.float32) / 65535.0

    # downscale to V2 pixel-scale (long edge), INTER_AREA (anti-aliased)
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
    """Develop a V1 negative to a linear-ACEScg float32 image (H, W, 3).

    Mirrors the DNG path's output contract so the result can be dropped into
    FlashbackProcessor.intermediate_acescg and rendered identically.
    """
    rgb = linear_rgb(path, black=black, dither_lsb=dither_lsb,
                     target_long_edge=target_long_edge, edge_aware=True, seed=seed)

    exp_gain = 2.0 ** V1_EXPOSURE_EV
    from .processor import (XYZ_D50_TO_ACESCG, LINSRGB_TO_ACESCG,
                            color_transform, _recover_highlights)
    if _CALIBRATED:
        # WB + highlight recovery in pre-WB raw space (recovery returns WB'd RGB),
        # then the exposure trim, then the WB-matched ForwardMatrix -> ACEScg.
        if V1_HIGHLIGHT_RECOVERY:
            rgb_wb = _recover_highlights(rgb, V1_ASN_D50, threshold=V1_HIGHLIGHT_THRESHOLD)
        else:
            rgb_wb = rgb / V1_ASN_D50[np.newaxis, np.newaxis, :]
        if exp_gain != 1.0:
            rgb_wb = rgb_wb * exp_gain
        fwd_to_acescg = (XYZ_D50_TO_ACESCG @ V1_FORWARD_MATRIX).astype(np.float32)
        acescg = color_transform(rgb_wb, fwd_to_acescg)
        # V2 WB match: same post-matrix ACEScg gain the WB/tint sliders apply.
        acescg = acescg * _wb_match_gain()
    else:
        # PLACEHOLDER: treat demosaiced raw as linear sRGB-ish for a sane
        # (not colour-accurate) preview until the matrix is calibrated.
        if exp_gain != 1.0:
            rgb = rgb * exp_gain
        acescg = color_transform(rgb, LINSRGB_TO_ACESCG)

    return np.ascontiguousarray(acescg, dtype=np.float32)
