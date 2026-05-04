#!/usr/bin/env python3
"""
Inject a HALD color calibration pattern into a DNG file.

For cameras other than Flashback: first convert the raw file to an
UNCOMPRESSED DNG with Adobe DNG Converter (Preferences → DNG Compatibility
→ check "Uncompressed").  The script copies that DNG and overwrites only the
pixel data, preserving every metadata tag, color matrix, and camera profile
exactly.

For Flashback Camera DNGs: use the original file directly.

Usage:
    python tools/inject_hald_dng.py input.dng output.dng [--n 33] [--patch-px 8]

Outputs:
    <output>.dng            — HALD-injected DNG (identical metadata to input)
    <output>.hald_meta.json — grid metadata required by build_lut_from_hald.py
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core  # noqa: F401  — numpy 2.0 shim

import numpy as np
import rawpy
import tifffile


# ---------------------------------------------------------------------------
# AsShotNeutral parsing
# ---------------------------------------------------------------------------

def _parse_asshotneutral(value):
    """Parse the AsShotNeutral DNG tag (50728) into [R, G, B] with G==1.

    tifffile delivers this tag in a couple of shapes depending on how the
    rationals were encoded; handle both:
      - 3 floats already converted (e.g. (0.5, 1.0, 0.61))
      - 3 numerators with a single denominator implied by the largest value
        (e.g. (508, 1024, 634) → [0.496, 1.0, 0.619])
      - 6 ints as (num, denom) pairs (e.g. (1000000, 1000000, ...) → [1, 1, 1])
    """
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 3:
        # Heuristic: if any value > ~10, treat as integer numerators sharing
        # an implicit denominator (typical of TIFF rational encoding).
        if arr.max() > 10:
            arr = arr / arr.max()
    elif arr.size == 6:
        nums = arr[::2]
        denoms = np.where(arr[1::2] == 0, 1, arr[1::2])
        arr = nums / denoms
    else:
        return None
    if arr[1] == 0:
        return None
    return arr / arr[1]   # normalise G to 1


# ---------------------------------------------------------------------------
# Natural-neutral ratios from DNG ColorMatrix tags (camera-agnostic)
# ---------------------------------------------------------------------------

# DNG LightSource enum → correlated colour temperature (Kelvin), per
# Adobe DNG spec section 6.4.1. Subset used for camera profile calibration.
_DNG_ILLUMINANT_K = {
    1: 6504, 17: 2856, 18: 4874, 19: 6774,
    20: 5500, 21: 6504, 22: 7504, 23: 5003, 24: 3200,
}


def _parse_rational_array(value, expected_len):
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == expected_len * 2:
        denoms = np.where(arr[1::2] == 0, 1, arr[1::2])
        return arr[::2] / denoms
    if arr.size == expected_len:
        return arr
    return None


def _kelvin_to_planckian_xy(kelvin):
    """Planckian-locus chromaticity at given CCT, via Krystek's (1985)
    polynomial approximation. Matches Adobe CR's "<K> Tint 0" interpretation
    (Tint=0 ↔ on the Planckian locus). Valid 1667–25000K.
    """
    kelvin = float(np.clip(kelvin, 1667, 25000))
    if kelvin < 4000:
        x = (-0.2661239e9 / kelvin**3 - 0.2343589e6 / kelvin**2
             + 0.8776956e3 / kelvin + 0.179910)
    else:
        x = (-3.0258469e9 / kelvin**3 + 2.1070379e6 / kelvin**2
             + 0.2226347e3 / kelvin + 0.240390)
    if kelvin < 2222:
        y = (-1.1063814 * x**3 - 1.34811020 * x**2
             + 2.18555832 * x - 0.20219683)
    elif kelvin < 4000:
        y = (-0.9549476 * x**3 - 1.37418593 * x**2
             + 2.09137015 * x - 0.16748867)
    else:
        y = (3.0817580 * x**3 - 5.87338670 * x**2
             + 3.75112997 * x - 0.37001483)
    return (x, y)


def _xy_to_temperature_uv(xy):
    """Compute CCT from xy via CIE 1960 (u,v) — matches DNG SDK's
    dng_temperature::Set_xy_coord. Returns Kelvin."""
    x, y = xy
    denom = -2 * x + 12 * y + 3
    if denom == 0:
        return 5000.0
    u = 4 * x / denom
    v = 6 * y / denom
    # Robertson's iso-temperature method via simplified polynomial inverse.
    # We don't need the exact Robertson tables for our use — we already know
    # target Kelvin going in. This is a fallback for xy → Kelvin mapping.
    n = (u - 0.3320) / (0.1858 - v)
    return 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33


def _illuminant_kelvin_to_xy(illuminant_code, fallback_kelvin):
    """Map a DNG LightSource enum value to the canonical (x, y) for that
    illuminant. Uses CIE-defined whitepoints where possible, falls back to
    Planckian-locus xy at the implied CCT."""
    # CIE-standard whitepoints
    fixed_xy = {
        17: (0.4476, 0.4075),  # Standard A
        18: (0.3484, 0.3516),  # Standard B (~4874K)
        19: (0.3101, 0.3162),  # Standard C
        20: (0.3324, 0.3474),  # D55
        21: (0.3127, 0.3290),  # D65
        22: (0.2990, 0.3149),  # D75
        23: (0.3457, 0.3585),  # D50
    }
    if illuminant_code in fixed_xy:
        return fixed_xy[illuminant_code]
    return _kelvin_to_planckian_xy(fallback_kelvin)


def _natural_ratios_from_colormatrix(tif, target_kelvin=5000):
    """Compute the camera's natural-neutral raw ratios at `target_kelvin` per
    Adobe DNG spec section 6.4.1.

    Algorithm (matches dng_color_spec::SetWhiteXY):
      1. Look up calibration illuminants' canonical xy from the LightSource enum.
      2. Compute target_xy on the Planckian locus at target_kelvin
         (CR slider's "K, Tint 0" interpretation).
      3. Compute mired-space interpolation weight g between illuminant CCTs.
      4. cm = lerp(cm1, cm2, g).
      5. neutral = cm @ (target_xy.x, target_xy.y, 1 - x - y)  ← chromaticity form
         (DNG spec uses the chromaticity vector that sums to 1, not Y=1 XYZ).
      6. Normalise to G=1 (display convention) — same camera_neutral up to scale
         as Adobe's max=1 normalization.

    Returns None if the DNG lacks ColorMatrix tags (e.g. FB's own DNGs).
    """
    p = tif.pages[0]
    if 50721 not in p.tags:
        return None
    cm1 = _parse_rational_array(p.tags[50721].value, 9)
    if cm1 is None:
        return None
    cm1 = cm1.reshape(3, 3)
    cm2 = None
    if 50722 in p.tags:
        cm2_arr = _parse_rational_array(p.tags[50722].value, 9)
        if cm2_arr is not None:
            cm2 = cm2_arr.reshape(3, 3)

    ill1 = int(p.tags[50778].value) if 50778 in p.tags else 21
    ill2 = int(p.tags[50779].value) if 50779 in p.tags else 17

    if cm2 is None:
        cm = cm1
    else:
        t1 = _DNG_ILLUMINANT_K.get(ill1, 6504)
        t2 = _DNG_ILLUMINANT_K.get(ill2, 2856)
        m1, m2, m_t = 1e6 / t1, 1e6 / t2, 1e6 / target_kelvin
        if m1 == m2:
            w = 0.5
        else:
            w = max(0.0, min(1.0, (m_t - m1) / (m2 - m1)))
        cm = (1 - w) * cm1 + w * cm2

    # Planckian locus xy at target Kelvin (matches CR slider's reference)
    x, y = _kelvin_to_planckian_xy(target_kelvin)
    XYZ_chromaticity = np.array([x, y, 1.0 - x - y], dtype=np.float64)
    cam_rgb = cm @ XYZ_chromaticity
    if cam_rgb[1] == 0:
        return None
    return cam_rgb / cam_rgb[1]


# ---------------------------------------------------------------------------
# Raw page discovery
# ---------------------------------------------------------------------------

def find_raw_page(tif):
    """
    Return the TiffPage that contains full-resolution raw sensor data.
    Checks both the main IFD and any SubIFDs (tag 330).
    Raises ValueError if none is found.
    """
    RAW_PHOTOMETRIC = {32803, 34892}   # CFA, LinearRaw
    candidates = []

    def _inspect(page):
        if 262 not in page.tags:
            return
        if page.tags[262].value not in RAW_PHOTOMETRIC:
            return
        h = page.tags[257].value if 257 in page.tags else 0
        w = page.tags[256].value if 256 in page.tags else 0
        candidates.append((h * w, page))

    for page in tif.pages:
        _inspect(page)
        if page.pages:
            for sub in page.pages:
                _inspect(sub)

    if not candidates:
        raise ValueError(
            'No CFA or LinearRaw IFD found — is this a valid DNG?\n'
            'For compressed DNGs, re-export with Adobe DNG Converter →\n'
            'Preferences → check "Uncompressed".'
        )
    return max(candidates, key=lambda x: x[0])[1]


# ---------------------------------------------------------------------------
# HALD generation
# ---------------------------------------------------------------------------

def compute_hald_layout(n, usable_w, usable_h, even_only=False):
    """Pick the largest patch_px (and cols/rows) that fits n³ patches in
    a usable_w × usable_h area. `even_only` enforces even patch_px for Bayer
    DNGs (CFA alignment). Tries square layout first, falls back to widest.

    Returns (patch_px, cols, rows). Raises if no layout fits.
    """
    target = n ** 3
    for p in range(8, 1, -1):
        if even_only and p % 2 != 0:
            continue
        max_cols = usable_w // p
        max_rows = usable_h // p
        if max_cols * max_rows < target:
            continue
        # Prefer square layout when it fits (cleaner sampling)
        cols_sq = math.ceil(math.sqrt(target))
        if cols_sq <= max_cols and math.ceil(target / cols_sq) <= max_rows:
            cols = cols_sq
            return p, cols, math.ceil(target / cols)
        # Fall back: pack as wide as possible
        cols = max_cols
        rows = math.ceil(target / cols)
        if rows <= max_rows:
            return p, cols, rows
    raise RuntimeError(
        f'Cannot fit {target} HALD patches in {usable_w}×{usable_h} '
        f'(even_only={even_only}). Try a smaller --n.')


def _make_channel_grids(n, patch_px, black, white, natural_ratios, cols, rows):
    """Return (r_img, g_img, b_img) each of shape (rows*patch_px, cols*patch_px).

    Caller passes `cols`/`rows` from compute_hald_layout — they encode the
    chosen layout (which depends on usable sensor area + Bayer/linear).
    """
    range_ = (white - black)
    nr = np.asarray(natural_ratios, dtype=np.float64)
    scale_rgb = nr * range_ / max(n - 1, 1)

    n_patches = n ** 3
    n_cells = rows * cols
    if n_cells < n_patches:
        raise RuntimeError(f'cols*rows={n_cells} < n³={n_patches}')

    idx = np.arange(n_cells)
    valid = idx < n_patches

    r_vals = np.where(valid, np.round(black + (idx % n)          * scale_rgb[0]), black).astype(np.uint16)
    g_vals = np.where(valid, np.round(black + ((idx // n) % n)   * scale_rgb[1]), black).astype(np.uint16)
    b_vals = np.where(valid, np.round(black + (idx // (n * n))   * scale_rgb[2]), black).astype(np.uint16)

    r_img = np.repeat(np.repeat(r_vals.reshape(rows, cols), patch_px, 0), patch_px, 1)
    g_img = np.repeat(np.repeat(g_vals.reshape(rows, cols), patch_px, 0), patch_px, 1)
    b_img = np.repeat(np.repeat(b_vals.reshape(rows, cols), patch_px, 0), patch_px, 1)

    return r_img, g_img, b_img


def generate_hald_3ch(n, patch_px, black, white, sensor_h, sensor_w,
                       natural_ratios, cols, rows, top_margin=0, left_margin=0):
    """HALD for a LinearRaw (3-channel) DNG. Placed at (top_margin, left_margin)
    so the HALD lands inside the DNG's active area (some DNGs mask leading
    rows/cols)."""
    r_img, g_img, b_img = _make_channel_grids(n, patch_px, black, white,
                                                natural_ratios, cols, rows)
    hald_h, hald_w = r_img.shape

    canvas = np.full((sensor_h, sensor_w, 3), black, dtype=np.uint16)
    y0, x0 = top_margin, left_margin
    ph = min(hald_h, sensor_h - y0)
    pw = min(hald_w, sensor_w - x0)
    canvas[y0:y0+ph, x0:x0+pw, 0] = r_img[:ph, :pw]
    canvas[y0:y0+ph, x0:x0+pw, 1] = g_img[:ph, :pw]
    canvas[y0:y0+ph, x0:x0+pw, 2] = b_img[:ph, :pw]

    meta = {'n': n, 'patch_px': patch_px, 'cols': cols, 'rows': rows,
            'black': black, 'white': white, 'channels': 3,
            'img_w': sensor_w, 'img_h': sensor_h,
            'natural_ratios': [float(x) for x in natural_ratios]}
    return canvas, meta


def generate_hald_1ch(n, patch_px, cfa_pattern, black, white, sensor_h, sensor_w,
                       natural_ratios, cols, rows, top_margin=0, left_margin=0):
    """HALD for a CFA (1-channel Bayer) DNG. Bayer alignment requires even
    patch_px and an even (top_margin, left_margin)."""
    r_img, g_img, b_img = _make_channel_grids(n, patch_px, black, white,
                                                natural_ratios, cols, rows)
    hald_h, hald_w = r_img.shape

    # Bayer alignment: ensure offset is even on both axes
    y0 = (top_margin // 2) * 2
    x0 = (left_margin // 2) * 2

    canvas = np.full((sensor_h, sensor_w), black, dtype=np.uint16)
    ph = min(hald_h, sensor_h - y0)
    pw = min(hald_w, sensor_w - x0)

    for dy in range(2):
        for dx in range(2):
            ch = int(cfa_pattern[dy, dx])
            src = g_img if ch in (1, 3) else (r_img if ch == 0 else b_img)
            canvas[y0+dy:y0+ph:2, x0+dx:x0+pw:2] = src[dy:ph:2, dx:pw:2]

    meta = {'n': n, 'patch_px': patch_px, 'cols': cols, 'rows': rows,
            'black': black, 'white': white, 'channels': 1,
            'cfa_pattern': cfa_pattern.tolist(),
            'img_w': sensor_w, 'img_h': sensor_h,
            'natural_ratios': [float(x) for x in natural_ratios]}
    return canvas, meta


# ---------------------------------------------------------------------------
# 10-bit packing
# ---------------------------------------------------------------------------

def pack_10bit(data: np.ndarray) -> bytes:
    """
    Pack a uint16 array (values 0–1023) into TIFF-standard packed 10-bit bytes.
    Format: 4 pixels → 5 bytes, MSB first.
    """
    flat = data.flatten().astype(np.uint32)
    n = len(flat)
    pad = (-n) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint32)])
    g = flat.reshape(-1, 4)
    out = np.zeros((len(g), 5), dtype=np.uint8)
    out[:, 0] =  (g[:, 0] >> 2) & 0xFF
    out[:, 1] = ((g[:, 0] & 0x3) << 6) | ((g[:, 1] >> 4) & 0x3F)
    out[:, 2] = ((g[:, 1] & 0xF) << 4) | ((g[:, 2] >> 6) & 0xF)
    out[:, 3] = ((g[:, 2] & 0x3F) << 2) | ((g[:, 3] >> 8) & 0x3)
    out[:, 4] =   g[:, 3] & 0xFF
    return out.flatten().tobytes()[:n * 10 // 8]


# ---------------------------------------------------------------------------
# Byte injection
# ---------------------------------------------------------------------------

def inject_pixels(dst_path: str, strip_offsets, strip_counts, pixel_bytes: bytes):
    """Overwrite strip data in a copied DNG with new pixel bytes."""
    if isinstance(strip_offsets, (int, np.integer)):
        strip_offsets = (int(strip_offsets),)
        strip_counts  = (int(strip_counts),)

    with open(dst_path, 'r+b') as f:
        pos = 0
        for offset, count in zip(strip_offsets, strip_counts):
            count = int(count)
            chunk = pixel_bytes[pos: pos + count]
            if len(chunk) < count:
                chunk += b'\x00' * (count - len(chunk))
            f.seek(int(offset))
            f.write(chunk)
            pos += count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def inject_hald(input_dng, output_dng, n=65, patch_px=None,
                 natural_ratios_override=None, target_kelvin=5000,
                 log=print):
    """Inject a HALD pattern into a DNG. Programmatic API for the GUI.

    `natural_ratios_override` is an iterable of 3 floats to use directly
    instead of the auto-derived natural ratios. `log` is a callable that
    receives status strings (defaults to print, the GUI replaces it).

    Returns dict: {'meta_path': ..., 'output_dng': ..., 'natural_ratios': ...,
    'is_flashback': bool}
    """
    # ── Detect format ──────────────────────────────────────────────────────
    log(f'Inspecting {input_dng}')
    with tifffile.TiffFile(input_dng) as tif:
        raw_page     = find_raw_page(tif)
        photometric  = raw_page.tags[262].value
        sensor_w     = raw_page.tags[256].value
        sensor_h     = raw_page.tags[257].value
        bps          = raw_page.tags[258].value
        bits_ps      = bps[0] if hasattr(bps, '__len__') else int(bps)
        compression  = int(raw_page.tags[259].value) if 259 in raw_page.tags else 1
        strip_offsets = raw_page.tags[273].value
        strip_counts  = raw_page.tags[279].value
        make_tag = ''
        if 271 in tif.pages[0].tags:
            make_tag = str(tif.pages[0].tags[271].value or '').strip()

    is_linear = (photometric == 34892)
    fmt_str = 'LinearRaw 3-ch' if is_linear else f'CFA 1-ch ({bits_ps}-bit)'
    log(f'  Format     : {fmt_str}')
    log(f'  Sensor     : {sensor_w}×{sensor_h}')
    log(f'  Compression: {compression}  {"✓ uncompressed" if compression == 1 else "✗ COMPRESSED"}')

    if compression != 1:
        raise RuntimeError(
            'DNG is compressed. Re-export with Adobe DNG Converter → '
            'Preferences → DNG Compatibility → check "Uncompressed".')

    # ── Compute natural-neutral ratios from DNG metadata ───────────────────
    natural_ratios = None
    with tifffile.TiffFile(input_dng) as tif:
        natural_ratios = _natural_ratios_from_colormatrix(tif, target_kelvin=target_kelvin)
    if natural_ratios is not None:
        log(f'  Natural    : R={natural_ratios[0]:.4f}  G={natural_ratios[1]:.4f}  '
            f'B={natural_ratios[2]:.4f}  (DNG-spec WB derivation @ {target_kelvin}K)')
    else:
        asn_raw = None
        with tifffile.TiffFile(input_dng) as tif:
            if 50728 in tif.pages[0].tags:
                asn_raw = tif.pages[0].tags[50728].value
        natural_ratios = _parse_asshotneutral(asn_raw)
        if natural_ratios is None:
            log('  ⚠ No ColorMatrix or AsShotNeutral; falling back to [1, 1, 1].')
            natural_ratios = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    # ── Calibration metadata ───────────────────────────────────────────────
    rawpy_sizes = {}
    try:
        with rawpy.imread(input_dng) as raw:
            # rawpy returns 4 black levels (RGGB or RGBG); for LinearRaw DNGs
            # from Adobe DNG Converter, the 4th slot is a bogus 0 that shifts
            # min() to clip the actual BL to zero — and then LR maps the
            # bottom several HALD patches below black, crushing them. Pick the
            # max of non-zero entries so we land on the real BL.
            bls = [int(b) for b in raw.black_level_per_channel]
            non_zero = [b for b in bls if b > 0]
            black = int(max(non_zero)) if non_zero else 0
            white = int(raw.white_level)
            raw_pat = raw.raw_pattern
            if raw_pat is not None and np.asarray(raw_pat).ndim == 2:
                cfa_pattern = np.array(raw_pat, dtype=np.int32)
            else:
                cfa_pattern = np.array([[0, 1], [1, 2]], dtype=np.int32)
            s = raw.sizes
            # Use DNG crop margins if present (Adobe DNG Converter sets these);
            # fall back to LibRaw's top/left_margin for non-DNG raws.
            top_margin  = getattr(s, 'crop_top_margin',  s.top_margin)
            left_margin = getattr(s, 'crop_left_margin', s.left_margin)
            crop_w      = getattr(s, 'crop_width',       s.width)
            crop_h      = getattr(s, 'crop_height',      s.height)
            rawpy_sizes = {
                'flip':         int(s.flip),
                'top_margin':   int(top_margin),
                'left_margin':  int(left_margin),
                'crop_width':   int(crop_w),
                'crop_height':  int(crop_h),
            }
    except Exception:
        black = 0
        white = (1 << bits_ps) - 1
        cfa_pattern = np.array([[0, 1], [1, 2]], dtype=np.int32)

    # FB-specific overrides (BL=64 + measured sensor response at 5000K).
    is_flashback = make_tag.lower().startswith('flashback')
    FB_NATURAL_5000K = np.array([0.5156, 1.0, 0.6519], dtype=np.float64)
    if is_flashback:
        from core.config import SENSOR_BLACK
        if black != SENSOR_BLACK:
            log(f'  → FB: overriding BL {black} → {SENSOR_BLACK} (pipeline value)')
            black = SENSOR_BLACK
        natural_ratios = FB_NATURAL_5000K.copy()
        log(f'  → FB: measured sensor response @ 5000K: '
            f'R={natural_ratios[0]:.4f}  G={natural_ratios[1]:.4f}  B={natural_ratios[2]:.4f}')

    if natural_ratios_override is not None:
        natural_ratios = np.asarray(natural_ratios_override, dtype=np.float64)
        natural_ratios = natural_ratios / natural_ratios[1]
        log(f'  → override natural ratios: '
            f'R={natural_ratios[0]:.4f}  G={natural_ratios[1]:.4f}  B={natural_ratios[2]:.4f}')

    log(f'  Black/White: {black} / {white}')
    log(f'  Natural    : R={natural_ratios[0]:.4f}  G={natural_ratios[1]:.4f}  B={natural_ratios[2]:.4f}')
    if not is_linear:
        log(f'  CFA pattern: {cfa_pattern.tolist()}')

    # ── Pick layout (auto-fit n³ patches into usable sensor area) ──────────
    tm = int(rawpy_sizes.get('top_margin', 0))
    lm = int(rawpy_sizes.get('left_margin', 0))
    # Bayer alignment: round margins down to even pixels so the CFA grid keeps
    # phase, then use the even-rounded margins as the HALD origin.
    if not is_linear:
        tm = (tm // 2) * 2
        lm = (lm // 2) * 2
    usable_w = sensor_w - lm
    usable_h = sensor_h - tm
    if patch_px is None:
        chosen_p, cols, rows = compute_hald_layout(
            n, usable_w, usable_h, even_only=not is_linear)
        log(f'  Auto-layout: patch_px={chosen_p}, cols×rows={cols}×{rows}')
    else:
        chosen_p = patch_px
        # Manual patch_px: derive cols/rows that fit (square-first)
        target = n ** 3
        max_cols = usable_w // chosen_p
        max_rows = usable_h // chosen_p
        if max_cols * max_rows < target:
            raise RuntimeError(
                f'patch_px={chosen_p} can\'t fit {target} patches in '
                f'{usable_w}×{usable_h}; let inject_hald auto-pick (patch_px=None)')
        cols_sq = math.ceil(math.sqrt(target))
        if cols_sq <= max_cols and math.ceil(target / cols_sq) <= max_rows:
            cols = cols_sq
        else:
            cols = max_cols
        rows = math.ceil(target / cols)
        log(f'  Layout: patch_px={chosen_p}, cols×rows={cols}×{rows} (manual)')

    # ── Generate HALD ──────────────────────────────────────────────────────
    log(f'Generating N={n} HALD ({n**3:,} patches, {chosen_p}px each)...')
    log(f'  HALD origin: ({tm}, {lm}) (DNG crop margins, Bayer-aligned)')
    if is_linear:
        hald, meta = generate_hald_3ch(n, chosen_p, black, white, sensor_h, sensor_w,
                                        natural_ratios, cols, rows,
                                        top_margin=tm, left_margin=lm)
        pixel_bytes = hald.tobytes()
    else:
        hald, meta = generate_hald_1ch(n, chosen_p, cfa_pattern, black, white,
                                        sensor_h, sensor_w, natural_ratios, cols, rows,
                                        top_margin=tm, left_margin=lm)
        if bits_ps == 10:
            pixel_bytes = pack_10bit(hald)
        elif bits_ps in (14, 16):
            pixel_bytes = hald.tobytes()
        else:
            raise NotImplementedError(f'{bits_ps}-bit packing not implemented')

    expected = sum(int(c) for c in (strip_counts if hasattr(strip_counts, '__len__') else [strip_counts]))
    log(f'  Payload    : {len(pixel_bytes):,} bytes (expected {expected:,})')
    log(f'  Grid       : {meta["cols"]}×{meta["rows"]} → {meta["cols"]*chosen_p}×{meta["rows"]*chosen_p} px HALD')

    # ── Copy DNG + inject pixels ───────────────────────────────────────────
    Path(output_dng).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_dng, output_dng)
    inject_pixels(output_dng, strip_offsets, strip_counts, pixel_bytes)
    log(f'Wrote {output_dng}')

    # ── Sidecar JSON ───────────────────────────────────────────────────────
    meta.update(rawpy_sizes)
    meta_path = str(Path(output_dng).with_suffix('')) + '.hald_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log(f'Wrote {meta_path}')
    log(f'✓ Done. {n}³={n**3:,} colour patches.')

    return {
        'meta_path': meta_path,
        'output_dng': output_dng,
        'natural_ratios': natural_ratios.tolist(),
        'is_flashback': is_flashback,
        'is_linear': is_linear,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input',      help='Source DNG (uncompressed for non-Flashback cameras)')
    ap.add_argument('output',     help='Output DNG path')
    ap.add_argument('--n',        type=int, default=65,
                    help='Patches per LUT axis — N³ total (default 65, matches '
                         'the LUT grid for one-patch-per-node sampling).')
    ap.add_argument('--patch-px', type=int, default=None,
                    help='Pixels per patch side. Default: auto-fit to sensor area '
                         '(largest that holds n³ patches).')
    ap.add_argument('--natural-ratios', default=None,
                    help='Override natural-neutral ratios as "R,G,B" (G=1).')
    ap.add_argument('--target-kelvin', type=int, default=5000)
    args = ap.parse_args()

    override = None
    if args.natural_ratios is not None:
        override = [float(x) for x in args.natural_ratios.split(',')]
        if len(override) != 3:
            print('Error: --natural-ratios must be "R,G,B" (3 floats)')
            sys.exit(1)

    inject_hald(args.input, args.output, n=args.n, patch_px=args.patch_px,
                natural_ratios_override=override, target_kelvin=args.target_kelvin)


if __name__ == '__main__':
    main()
