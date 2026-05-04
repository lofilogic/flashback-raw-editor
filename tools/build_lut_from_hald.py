#!/usr/bin/env python3
"""
Build a 3D LUT from a pair of HALD-processed TIFFs.

  TIFF 1 (Lightroom)  — exported from the HALD DNG after opening in Lightroom.
                        Use Rec.2020 Gamma 2.4 (or 2.2), 16-bit.
                        All tone controls at 0; WB fixed/neutral.

  TIFF 2 (Flashback)  — exported from the Flashback-camera HALD DNG via
                        Flashback Editor (all spatial effects + LUT disabled).
                        This is the raw ACEScct intermediate, 16-bit TIFF.

Patch coordinates are read from the .hald_meta.json sidecar produced by
inject_hald_dng.py.  The script handles:
  • Lightroom exporting the full rotated sensor canvas (flip=6 / 90° CW).
  • Flashback Editor exporting at half the raw resolution (half_size=True).

Usage:
    python tools/build_lut_from_hald.py lightroom.tif flashback.tif meta.json \\
        --out match.cube [--lut-size 65] [--gamma 2.4] [--sample-px 4]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core  # noqa: F401

import cv2
import numpy as np
import colour


def auto_sample_px(meta):
    """Sensible sampling window: leave ~1px border per side from the patch.
    For patch_px=8 → sample 4 (clean), 6 → 4 (1px border), 5/4 → 2 (2x2 inner).
    Always ≥ 2 so we average a few photosites and beat read noise.
    """
    p = int(meta.get('patch_px', 8))
    return max(2, p - 2 if p <= 6 else 4)


# ---------------------------------------------------------------------------
# TIFF loading
# ---------------------------------------------------------------------------

def load_tiff_float(path: str) -> np.ndarray:
    """Load a 16-bit or 8-bit RGB TIFF as float32 in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'Cannot read TIFF: {path}')
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Patch sampling — two variants
# ---------------------------------------------------------------------------

def sample_patches_lr(img: np.ndarray, meta: dict, sample_px: int) -> np.ndarray:
    """
    Sample the Lightroom TIFF, which may be the full rotated sensor canvas.

    Handles flip=6 (90° CW) by transforming sensor-space patch centres into
    the LR TIFF's rotated coordinate system.  Falls back to unrotated sampling
    when flip metadata is absent.
    """
    n        = meta['n']
    patch_px = meta['patch_px']
    cols     = meta['cols']
    n_patches = n ** 3

    flip         = meta.get('flip', 0)
    top_margin   = meta.get('top_margin', 0)
    left_margin  = meta.get('left_margin', 0)
    crop_height  = meta.get('crop_height', img.shape[0])

    half    = max(1, sample_px // 2)
    img_h, img_w = img.shape[:2]
    result  = np.full((n_patches, 3), np.nan, dtype=np.float32)

    for k in range(n_patches):
        pr = k // cols
        pc = k % cols
        # Centre in sensor (raw) coordinates — HALD is placed by inject at
        # (top_margin, left_margin), so sensor coord = margin + within-HALD.
        cy_s = top_margin  + pr * patch_px + patch_px // 2
        cx_s = left_margin + pc * patch_px + patch_px // 2

        if flip == 6:   # 90° CW: top-left of sensor → top-right of portrait
            cy_img = cx_s - left_margin
            cx_img = (crop_height - 1) - (cy_s - top_margin)
        elif flip == 5: # 90° CCW: top-left of sensor → bottom-left of portrait
            cy_img = (meta.get('crop_width', img_w) - 1) - (cx_s - left_margin)
            cx_img = cy_s - top_margin
        else:
            cy_img = cy_s - top_margin
            cx_img = cx_s - left_margin

        y0 = max(0, int(cy_img) - half)
        y1 = min(img_h, int(cy_img) + half)
        x0 = max(0, int(cx_img) - half)
        x1 = min(img_w, int(cx_img) + half)

        if y0 >= y1 or x0 >= x1:
            continue
        result[k] = img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)

    return result


def sample_patches_fb(img: np.ndarray, meta: dict, sample_px: int,
                       fb_meta=None) -> np.ndarray:
    """
    Sample the Flashback TIFF.

    `fb_meta` (when provided) carries the FB-side sensor canvas dims; scale is
    image/canvas (e.g. 0.5 for half_size exports). When omitted, falls back
    to a heuristic against HALD dims — only correct if the export was cropped
    to the HALD region.
    The HALD is always in the top-left corner of the Flashback image.
    """
    n         = meta['n']
    patch_px  = meta['patch_px']
    cols      = meta['cols']
    rows      = meta['rows']
    n_patches = n ** 3

    hald_h = rows * patch_px
    hald_w = cols * patch_px
    img_h, img_w = img.shape[:2]

    if fb_meta is not None:
        canvas_w = fb_meta['img_w']
        canvas_h = fb_meta['img_h']
        scale = min(img_h / canvas_h, img_w / canvas_w)
        print(f'  FB scale (from fb meta canvas {canvas_w}×{canvas_h}): {scale:.3f}×')
    else:
        scale = min(img_h / hald_h, img_w / hald_w)
        if abs(scale - 1.0) > 0.01:
            print(f'  FB scale (auto, vs HALD {hald_w}×{hald_h}): {scale:.3f}×  '
                  '(pass --fb-meta if export is full canvas)')

    half   = max(1, int(sample_px * scale / 2))
    result = np.full((n_patches, 3), np.nan, dtype=np.float32)

    for k in range(n_patches):
        pr = k // cols
        pc = k % cols
        cy = int((pr * patch_px + patch_px // 2) * scale)
        cx = int((pc * patch_px + patch_px // 2) * scale)

        y0 = max(0, cy - half)
        y1 = min(img_h, cy + half)
        x0 = max(0, cx - half)
        x1 = min(img_w, cx + half)

        if y0 >= y1 or x0 >= x1:
            continue
        result[k] = img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)

    return result


# ---------------------------------------------------------------------------
# LUT building
# ---------------------------------------------------------------------------

def build_lut(src_acescct: np.ndarray,
              tgt_linear: np.ndarray,
              lut_size: int,
              k: int = 8,
              log=print) -> np.ndarray:
    """Build a lut_size³ LUT from scattered (source → target) pairs using
    inverse-distance-weighted KNN.

    cKDTree-based; ~1–10 s for 65³ training × 65³ grid versus minutes for
    Delaunay. With training density on the order of the LUT grid (n ≈
    lut_size), the answer is visually equivalent.
    """
    from scipy.spatial import cKDTree

    valid = (np.isfinite(src_acescct).all(axis=1) &
             np.isfinite(tgt_linear).all(axis=1))
    src = src_acescct[valid].astype(np.float64)
    tgt = tgt_linear[valid].astype(np.float64)

    log(f'  {src.shape[0]:,} valid sample pairs')
    log(f'  Source range  R:[{src[:,0].min():.3f},{src[:,0].max():.3f}]'
        f'  G:[{src[:,1].min():.3f},{src[:,1].max():.3f}]'
        f'  B:[{src[:,2].min():.3f},{src[:,2].max():.3f}]')
    log(f'  Target range  R:[{tgt[:,0].min():.3f},{tgt[:,0].max():.3f}]'
        f'  G:[{tgt[:,1].min():.3f},{tgt[:,1].max():.3f}]'
        f'  B:[{tgt[:,2].min():.3f},{tgt[:,2].max():.3f}]')

    log('  Building kd-tree...')
    tree = cKDTree(src)

    t = np.linspace(0.0, 1.0, lut_size, dtype=np.float64)
    gx, gy, gz = np.meshgrid(t, t, t, indexing='ij')
    query = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    n_query = len(query)

    log(f'  Querying KNN (k={k}) for {n_query:,} LUT nodes...')
    lut_vals = np.empty((n_query, 3), dtype=np.float64)
    chunk = 32768
    for start in range(0, n_query, chunk):
        end = min(start + chunk, n_query)
        dists, idxs = tree.query(query[start:end], k=k)
        # Inverse-distance weights, with epsilon to handle exact hits.
        weights = 1.0 / (dists + 1e-9)
        weights /= weights.sum(axis=1, keepdims=True)
        # Gather neighbour targets and combine.
        neighbour_tgts = tgt[idxs]                          # (chunk, k, 3)
        lut_vals[start:end] = np.einsum('ij,ijk->ik', weights, neighbour_tgts)
        if start % (chunk * 4) == 0 and start > 0:
            log(f'    {start:,} / {n_query:,}')

    return np.clip(lut_vals, 0.0, 1.0).reshape(
        lut_size, lut_size, lut_size, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_lut_from_target_only(target_tif, target_meta_path, fb_samples_npz,
                                output_lut_path, lut_size=65, sample_px=None, log=print):
    """Build a Flashback look-match LUT using a baked-in FB-side sample set.

    This is the GUI-friendly path: the user only ever provides the target
    camera DNG/TIFF — the FB side is loaded from a frozen .npz of pre-sampled
    ACEScct values (so users never need an FB DNG, FB Editor pass, etc.).

    `fb_samples_npz` is the path to fb_hald_samples.npz produced by
    calibrate_fb_baseline.py. The HALD layout there must match what was
    injected into the target DNG (same N, patch_px, cols, rows).
    """
    log(f'Loading target meta: {target_meta_path}')
    with open(target_meta_path) as f:
        meta = json.load(f)
    n = meta['n']; patch_px = meta['patch_px']
    cols = meta['cols']; rows = meta['rows']
    n_patches = n ** 3

    if sample_px is None:
        sample_px = auto_sample_px(meta)
    log(f'  Grid: {cols}×{rows}, N={n}, patch={patch_px}px → {n_patches:,} patches '
        f'(sample window: {sample_px}px)')

    log(f'Loading baked FB samples: {fb_samples_npz}')
    fb_data = np.load(fb_samples_npz)
    fb_samples = fb_data['samples']
    # Only N must match — patch_px and physical cols/rows differ per camera
    # (Bayer vs LinearRaw, varying sensor sizes), but k → chromaticity is
    # determined by N alone.
    if int(fb_data['n']) != n:
        raise RuntimeError(
            f'FB samples N mismatch: target HALD has n={n}, baked FB has '
            f'n={int(fb_data["n"])}. Re-bake the FB samples with the same n.')
    if fb_samples.shape[0] != n_patches:
        raise RuntimeError(
            f'FB samples count mismatch: expected {n_patches}, got {fb_samples.shape[0]}.')
    log(f'  Loaded {fb_samples.shape[0]:,} FB ACEScct samples')

    log(f'Loading target TIFF: {target_tif}')
    lr_img = load_tiff_float(target_tif)
    log(f'  Shape: {lr_img.shape[1]}×{lr_img.shape[0]}')

    log(f'Sampling target ({n_patches:,} patches, window={sample_px}px)...')
    lr_samples = sample_patches_lr(lr_img, meta, sample_px).astype(np.float32)

    log(f'Building {lut_size}³ LUT (KNN-weighted, k=8)...')
    log(f'  FB ACEScct ranges  R:[{fb_samples[:,0].min():.3f},{fb_samples[:,0].max():.3f}]'
        f'  G:[{fb_samples[:,1].min():.3f},{fb_samples[:,1].max():.3f}]'
        f'  B:[{fb_samples[:,2].min():.3f},{fb_samples[:,2].max():.3f}]')
    log(f'  Target ranges      R:[{np.nanmin(lr_samples[:,0]):.3f},{np.nanmax(lr_samples[:,0]):.3f}]'
        f'  G:[{np.nanmin(lr_samples[:,1]):.3f},{np.nanmax(lr_samples[:,1]):.3f}]'
        f'  B:[{np.nanmin(lr_samples[:,2]):.3f},{np.nanmax(lr_samples[:,2]):.3f}]')
    lut_table = build_lut(fb_samples, lr_samples, lut_size, log=log)
    log(f'  Output range: [{lut_table.min():.4f}, {lut_table.max():.4f}]')

    Path(output_lut_path).parent.mkdir(parents=True, exist_ok=True)
    lut3d = colour.LUT3D(
        table=lut_table,
        name=Path(output_lut_path).stem,
        domain=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )
    colour.write_LUT(lut3d, output_lut_path)
    log(f'✓ LUT written → {output_lut_path}')
    return output_lut_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('lightroom',   help='Lightroom output TIFF (sRGB or Rec.2020 Gamma 2.4)')
    ap.add_argument('flashback',   nargs='?', default=None,
                    help='Flashback Editor output TIFF. Omit to use baked FB samples.')
    ap.add_argument('meta',        help='Target HALD metadata JSON')
    ap.add_argument('--fb-meta',   default=None)
    ap.add_argument('--fb-samples', default=None,
                    help='Path to baked FB samples .npz (uses tools/fb_hald_samples.npz '
                         'by default if `flashback` arg is omitted).')
    ap.add_argument('--out',       default='match.cube')
    ap.add_argument('--lut-size',  type=int,   default=65)
    ap.add_argument('--sample-px', type=int,   default=4)
    args = ap.parse_args()

    if args.flashback is None:
        # Baked-FB-samples mode
        baked = args.fb_samples or str(Path(__file__).resolve().parent / 'fb_hald_samples.npz')
        build_lut_from_target_only(args.lightroom, args.meta, baked,
                                    args.out, lut_size=args.lut_size,
                                    sample_px=args.sample_px)
        return

    # Legacy mode: live FB TIFF + meta (for development / re-calibration)
    print(f'Loading metadata from {args.meta}')
    with open(args.meta) as f:
        meta = json.load(f)
    n_patches = meta['n'] ** 3
    print(f'\nLoading Lightroom TIFF: {args.lightroom}')
    lr_img = load_tiff_float(args.lightroom)
    print(f'Loading Flashback TIFF: {args.flashback}')
    fb_img = load_tiff_float(args.flashback)
    fb_meta = None
    if args.fb_meta:
        with open(args.fb_meta) as f:
            fb_meta = json.load(f)
    lr_samples = sample_patches_lr(lr_img, meta, args.sample_px)
    fb_samples = sample_patches_fb(fb_img, meta, args.sample_px, fb_meta=fb_meta)
    print(f'\nBuilding {args.lut_size}³ LUT...')
    lut_table = build_lut(fb_samples, lr_samples.astype(np.float32), args.lut_size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lut3d = colour.LUT3D(table=lut_table, name=Path(args.out).stem,
                          domain=np.array([[0,0,0],[1,1,1]]))
    colour.write_LUT(lut3d, args.out)
    print(f'\n✓ LUT written → {args.out}')


if __name__ == '__main__':
    main()
