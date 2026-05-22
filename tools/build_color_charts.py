"""
Build paired color-swatch charts from aligned film/digital image pairs.

For each pair (matched by filename stem):
  1. Find patches that are flat in BOTH images (defends against alignment slop).
  2. Average each surviving patch over a wide window for noise/parallax tolerance.
  3. Farthest-point-sample in CIELAB on the film side to spread across the gamut.
  4. Sort by L (rows) then hue (within row) and render two charts with an
     identical grid layout — both as 16-bit TIFFs (film in sRGB, digital
     preserving raw ACEScct/Rec.2020 values).

The digital TIFFs are written with raw values preserved, so colormatch sees
exactly the ACEScct/Rec.2020 numbers your runtime LUT would receive.

Usage:
  python tools/build_color_charts.py \
      --film-dir path/to/film \
      --digital-dir path/to/digital \
      --out-dir path/to/charts \
      [--grid 8x6] [--border-frac 0.12] [--patch-detect 16] \
      [--patch-sample 48] [--blur-sigma 1.5] [--flatness-pct 10] [--cell-px 80]
"""
import argparse
import os
import re
import sys
from pathlib import Path

# Make this script runnable from anywhere and pick up the project's numpy 2.0
# shim from core/__init__.py before colour-science is imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core  # noqa: F401  — applies np.float_ shim before `import colour`

import colour
import cv2
import numpy as np

from core.config import POST_AE_EXPOSURE_BOOST_EV


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

FILM_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
DIGITAL_EXTS = {'.tif', '.tiff'}


def load_film(path: Path) -> np.ndarray:
    """Load film image as float32 RGB in [0,1]. Pixel values pass through
    untouched — the encoding is told to downstream metrics via
    --film-colorspace."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read film image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def _acescct_decode(x: np.ndarray) -> np.ndarray:
    return np.where(
        x < 0.155251141552511,
        (x - 0.0729055341958355) / 10.5402377416545,
        np.power(2.0, x * 17.52 - 9.72),
    ).astype(np.float32)


def _acescct_encode(linear: np.ndarray) -> np.ndarray:
    return np.where(
        linear <= 0.0078125,
        10.5402377416545 * linear + 0.0729055341958355,
        (np.log2(np.maximum(linear, 1e-10)) + 9.72) / 17.52,
    ).astype(np.float32)


def load_digital(path: Path, boost_ev: float = 0.0) -> np.ndarray:
    """Load 16-bit ACEScct/Rec.2020 TIFF as float32 in [0,1].
    If boost_ev != 0, applies a linear-space exposure gain (decode → mul → encode)
    so the saved chart contains boosted ACEScct values that match what the
    runtime app produces when DebugConfig.enable_post_ae_exposure_boost is on.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read digital TIFF: {path}")
    if img.ndim == 2 or img.shape[2] < 3:
        raise ValueError(f"Digital TIFF must be RGB: {path}")
    img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    if boost_ev != 0.0:
        linear = _acescct_decode(img)
        linear *= 2.0 ** boost_ev
        img = _acescct_encode(linear)
    return img


def _trailing_number(stem: str):
    """Return the trailing integer in `stem` as a string, or None."""
    m = re.search(r'(\d+)$', stem)
    return m.group(1) if m else None


def pair_files(film_dir: Path, digital_dir: Path):
    """Match film/digital files. Prefers exact stem match; if no exact
    matches are found, falls back to matching by the trailing integer in
    each stem (so 'film1.tif' pairs with 'digital1.tif', 'film_42.tif'
    with 'digital_42.tif', etc.)."""
    film_files = [p for p in film_dir.iterdir()
                  if p.suffix.lower() in FILM_EXTS]
    digital_files = [p for p in digital_dir.iterdir()
                     if p.suffix.lower() in DIGITAL_EXTS]
    film_by_stem = {p.stem: p for p in film_files}
    digital_by_stem = {p.stem: p for p in digital_files}
    common = sorted(set(film_by_stem) & set(digital_by_stem))

    if common:
        only_film = sorted(set(film_by_stem) - set(digital_by_stem))
        only_dig = sorted(set(digital_by_stem) - set(film_by_stem))
        if only_film:
            print(f"  ⚠ No digital match for: {', '.join(only_film)}")
        if only_dig:
            print(f"  ⚠ No film match for: {', '.join(only_dig)}")
        return [(s, film_by_stem[s], digital_by_stem[s]) for s in common]

    # Fallback: trailing-number pairing.
    film_by_num = {}
    digital_by_num = {}
    for p in film_files:
        n = _trailing_number(p.stem)
        if n is not None:
            film_by_num.setdefault(n, p)
    for p in digital_files:
        n = _trailing_number(p.stem)
        if n is not None:
            digital_by_num.setdefault(n, p)
    common_nums = sorted(set(film_by_num) & set(digital_by_num),
                         key=lambda x: int(x))
    if common_nums:
        print(f"  ℹ Pairing by trailing number ({len(common_nums)} pairs).")
    only_film = sorted(set(film_by_num) - set(digital_by_num), key=lambda x: int(x))
    only_dig = sorted(set(digital_by_num) - set(film_by_num), key=lambda x: int(x))
    if only_film:
        print(f"  ⚠ No digital match for film numbers: {', '.join(only_film)}")
    if only_dig:
        print(f"  ⚠ No film match for digital numbers: {', '.join(only_dig)}")
    # Zero-pad the chart key to the width of the largest number so output
    # files sort correctly in any browser and cross-reference cleanly to
    # the source filmN/digitalN.
    width = max((len(n) for n in common_nums), default=1)
    return [(n.zfill(width), film_by_num[n], digital_by_num[n])
            for n in common_nums]


# ---------------------------------------------------------------------------
# Flatness + sampling
# ---------------------------------------------------------------------------

def local_std(img: np.ndarray, win: int) -> np.ndarray:
    """Per-pixel local standard deviation, summed across channels (HxW).
    Suitable for either a full RGB image (H,W,3) or a 2-channel a*b* slab."""
    img = img.astype(np.float32)
    mean = cv2.boxFilter(img, ddepth=-1, ksize=(win, win))
    sq_mean = cv2.boxFilter(img * img, ddepth=-1, ksize=(win, win))
    var = np.maximum(sq_mean - mean * mean, 0.0)
    std = np.sqrt(var)
    return std.sum(axis=2) if std.ndim == 3 else std


def flatness_map(img: np.ndarray, win: int, metric: str, colorspace: str) -> np.ndarray:
    """Compute a per-pixel flatness map (HxW). Lower = flatter.

    metric='rgb':    std-dev of all 3 channels (current default).
    metric='chroma': std-dev of (a*, b*) only — ignores L, so a region with
                     consistent hue but light/shadow variation passes. This is
                     usually what you want for LUT chart sampling, since the
                     mean color of such a region is still meaningful for the
                     mapping (a flower bed is still pink even if some flowers
                     are in shade).
    """
    if metric == 'rgb':
        return local_std(img, win)
    if metric == 'chroma':
        lab = rgb_to_lab(img, colorspace)              # (H,W,3)
        ab = lab[:, :, 1:].astype(np.float32)           # (H,W,2)
        return local_std(ab, win)
    raise ValueError(f"Unknown flatness metric: {metric}")


def extract_patch(img: np.ndarray, y: int, x: int, win: int):
    """Return per-channel median, sample-window std-dev (sum of channels),
    and max pairwise quadrant disagreement (Euclidean distance between
    per-quadrant medians). All used to validate that the patch is genuinely
    a single solid color rather than a window straddling an edge.
    """
    h = win // 2
    patch = img[y - h:y + h, x - h:x + h]
    flat = patch.reshape(-1, patch.shape[-1])
    median = np.median(flat, axis=0)
    std = float(flat.std(axis=0).sum())

    # Quadrant medians — splits the window into 4 sub-windows.
    qs = [
        np.median(patch[:h, :h].reshape(-1, 3), axis=0),
        np.median(patch[:h, h:].reshape(-1, 3), axis=0),
        np.median(patch[h:, :h].reshape(-1, 3), axis=0),
        np.median(patch[h:, h:].reshape(-1, 3), axis=0),
    ]
    max_q_dist = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            d = float(np.linalg.norm(qs[i] - qs[j]))
            if d > max_q_dist:
                max_q_dist = d
    return median.astype(np.float32), std, max_q_dist


def hue_diff_deg(rgb_a: np.ndarray, rgb_b: np.ndarray,
                 cs_a: str, cs_b: str):
    """Per-sample (|hue_a - hue_b|, min_chroma) using per-side Lab."""
    la = rgb_to_lab(rgb_a, cs_a)
    lb = rgb_to_lab(rgb_b, cs_b)
    ha = np.degrees(np.arctan2(la[:, 2], la[:, 1]))
    hb = np.degrees(np.arctan2(lb[:, 2], lb[:, 1]))
    diff = np.abs(((ha - hb + 180.0) % 360.0) - 180.0)
    chroma = np.minimum(np.hypot(la[:, 1], la[:, 2]),
                        np.hypot(lb[:, 1], lb[:, 2]))
    return diff, chroma


def collect_candidates(film: np.ndarray, digital: np.ndarray,
                       patch_detect: int, patch_sample: int,
                       blur_sigma: float, border_frac: float,
                       flatness_pct: float, stride: int,
                       sample_flatness_pct: float = 30.0,
                       max_quad_disagreement: float = 0.04,
                       max_hue_diff_deg: float = 25.0,
                       hue_chroma_min: float = 8.0,
                       max_delta_mad: float = 3.0,
                       border_px: int = 0,
                       film_colorspace: str = 'srgb',
                       digital_colorspace: str = 'acescct_ap1',
                       flatness_metric: str = 'rgb',
                       detect_blur_sigma: float = -1.0):
    """Find patches that are flat in both images, then validate sample-window
    quality (flatness within the averaging window, quadrant agreement, and
    film/digital hue consistency). Returns list of (film_color, digital_color)
    plus a stats dict describing how many candidates were rejected at each
    stage — useful to audit the chart-building.
    """
    if film.shape[:2] != digital.shape[:2]:
        digital = cv2.resize(digital, (film.shape[1], film.shape[0]),
                             interpolation=cv2.INTER_LINEAR)

    # Pre-blur once for color extraction. Detecting on the un-blurred image
    # is dominated by film grain rather than scene structure.
    if blur_sigma > 0:
        film_b = cv2.GaussianBlur(film, (0, 0), sigmaX=blur_sigma)
        digital_b = cv2.GaussianBlur(digital, (0, 0), sigmaX=blur_sigma)
    else:
        film_b, digital_b = film, digital

    # Optionally apply a *stronger* blur just for the flatness detector. This
    # smooths out fabric/paint/skin texture that elevates chroma std even in
    # regions that have one consistent dominant hue — letting saturated-but-
    # textured content (awnings, painted walls, foliage) survive the gate.
    if detect_blur_sigma > 0 and detect_blur_sigma != blur_sigma:
        film_d = cv2.GaussianBlur(film, (0, 0), sigmaX=detect_blur_sigma)
        digital_d = cv2.GaussianBlur(digital, (0, 0), sigmaX=detect_blur_sigma)
    else:
        film_d, digital_d = film_b, digital_b

    std_f = flatness_map(film_d, patch_detect, flatness_metric, film_colorspace)
    std_d = flatness_map(digital_d, patch_detect, flatness_metric, digital_colorspace)

    h, w = film.shape[:2]
    if border_px > 0:
        by, bx = border_px, border_px
    else:
        by, bx = int(h * border_frac), int(w * border_frac)
    half = patch_sample // 2
    y0, y1 = max(by, half), min(h - by, h - half)
    x0, x1 = max(bx, half), min(w - bx, w - half)

    grid_y = np.arange(y0, y1, stride)
    grid_x = np.arange(x0, x1, stride)
    stats = {'detect_pass': 0, 'sample_flat_rejected': 0,
             'quad_rejected': 0, 'hue_rejected': 0,
             'delta_outlier_rejected': 0, 'kept': 0}
    if grid_y.size == 0 or grid_x.size == 0:
        return [], stats

    valid_f = std_f[y0:y1, x0:x1]
    valid_d = std_d[y0:y1, x0:x1]
    thr_f = np.percentile(valid_f, flatness_pct)
    thr_d = np.percentile(valid_d, flatness_pct)

    # First pass: detect-window flatness + per-window quality on both sides.
    raw = []  # (film_med, dig_med, film_std, dig_std, quad_max)
    for y in grid_y:
        for x in grid_x:
            if std_f[y, x] > thr_f or std_d[y, x] > thr_d:
                continue
            stats['detect_pass'] += 1
            fm, fs, fq = extract_patch(film_b, int(y), int(x), patch_sample)
            dm, ds, dq = extract_patch(digital_b, int(y), int(x), patch_sample)
            quad_max = max(fq, dq)
            raw.append((fm, dm, fs, ds, quad_max))

    if not raw:
        return [], stats

    # Sample-window flatness threshold: per-image low-percentile of the
    # window-std distribution, mirroring how detect-window flatness is set.
    f_stds = np.array([r[2] for r in raw])
    d_stds = np.array([r[3] for r in raw])
    thr_fs = np.percentile(f_stds, sample_flatness_pct)
    thr_ds = np.percentile(d_stds, sample_flatness_pct)

    out = []
    for fm, dm, fs, ds, qmax in raw:
        if fs > thr_fs or ds > thr_ds:
            stats['sample_flat_rejected'] += 1
            continue
        if qmax > max_quad_disagreement:
            stats['quad_rejected'] += 1
            continue
        out.append((fm, dm))

    # Hue-agreement filter (only above a chroma floor — neutrals have no hue).
    if out and max_hue_diff_deg < 180.0:
        film_arr = np.stack([o[0] for o in out])
        dig_arr = np.stack([o[1] for o in out])
        hd, chroma = hue_diff_deg(film_arr, dig_arr,
                                  film_colorspace, digital_colorspace)
        keep = (chroma < hue_chroma_min) | (hd <= max_hue_diff_deg)
        stats['hue_rejected'] = int((~keep).sum())
        out = [o for o, k in zip(out, keep) if k]

    # Delta-consistency filter: within a pair, film/digital color deltas should
    # cluster around a single relationship (the LUT mapping). Patches whose
    # delta is far from the median are likely parallax-induced foreground/
    # background mismatches — flat in both, but on different objects.
    if out and len(out) >= 6 and max_delta_mad > 0:
        film_arr = np.stack([o[0] for o in out])
        dig_arr = np.stack([o[1] for o in out])
        delta_lab = (rgb_to_lab(film_arr, film_colorspace)
                     - rgb_to_lab(dig_arr, digital_colorspace))
        center = np.median(delta_lab, axis=0)
        residual = np.linalg.norm(delta_lab - center, axis=1)
        mad = np.median(np.abs(residual - np.median(residual))) + 1e-6
        keep = residual <= max_delta_mad * 1.4826 * mad + np.median(residual)
        stats['delta_outlier_rejected'] = int((~keep).sum())
        out = [o for o, k in zip(out, keep) if k]

    stats['kept'] = len(out)
    return out, stats


# ---------------------------------------------------------------------------
# Lab + farthest-point sampling
# ---------------------------------------------------------------------------

def rgb_to_lab(rgb01: np.ndarray, colorspace: str) -> np.ndarray:
    """Convert (..., 3) RGB float [0,1] in the given encoding to CIELAB.
    Reshapes back to the input shape so the same function works for both
    flat color arrays and full (H,W,3) images.
    """
    in_shape = rgb01.shape
    flat = np.clip(rgb01.reshape(-1, 3), 0.0, 1.0).astype(np.float32)
    if colorspace == 'srgb':
        u8 = (flat * 255.0 + 0.5).astype(np.uint8).reshape(1, -1, 3)
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(np.float32)
        lab[:, 0] *= 100.0 / 255.0
        lab[:, 1] -= 128.0
        lab[:, 2] -= 128.0
    elif colorspace == 'rec2020_g24':
        linear = np.power(flat, 2.4)
        cs = colour.RGB_COLOURSPACES['ITU-R BT.2020']
        xyz = colour.RGB_to_XYZ(linear, cs, apply_cctf_decoding=False)
        lab = colour.XYZ_to_Lab(xyz, illuminant=cs.whitepoint).astype(np.float32)
    elif colorspace == 'acescct_rec2020':
        linear = _acescct_decode(flat)
        cs = colour.RGB_COLOURSPACES['ITU-R BT.2020']
        xyz = colour.RGB_to_XYZ(linear, cs, apply_cctf_decoding=False)
        lab = colour.XYZ_to_Lab(xyz, illuminant=cs.whitepoint).astype(np.float32)
    elif colorspace == 'acescct_ap1':
        # True ACEScct: log AP1 -> linear AP1 -> XYZ_D60 -> Lab. Matches what
        # processor_v2's TIFF export writes.
        linear = _acescct_decode(flat)
        cs = colour.RGB_COLOURSPACES['ACEScg']
        xyz = colour.RGB_to_XYZ(linear, cs, apply_cctf_decoding=False)
        lab = colour.XYZ_to_Lab(xyz, illuminant=cs.whitepoint).astype(np.float32)
    else:
        raise ValueError(f"Unknown colorspace: {colorspace}")
    return lab.reshape(in_shape)


def farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    """Greedy FPS in Euclidean space. Returns indices into `points`."""
    k = points.shape[0]
    if k <= n:
        return np.arange(k)
    # Seed with the point closest to the median (a "central" anchor).
    median = np.median(points, axis=0)
    seed = int(np.argmin(np.linalg.norm(points - median, axis=1)))
    selected = [seed]
    min_d2 = np.sum((points - points[seed]) ** 2, axis=1)
    for _ in range(n - 1):
        idx = int(np.argmax(min_d2))
        selected.append(idx)
        d2 = np.sum((points - points[idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    return np.array(selected)


def kmeans_select(points: np.ndarray, n: int) -> np.ndarray:
    """Partition `points` into N k-means clusters and return one medoid per
    cluster. Better than FPS at balancing dense vs sparse regions of the
    candidate cloud — guarantees that no single cluster gets over-represented.
    Per-axis normalization equalizes contribution of L vs a* vs b*.
    """
    from scipy.cluster.vq import kmeans2
    k = points.shape[0]
    if k <= n:
        return np.arange(k)
    scale = points.std(axis=0) + 1e-6
    norm = points / scale
    # Multiple restarts so we don't get stuck in a bad local minimum.
    best_labels = None
    best_centroids = None
    best_inertia = np.inf
    for seed in range(5):
        centroids, labels = kmeans2(norm, k=n, minit='++', seed=seed)
        inertia = float(np.sum((norm - centroids[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids
    selected = []
    used = set()
    for i in range(n):
        mask = best_labels == i
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        d = np.linalg.norm(norm[idxs] - best_centroids[i], axis=1)
        chosen = int(idxs[np.argmin(d)])
        selected.append(chosen)
        used.add(chosen)
    # Cover any empty clusters by FPS-padding from remaining points.
    if len(selected) < n:
        remaining = np.array(sorted(set(range(k)) - used))
        if len(remaining) > 0:
            need = n - len(selected)
            extras = farthest_point_sample(points[remaining], need)
            selected.extend(remaining[extras].tolist())
    return np.array(selected[:n])


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------

def order_for_grid(lab: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Sort by L into rows (dark→light), then by hue within each row."""
    order_l = np.argsort(lab[:, 0])
    out = np.empty(rows * cols, dtype=np.int64)
    for r in range(rows):
        chunk = order_l[r * cols:(r + 1) * cols]
        hue = np.arctan2(lab[chunk, 2], lab[chunk, 1])
        out[r * cols:(r + 1) * cols] = chunk[np.argsort(hue)]
    return out


def render_chart(colors: np.ndarray, rows: int, cols: int, cell_px: int) -> np.ndarray:
    """Render an (rows*cell_px, cols*cell_px, 3) chart of solid swatches."""
    H, W = rows * cell_px, cols * cell_px
    out = np.zeros((H, W, colors.shape[1]), dtype=np.float32)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        out[r * cell_px:(r + 1) * cell_px, c * cell_px:(c + 1) * cell_px] = colors[i]
    return out


# IMWRITE_TIFF_COMPRESSION=1 → no compression (colormatch chokes on LZW).
_TIFF_PARAMS = [int(cv2.IMWRITE_TIFF_COMPRESSION), 1]


def save_film_chart(chart: np.ndarray, path: Path) -> None:
    """Save as 16-bit sRGB TIFF, uncompressed."""
    u16 = np.clip(chart * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), cv2.cvtColor(u16, cv2.COLOR_RGB2BGR), _TIFF_PARAMS)


def save_digital_chart(chart: np.ndarray, path: Path) -> None:
    """Save as 16-bit TIFF preserving raw ACEScct/Rec.2020 values, uncompressed."""
    u16 = np.clip(chart * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), cv2.cvtColor(u16, cv2.COLOR_RGB2BGR), _TIFF_PARAMS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_grid(s: str):
    a, b = s.lower().split('x')
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--film-dir', type=Path, required=True)
    ap.add_argument('--digital-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--grid', type=parse_grid, default=(8, 6),
                    help='cols x rows, e.g. 8x6 (default 8x6 → 48 swatches)')
    ap.add_argument('--border-frac', type=float, default=0.12)
    ap.add_argument('--border-px', type=int, default=100,
                    help='absolute border (in pixels) to ignore on each edge; '
                         'overrides --border-frac when > 0')
    ap.add_argument('--patch-detect', type=int, default=16)
    ap.add_argument('--patch-sample', type=int, default=12)
    ap.add_argument('--blur-sigma', type=float, default=1.5,
                    help='Gaussian blur sigma applied for color extraction')
    ap.add_argument('--detect-blur-sigma', type=float, default=5.0,
                    help='separate blur sigma for the flatness DETECTOR; '
                         'higher = more permissive on textured content '
                         '(e.g. fabric, paint). -1 = use --blur-sigma value.')
    ap.add_argument('--flatness-pct', type=float, default=50.0,
                    help='per-image percentile of local std-dev that defines "flat"')
    ap.add_argument('--cell-px', type=int, default=80)
    cs_choices = ['srgb', 'rec2020_g24', 'acescct_rec2020', 'acescct_ap1']
    ap.add_argument('--film-colorspace', choices=cs_choices,
                    default='acescct_ap1',
                    help='encoding of the FILM images, used for perceptual '
                         'metrics on the film side. Pixel values themselves '
                         'are passed through untouched. Default acescct_ap1 '
                         'matches v2 film export; use srgb for legacy 8-bit '
                         'JPEG film.')
    ap.add_argument('--digital-colorspace', choices=cs_choices,
                    default='acescct_ap1',
                    help='encoding of the DIGITAL TIFFs, used for perceptual '
                         'metrics on the digital side. Default acescct_ap1 '
                         'matches processor_v2 TIFF output. Use '
                         'acescct_rec2020 for v1 TIFFs.')
    ap.add_argument('--flatness-metric', choices=['rgb', 'chroma'],
                    default='chroma',
                    help='"chroma" (default): std-dev on (a*, b*) only — keeps '
                         'regions with consistent hue even if lighting varies. '
                         '"rgb": std-dev on all 3 channels — biases against '
                         'chromatic content.')
    ap.add_argument('--exposure-boost-ev', type=float, default=0.0,
                    help='linear-space EV boost applied to digital pixels '
                         'before sampling. Default 0 — leave this at 0 if '
                         'the digital TIFFs were exported from processor_v2 '
                         'with enable_post_ae_exposure_boost ON (the boost '
                         'is already baked in). Set it to '
                         f'POST_AE_EXPOSURE_BOOST_EV ({POST_AE_EXPOSURE_BOOST_EV}) '
                         'only if you exported the TIFFs *without* the boost.')
    ap.add_argument('--stride', type=int, default=None,
                    help='spacing between candidate centers (default: patch_sample/2)')
    ap.add_argument('--sample-flatness-pct', type=float, default=70.0,
                    help='per-image percentile of *sample-window* std-dev to '
                         'keep — second-stage flatness gate (default 70)')
    ap.add_argument('--max-quad-disagreement', type=float, default=0.04,
                    help='reject candidate if max distance between its 4 '
                         'quadrant medians exceeds this (default 0.04)')
    ap.add_argument('--max-hue-diff-deg', type=float, default=25.0,
                    help='reject candidate if film vs digital hue differ by '
                         'more than this many degrees (default 25)')
    ap.add_argument('--hue-chroma-min', type=float, default=8.0,
                    help='hue check is skipped below this Lab chroma '
                         '(default 8.0 — neutrals have no meaningful hue)')
    ap.add_argument('--max-delta-mad', type=float, default=3.0,
                    help='per-pair: reject candidates whose film/digital Lab '
                         'delta is more than N robust-MADs from the pair '
                         'median (catches parallax-swap outliers; default 3)')
    ap.add_argument('--report', action='store_true',
                    help='print per-pair candidate funnel (kept vs rejected)')
    ap.add_argument('--only', type=str, default='',
                    help='comma-separated list of pair keys to process '
                         '(e.g. "5" or "3,7,12"). Matches against either the '
                         'chart key (zero-padded number) or its int value, so '
                         '"--only 5" finds pair "05". Empty = all pairs.')
    args = ap.parse_args()

    cols, rows = args.grid
    n = rows * cols
    stride = args.stride or max(args.patch_sample // 2, 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pair_files(args.film_dir, args.digital_dir)
    if not pairs:
        print("No matching pairs found.")
        sys.exit(1)

    if args.only.strip():
        wanted = {t.strip() for t in args.only.split(',') if t.strip()}
        def _matches(stem):
            if stem in wanted:
                return True
            try:
                return str(int(stem)) in wanted
            except ValueError:
                return False
        pairs = [p for p in pairs if _matches(p[0])]
        if not pairs:
            print(f"No pairs match --only {args.only!r}.")
            sys.exit(1)

    print(f"Processing {len(pairs)} pair(s). Grid: {cols}×{rows} = {n} swatches.")

    for stem, film_path, dig_path in pairs:
        print(f"\n[{stem}]")
        film = load_film(film_path)
        digital = load_digital(dig_path, boost_ev=args.exposure_boost_ev)

        cands, stats = collect_candidates(
            film, digital,
            patch_detect=args.patch_detect,
            patch_sample=args.patch_sample,
            blur_sigma=args.blur_sigma,
            border_frac=args.border_frac,
            flatness_pct=args.flatness_pct,
            stride=stride,
            sample_flatness_pct=args.sample_flatness_pct,
            max_quad_disagreement=args.max_quad_disagreement,
            max_hue_diff_deg=args.max_hue_diff_deg,
            hue_chroma_min=args.hue_chroma_min,
            max_delta_mad=args.max_delta_mad,
            border_px=args.border_px,
            film_colorspace=args.film_colorspace,
            digital_colorspace=args.digital_colorspace,
            flatness_metric=args.flatness_metric,
            detect_blur_sigma=args.detect_blur_sigma,
        )
        if args.report:
            print(f"  Funnel: detect-pass={stats['detect_pass']}  "
                  f"sample-flat-rejected={stats['sample_flat_rejected']}  "
                  f"quad-rejected={stats['quad_rejected']}  "
                  f"hue-rejected={stats['hue_rejected']}  "
                  f"delta-outlier-rejected={stats['delta_outlier_rejected']}  "
                  f"kept={stats['kept']}")
        else:
            print(f"  {len(cands)} validated candidates "
                  f"(of {stats['detect_pass']} flat-in-both)")
        if len(cands) == 0:
            print(f"  ⚠ Skipping {stem}: no patches survived validation")
            continue

        film_colors = np.stack([c[0] for c in cands])
        dig_colors = np.stack([c[1] for c in cands])
        lab = rgb_to_lab(film_colors, args.film_colorspace)

        if len(cands) < n:
            print(f"  ⚠ Only {len(cands)} candidates < {n}; padding with repeats")
            sel = np.concatenate([
                np.arange(len(cands)),
                np.random.choice(len(cands), n - len(cands), replace=True),
            ])
        else:
            sel = kmeans_select(lab, n)

        film_sel = film_colors[sel]
        dig_sel = dig_colors[sel]
        lab_sel = rgb_to_lab(film_sel, args.film_colorspace)
        order = order_for_grid(lab_sel, rows, cols)

        film_chart = render_chart(film_sel[order], rows, cols, args.cell_px)
        dig_chart = render_chart(dig_sel[order], rows, cols, args.cell_px)

        save_film_chart(film_chart, args.out_dir / f"{stem}_film.tif")
        save_digital_chart(dig_chart, args.out_dir / f"{stem}_digital.tif")
        print(f"  ✓ {stem}_film.tif + {stem}_digital.tif")

    print(f"\nDone. Charts written to {args.out_dir}")


if __name__ == '__main__':
    main()
