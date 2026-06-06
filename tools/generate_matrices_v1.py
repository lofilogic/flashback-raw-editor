"""Generate the V1 negative colour matrix + ASN from a ColorChecker shot.

The V1 twin of tools/generate_matrices_from_colorchart.py. Identical maths
(lstsq camera_wb -> XYZ_D50, D50 white-point normalisation), but the input is
a headerless V1 negative read + demosaiced through core.v1_negative instead of
a DNG read through rawpy. White balance (ASN) is taken from the ColorChecker's
neutral patches — no separate flat field needed.

Built for the V1 camera's real constraints (full-auto point-and-shoot, fixed
infinity focus, no manual exposure):

  * Grid mode (default): click only the 4 corner patch centres; the 6x4 grid is
    interpolated and each patch is sampled from a small *centre* window — robust
    to out-of-focus bleed between patches and to a chart that doesn't fill the
    frame.
  * Auto clip-rejection: patches with any channel at/above --clip are dropped
    from the fit, and clipped neutrals are excluded from the WB average — so
    frames you couldn't expose darker still calibrate from their good patches.

Usage:
    python -m tools.generate_matrices_v1 path/to/<uuid>.raw          # grid (4 clicks)
    python -m tools.generate_matrices_v1 path/to/<uuid>.raw --manual # 24 clicks
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from core.v1_negative import linear_rgb, V1_BLACK_LEVEL

parser = argparse.ArgumentParser(description="V1 ColorChecker -> matrix + ASN.")
parser.add_argument("raw", help="Path to a V1 negative (.raw / .json / extensionless).")
parser.add_argument("--illuminant", choices=["d50", "tungsten"], default="d50",
                    help="Label only; the in-app path uses the d50 matrix.")
parser.add_argument("--black", type=float, default=V1_BLACK_LEVEL,
                    help=f"Black pedestal in code values (default {V1_BLACK_LEVEL}).")
parser.add_argument("--manual", action="store_true",
                    help="Click all 24 patch centres instead of 4 corners.")
parser.add_argument("--sample-frac", type=float, default=0.22,
                    help="Patch centre window as a fraction of patch spacing "
                         "(smaller = safer against out-of-focus / small-chart bleed).")
parser.add_argument("--clip", type=float, default=0.985,
                    help="Patches with any channel >= this are treated as clipped.")
args = parser.parse_args()

ILLUMINANT_TAG = {"d50": 23, "tungsten": 17}
ILLUMINANT_LABEL = {"d50": "D50 (tag 23)", "tungsten": "Std Light A / Tungsten (tag 17)"}
idx = {"d50": 1, "tungsten": 2}[args.illuminant]
COLS, ROWS = 6, 4  # ColorChecker Classic layout

# ColorChecker Classic 24, CIE XYZ under D50 (0-1) — same table as the V2 tool.
REF_XYZ_D50 = np.array([
    [0.1150, 0.1009, 0.0336], [0.3920, 0.3582, 0.2505], [0.1834, 0.1927, 0.3347],
    [0.1086, 0.1345, 0.0639], [0.2644, 0.2452, 0.4485], [0.3168, 0.4357, 0.4411],
    [0.3797, 0.3013, 0.0526], [0.1417, 0.1200, 0.3541], [0.2936, 0.1963, 0.1287],
    [0.0867, 0.0655, 0.1171], [0.3583, 0.4443, 0.1165], [0.4729, 0.4326, 0.0768],
    [0.0861, 0.0583, 0.2185], [0.1506, 0.2359, 0.0767], [0.2039, 0.1193, 0.0381],
    [0.5739, 0.6033, 0.1044], [0.2987, 0.1947, 0.3477], [0.1450, 0.1952, 0.3794],
    [0.9069, 0.9634, 0.7951], [0.5898, 0.6277, 0.5215], [0.3622, 0.3848, 0.3207],
    [0.1952, 0.2069, 0.1741], [0.0898, 0.0950, 0.0805], [0.0322, 0.0332, 0.0287]
])
NEUTRALS = [19, 20, 21, 22]  # Neutral 8 / 6.5 / 5 / 3.5 (skip clip-prone white #18)


def sample(image, cx, cy, half):
    """Per-channel MEDIAN of a small centre window. Median (not mean) rejects
    the dark inter-patch grid lines and neighbour bleed that creep into the
    window when the chart is small and out of focus — those are low-side
    outliers a mean would fold into the patch colour and corrupt the fit."""
    half = max(1, int(half))
    x, y = int(round(cx)), int(round(cy))
    win = image[y - half:y + half, x - half:x + half].reshape(-1, 3)
    return np.median(win, axis=0)


def grid_centres(corners):
    """Bilinear-interpolate 24 patch centres from 4 corner centres, clicked
    TL(#0) -> TR(#5) -> BR(#23) -> BL(#18). Returns (centres[24,2], spacing)."""
    tl, tr, br, bl = (np.asarray(c, float) for c in corners)
    pts = []
    for r in range(ROWS):
        v = r / (ROWS - 1)
        for c in range(COLS):
            u = c / (COLS - 1)
            top = tl + (tr - tl) * u
            bot = bl + (br - bl) * u
            pts.append(top + (bot - top) * v)
    pts = np.array(pts)
    spacing = min(np.linalg.norm(tr - tl) / (COLS - 1),
                  np.linalg.norm(bl - tl) / (ROWS - 1))
    return pts, spacing


# 1. Read + demosaic the V1 negative (full res, no dither — patch averaging
#    removes noise; same CFA/black as the develop path).
img = linear_rgb(args.raw, black=args.black, dither_lsb=0.0,
                 target_long_edge=None, edge_aware=False)
disp = np.power(np.clip(img, 0, 1), 1 / 2.2)

# 2. Pick patches.
if args.manual:
    print("Click 24 patch centres: Dark Skin (top-left) -> left-to-right, top-to-bottom.")
    plt.imshow(disp); pts = plt.ginput(24, timeout=0); plt.close()
    if len(pts) != 24:
        raise ValueError(f"Need 24 points, got {len(pts)}.")
    centres = np.array(pts)
    # Estimate patch spacing from the clicked grid (median nearest-neighbour
    # distance) so the window scales to a small chart instead of a fixed,
    # too-large fraction of the frame.
    d = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    np.fill_diagonal(d, np.inf)
    spacing = float(np.median(d.min(axis=1)))
else:
    print("Click the 4 CORNER patch centres in order: "
          "Dark Skin (top-left) -> top-right -> bottom-right -> bottom-left.")
    plt.imshow(disp); corners = plt.ginput(4, timeout=0); plt.close()
    if len(corners) != 4:
        raise ValueError(f"Need 4 corner points, got {len(corners)}.")
    centres, spacing = grid_centres(corners)

half = max(2, spacing * args.sample_frac / 2.0)
print(f"Patch spacing ~{spacing:.0f}px, sampling a {2*half:.0f}px median window "
      f"({100*args.sample_frac:.0f}% of spacing).")
camera_rgb = np.array([sample(img, x, y, half) for (x, y) in centres])

# 3. Clip-rejection. Drop clipped patches from the fit; exclude clipped neutrals
#    from the WB average — frames you couldn't expose darker still calibrate.
clipped = camera_rgb.max(axis=1) >= args.clip
if clipped.any():
    print(f"Clipped patches (dropped from fit): {np.where(clipped)[0].tolist()}")

good_neutrals = [i for i in NEUTRALS if not clipped[i]]
if not good_neutrals:
    raise ValueError("All neutral patches clipped — cannot derive WB. Reshoot darker "
                     "(try a brighter surround so AE pulls exposure down).")
grey_avg = camera_rgb[good_neutrals].mean(axis=0)
ASN = grey_avg / grey_avg[1]
print(f"WB from neutrals {good_neutrals}: ASN = [{ASN[0]:.7f}, 1.0, {ASN[2]:.7f}]")

# 4. ForwardMatrix from the unclipped patches, by **white-point-constrained**
#    least squares: for each XYZ row m_i, minimise ||cam_wb @ m_i - ref_i||^2
#    subject to sum(m_i) = wp_i (a WB-neutral maps exactly to the D50 white
#    point). Building the constraint into the solve avoids the old fit-then-
#    rescale-rows trick, which inflated coefficients into non-physical territory
#    (e.g. negative red luminance -> dark reds).
keep = ~clipped
if keep.sum() < 6:
    raise ValueError(f"Only {keep.sum()} unclipped patches — too few for a stable fit.")
cam_wb = camera_rgb[keep] * (1.0 / ASN)
d50_wp = np.array([0.9642, 1.0000, 0.8249])


def constrained_column(A, b, s):
    """argmin_m ||A m - b||^2  s.t.  sum(m) = s, via the KKT system."""
    AtA = A.T @ A
    ones = np.ones((3, 1))
    kkt = np.block([[2 * AtA, ones], [ones.T, np.zeros((1, 1))]])
    rhs = np.concatenate([2 * A.T @ b, [s]])
    return np.linalg.solve(kkt, rhs)[:3]


ForwardMatrix = np.array([constrained_column(cam_wb, REF_XYZ_D50[keep, i], d50_wp[i])
                          for i in range(3)])

# 5. Sanity / residual + per-patch diagnostics (worst offenders flag mis-clicks
#    or out-of-focus bleed — reshoot/reclick those rather than trust the fit).
fit = cam_wb @ ForwardMatrix.T
per_patch = np.sqrt(np.mean((fit - REF_XYZ_D50[keep]) ** 2, axis=1))
rms = float(np.sqrt(np.mean(per_patch ** 2)))
neutral = ForwardMatrix @ np.array([1.0, 1.0, 1.0])
y_resp = ForwardMatrix[1]  # luminance row; all-positive == physical
kept_idx = np.where(keep)[0]
worst = kept_idx[np.argsort(per_patch)[::-1][:4]]
print(f"\n--- {ILLUMINANT_LABEL[args.illuminant]} | {int(keep.sum())} patches | "
      f"fit RMS(XYZ)={rms:.4f} | neutral->{neutral.round(4)} (target 0.964,1.0,0.825) ---")
print(f"Luminance(Y) weights R,G,B = {y_resp.round(3)}  "
      f"{'OK' if (y_resp >= 0).all() else 'WARNING: negative weight -> that primary renders too dark'}")
print(f"Worst patches (idx:rms): " +
      ", ".join(f"{i}:{per_patch[kept_idx == i][0]:.3f}" for i in worst))
if rms > 0.05:
    print("High RMS — check patch order/orientation when clicking, and that the "
          "chart isn't clipped or badly out of focus.")

print("\n# ---- paste into core/v1_negative.py ----")
print(f"V1_ASN_D50 = np.array([{ASN[0]:.6f}, 1.0, {ASN[2]:.6f}], dtype=np.float32)")
print("V1_FORWARD_MATRIX = np.array([")
for r in ForwardMatrix:
    print(f"    [{r[0]: .6f}, {r[1]: .6f}, {r[2]: .6f}],")
print(f"], dtype=np.float32)  # illuminant={args.illuminant} fit_rms={rms:.4f} "
      f"n={int(keep.sum())}")
