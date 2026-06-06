"""Diagnose whether V1 "negative" data is LINEAR or tone-curve/gamma encoded.

A linear colour matrix can only fit linear data. If the camera bakes an OETF
into the 8-bit export, the ColorChecker fit blows up (high RMS, non-physical
matrix with negative luminance weights). This checks that directly.

Click the 4 corner patch centres (Dark Skin TL -> TR -> BR -> BL). Then, with
no further input, it writes /tmp/v1_chart_diag.txt with:
  * the 24 sampled patch RGBs (saved to /tmp/v1_patches.npy too),
  * a NEUTRAL-RAMP linearity test: log-log slope of measured vs reference Y
    over the unclipped greys (slope ~1.0 => linear; ~0.45 => ~gamma 2.2),
  * constrained-fit RMS + luminance row under three decodes: linear (as-is),
    pure gamma 2.2, and sRGB EOTF — whichever collapses RMS is the encoding.

Usage:
    python -m tools.diagnose_v1_chart /path/to/<chart raw>
"""
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from core.v1_negative import linear_rgb, V1_BLACK_LEVEL

COLS, ROWS = 6, 4
WP = np.array([0.9642, 1.0000, 0.8249])
REF_XYZ_D50 = np.array([
    [0.1150, 0.1009, 0.0336], [0.3920, 0.3582, 0.2505], [0.1834, 0.1927, 0.3347],
    [0.1086, 0.1345, 0.0639], [0.2644, 0.2452, 0.4485], [0.3168, 0.4357, 0.4411],
    [0.3797, 0.3013, 0.0526], [0.1417, 0.1200, 0.3541], [0.2936, 0.1963, 0.1287],
    [0.0867, 0.0655, 0.1171], [0.3583, 0.4443, 0.1165], [0.4729, 0.4326, 0.0768],
    [0.0861, 0.0583, 0.2185], [0.1506, 0.2359, 0.0767], [0.2039, 0.1193, 0.0381],
    [0.5739, 0.6033, 0.1044], [0.2987, 0.1947, 0.3477], [0.1450, 0.1952, 0.3794],
    [0.9069, 0.9634, 0.7951], [0.5898, 0.6277, 0.5215], [0.3622, 0.3848, 0.3207],
    [0.1952, 0.2069, 0.1741], [0.0898, 0.0950, 0.0805], [0.0322, 0.0332, 0.0287]])
NEUTRALS = [18, 19, 20, 21, 22, 23]  # white -> black ramp
CLIP = 0.985

ap = argparse.ArgumentParser()
ap.add_argument("raw")
ap.add_argument("--black", type=float, default=V1_BLACK_LEVEL)
args = ap.parse_args()


def grid_centres(corners):
    tl, tr, br, bl = (np.asarray(c, float) for c in corners)
    pts, = ([tl + (tr - tl) * (c / 5) + (bl + (br - bl) * (c / 5) - (tl + (tr - tl) * (c / 5))) * (r / 3)
             for r in range(ROWS) for c in range(COLS)],)
    spacing = min(np.linalg.norm(tr - tl) / 5, np.linalg.norm(bl - tl) / 3)
    return np.array(pts), spacing


def sample(img, c, half):
    half = max(1, int(half)); x, y = int(round(c[0])), int(round(c[1]))
    return np.median(img[y - half:y + half, x - half:x + half].reshape(-1, 3), 0)


def constrained_fit(cam_wb, ref):
    M = []
    for i in range(3):
        AtA = cam_wb.T @ cam_wb; ones = np.ones((3, 1))
        kkt = np.block([[2 * AtA, ones], [ones.T, np.zeros((1, 1))]])
        M.append(np.linalg.solve(kkt, np.concatenate([2 * cam_wb.T @ ref[:, i], [WP[i]]]))[:3])
    return np.array(M)


def fit_under(decode, cam_raw):
    cam = decode(cam_raw)
    clipped = cam_raw.max(1) >= CLIP            # clip judged on the encoded data
    keep = ~clipped
    gn = [i for i in NEUTRALS if not clipped[i] and i >= 19]
    asn = cam[gn].mean(0); asn = asn / asn[1]
    cam_wb = cam[keep] / asn
    M = constrained_fit(cam_wb, REF_XYZ_D50[keep])
    rms = float(np.sqrt(np.mean((cam_wb @ M.T - REF_XYZ_D50[keep]) ** 2)))
    return rms, M, asn, np.where(clipped)[0]


def srgb_eotf(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


img = linear_rgb(args.raw, black=args.black, dither_lsb=0.0, target_long_edge=None, edge_aware=False)
print("Click 4 CORNER patch centres: Dark Skin (TL) -> TR -> BR -> BL.")
plt.imshow(np.power(np.clip(img, 0, 1), 1 / 2.2))
corners = plt.ginput(4, timeout=0); plt.close()
if len(corners) != 4:
    sys.exit(f"Need 4 points, got {len(corners)}")
centres, spacing = grid_centres(corners)
cam_raw = np.array([sample(img, c, spacing * 0.22 / 2) for c in centres])
np.save("/tmp/v1_patches.npy", cam_raw)

L = [f"chart: {args.raw}", f"spacing ~{spacing:.0f}px", ""]
L.append("24 patch RGB (black-subtracted, normalised):")
for i, v in enumerate(cam_raw):
    L.append(f"  {i:2d}: [{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}]")

# Neutral-ramp linearity: measured green vs reference Y over unclipped greys.
L += ["", "NEUTRAL RAMP (measured G vs reference Y):"]
gy = []
for i in NEUTRALS:
    clipped = cam_raw[i].max() >= CLIP
    L.append(f"  patch {i}: G_meas={cam_raw[i,1]:.4f} refY={REF_XYZ_D50[i,1]:.4f}"
             f"{'  [CLIPPED]' if clipped else ''}")
    if not clipped and cam_raw[i, 1] > 1e-4:
        gy.append((REF_XYZ_D50[i, 1], cam_raw[i, 1]))
if len(gy) >= 2:
    refY, g = np.array(gy).T
    slope = np.polyfit(np.log(refY), np.log(g), 1)[0]   # 1/gamma if OETF present
    L.append(f"  log-log slope = {slope:.3f}  => implied encoding gamma ~ {1/slope:.2f}  "
             f"({'LINEAR' if abs(slope-1) < 0.12 else 'NON-LINEAR (tone curve baked in)'})")

# Constrained fit under three decodes.
L += ["", "CONSTRAINED FIT under candidate decodes:"]
for name, dec in [("linear (as-is)", lambda x: x),
                  ("gamma 2.2     ", lambda x: np.clip(x, 0, None) ** 2.2),
                  ("sRGB EOTF     ", srgb_eotf)]:
    rms, M, asn, clp = fit_under(dec, cam_raw)
    yrow = M[1]
    L.append(f"  {name}: RMS={rms:.4f}  Y(R,G,B)={yrow.round(3)} "
             f"{'OK' if (yrow >= 0).all() else 'NEG->dark'}  ASN=[{asn[0]:.3f},1,{asn[2]:.3f}]")

open("/tmp/v1_chart_diag.txt", "w").write("\n".join(L))
print("\n".join(L))
print("\nWrote /tmp/v1_chart_diag.txt and /tmp/v1_patches.npy")
