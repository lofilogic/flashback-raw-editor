"""
Fit a 3x3 CCM that maps a third-party camera's raw RGB to the Flashback's
linear-sRGB output, using a ColorChecker shot taken under the same lighting
with both cameras.

Usage:
    python tools/match_camera.py <flashback.dng> <other.dng> [--out assets/other_ccm.npy]

Workflow:
  1. Both DNGs are developed with output_color=raw, gamma=(1,1), no_auto_bright.
     Flashback uses BASE_WB_SETTINGS + SENSOR_BLACK and then FLASHBACK_CCM is
     applied -> target linear-sRGB patch values (including Flashback's color
     signature).
     The "other" camera is developed with its daylight_whitebalance, no CCM.
  2. You click the 24 patch centers in each image, row-major
     (left-to-right, then top-to-bottom). Orientation/rotation may differ
     between shots - click them in the same semantic order in both images.
  3. A 3x3 matrix M is solved via least squares so that
         other_raw @ M.T  ~=  flashback_linear_srgb
     and printed + saved.

Controls in the click window:
  left-click   add a patch center (auto-advances)
  u            undo last point
  r            reset all points
  Enter/Space  confirm when 24 points are placed
  q / Esc      abort
"""
import argparse
import os

import numpy as np
import rawpy
import cv2

# Mirror of core/config.py values. Duplicated here so this tool doesn't
# trigger the package's numba/llvmlite import chain. Keep in sync if those
# constants change.
FLASHBACK_CCM = np.array([
    [ 3.8045148 , -0.40716213,  0.03187762],
    [-0.45492041,  0.73636414,  0.02067507],
    [ 0.11892583, -0.55000283,  2.91999937],
])
BASE_WB_SETTINGS = [0.5, 1.0, 0.61, 1.0]
SENSOR_BLACK = 64


PATCH_COUNT = 24
GRID_ROWS = 4
GRID_COLS = 6
# Patch sample box is a fraction of the shorter image dim
SAMPLE_FRAC = 0.006  # ~0.6% -> generous on a handheld chart; uses median


# ---------------------------------------------------------------------------
# RAW development
# ---------------------------------------------------------------------------

def develop_flashback(dng_path: str) -> np.ndarray:
    """Develop a Flashback DNG up through FLASHBACK_CCM -> linear sRGB."""
    with rawpy.imread(dng_path) as raw:
        rgb = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            user_wb=BASE_WB_SETTINGS,
            user_black=SENSOR_BLACK,
            half_size=False,
            no_auto_bright=True,
            bright=0.5,  # MUST match pipeline (core/processor.py Flashback branch)
            highlight_mode=1,
            gamma=(1, 1),
            output_bps=16,
            output_color=rawpy.ColorSpace.raw,
        ).astype(np.float32) / 65535.0
    # Apply Flashback CCM -> linear sRGB (this is the target color space)
    flat = rgb.reshape(-1, 3) @ FLASHBACK_CCM.T
    return flat.reshape(rgb.shape)


def develop_other_raw(dng_path: str) -> np.ndarray:
    """Develop a non-Flashback DNG with the camera's daylight WB pre-applied.

    `raw.daylight_whitebalance` is derived from the DNG's ColorMatrix /
    CalibrationIlluminant tags and is constant per (camera model, firmware).
    Pre-applying it balances the channels so the fitted CCM only has to do
    color rotation, matching the Flashback pipeline's
    (BASE_WB_SETTINGS -> FLASHBACK_CCM) architecture.
    """
    with rawpy.imread(dng_path) as raw:
        rgb = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            user_wb=list(raw.daylight_whitebalance),
            half_size=False,
            no_auto_bright=True,
            bright=1.0,  # MUST match pipeline (core/processor.py non-Flashback branch)
            highlight_mode=1,
            gamma=(1, 1),
            output_bps=16,
            output_color=rawpy.ColorSpace.raw,
        ).astype(np.float32) / 65535.0
    return rgb


# ---------------------------------------------------------------------------
# Interactive patch picker
# ---------------------------------------------------------------------------

def _tonemap_for_display(linear: np.ndarray) -> np.ndarray:
    """Gamma + gentle auto-exposure so the linear image is visible for clicking.
    Display only; sampling uses the linear data.
    """
    img = np.clip(linear, 0.0, None)
    # Auto-expose to the 99.5th percentile of luminance
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    p = np.percentile(lum, 99.5)
    if p > 1e-6:
        img = img / p
    img = np.clip(img, 0.0, 1.0) ** (1.0 / 2.2)
    return (img * 255.0).astype(np.uint8)


def pick_patches(linear_img: np.ndarray, window_title: str) -> np.ndarray:
    """Open an interactive window; return (24, 3) median patch values in linear space."""
    h, w = linear_img.shape[:2]

    # Scale to fit on screen while keeping enough detail
    screen_target = 1400
    scale = min(1.0, screen_target / max(h, w))
    disp_w, disp_h = int(round(w * scale)), int(round(h * scale))

    disp_base = _tonemap_for_display(linear_img)
    disp_base = cv2.resize(disp_base, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    disp_base = cv2.cvtColor(disp_base, cv2.COLOR_RGB2BGR)

    # Sample box in full-resolution pixels
    box = max(3, int(round(min(h, w) * SAMPLE_FRAC)))
    if box % 2 == 0:
        box += 1
    half = box // 2

    points = []  # list of (x_full, y_full)
    state = {"confirmed": False, "aborted": False}

    def redraw():
        canvas = disp_base.copy()
        for i, (x, y) in enumerate(points):
            dx, dy = int(round(x * scale)), int(round(y * scale))
            color = (0, 255, 0)
            cv2.drawMarker(canvas, (dx, dy), color, cv2.MARKER_CROSS, 14, 2)
            cv2.putText(canvas, str(i + 1), (dx + 6, dy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            # Draw sample box at display scale
            box_disp = max(2, int(round(box * scale)))
            cv2.rectangle(canvas,
                          (dx - box_disp // 2, dy - box_disp // 2),
                          (dx + box_disp // 2, dy + box_disp // 2),
                          color, 1)
        status = f"[{len(points)}/{PATCH_COUNT}]  row-major  L->R top->bottom   u:undo  r:reset  Enter:done  q:abort"
        cv2.rectangle(canvas, (0, 0), (disp_w, 22), (0, 0, 0), -1)
        cv2.putText(canvas, status, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window_title, canvas)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < PATCH_COUNT:
            fx, fy = x / scale, y / scale
            points.append((fx, fy))
            redraw()

    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_title, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('u') and points:
            points.pop()
            redraw()
        elif key == ord('r'):
            points.clear()
            redraw()
        elif key in (13, 10, 32):  # Enter / Space
            if len(points) == PATCH_COUNT:
                state["confirmed"] = True
                break
            print(f"  need {PATCH_COUNT} points, have {len(points)}")
        elif key in (ord('q'), 27):
            state["aborted"] = True
            break

    cv2.destroyWindow(window_title)

    if state["aborted"] or not state["confirmed"]:
        raise SystemExit("Aborted.")

    samples = np.zeros((PATCH_COUNT, 3), dtype=np.float64)
    for i, (x, y) in enumerate(points):
        ix, iy = int(round(x)), int(round(y))
        x0 = max(0, ix - half); x1 = min(w, ix + half + 1)
        y0 = max(0, iy - half); y1 = min(h, iy + half + 1)
        patch = linear_img[y0:y1, x0:x1].reshape(-1, 3)
        samples[i] = np.median(patch, axis=0)
    return samples


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_ccm(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares solve M (3x3) such that src @ M.T ~= dst.

    src, dst: (N, 3) arrays. Returns (3, 3).
    """
    # src @ M.T = dst   <=>   M.T = lstsq(src, dst)
    mt, *_ = np.linalg.lstsq(src, dst, rcond=None)
    return mt.T


def report_fit(M: np.ndarray, src: np.ndarray, dst: np.ndarray) -> None:
    pred = src @ M.T
    err = pred - dst
    per_patch = np.linalg.norm(err, axis=1)
    # Relative error vs. target magnitude, for intuition
    mag = np.linalg.norm(dst, axis=1) + 1e-9
    rel = per_patch / mag
    print("\nCCM (row-major, applied as: rgb @ M.T):")
    for row in M:
        print("  [{: .6f}, {: .6f}, {: .6f}],".format(*row))
    print(f"\nMean per-patch error: {per_patch.mean():.4f}")
    print(f"Max  per-patch error: {per_patch.max():.4f}")
    print(f"Mean relative error:  {rel.mean()*100:.2f}%")
    print("\nRow-sums (white balance sanity, should be roughly equal if chart is neutral-lit):")
    print(f"  {M.sum(axis=1)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("flashback_dng", help="Flashback ColorChecker DNG (reference)")
    ap.add_argument("other_dng", help="Other camera's DNG of the same chart")
    ap.add_argument("--out", default=None,
                    help="Path to save fitted CCM as .npy (default: <other_dng>.ccm.npy)")
    args = ap.parse_args()

    if not os.path.exists(args.flashback_dng):
        raise SystemExit(f"not found: {args.flashback_dng}")
    if not os.path.exists(args.other_dng):
        raise SystemExit(f"not found: {args.other_dng}")

    print(f"Developing Flashback reference: {args.flashback_dng}")
    fb_linear_srgb = develop_flashback(args.flashback_dng)
    print(f"  -> {fb_linear_srgb.shape}, range [{fb_linear_srgb.min():.4f}, {fb_linear_srgb.max():.4f}]")

    print(f"Developing other camera raw:    {args.other_dng}")
    other_raw = develop_other_raw(args.other_dng)
    print(f"  -> {other_raw.shape}, range [{other_raw.min():.4f}, {other_raw.max():.4f}]")

    print("\nClick the 24 patch centers in row-major order (L->R, top->bottom).")
    print("  Start with the Flashback reference image...")
    fb_patches = pick_patches(fb_linear_srgb, "Flashback reference - click 24 patches")

    print("  Now the other camera image (same semantic order)...")
    other_patches = pick_patches(other_raw, "Other camera - click 24 patches")

    # Sanity: both sets should have no zeros/negatives after raw development
    if (other_patches <= 0).any() or (fb_patches <= 0).any():
        print("  note: some patches contain non-positive values; fit will still proceed.")

    M = fit_ccm(other_patches, fb_patches)
    report_fit(M, other_patches, fb_patches)

    out_path = args.out or (os.path.splitext(args.other_dng)[0] + ".ccm.npy")
    np.save(out_path, M)
    print(f"\nSaved CCM to: {out_path}")

    # Also print a python-literal form suitable for pasting into core/config.py
    print("\nPython literal:")
    print("OTHER_CCM = np.array([")
    for row in M:
        print("    [{: .8f}, {: .8f}, {: .8f}],".format(*row))
    print("])")


if __name__ == "__main__":
    main()
