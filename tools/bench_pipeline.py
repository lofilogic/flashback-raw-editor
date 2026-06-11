"""Headless render-pipeline benchmark — no GUI, deterministic, cross-platform.

Loads one DNG through the real FlashbackProcessor and runs the full-quality
render several times, printing the built-in per-stage timing (the same
LOFILOGIC_DEBUG_TIMING output the app emits) plus a per-run wall time and a
median. Run the identical command on the M3 and the Windows/3090 box to get an
apples-to-apples, slider-free comparison of where the milliseconds go.

This is a measurement tool only — it never writes an image and never changes
pipeline output.

Usage:
    python tools/bench_pipeline.py path/to/file.dng
    python tools/bench_pipeline.py path/to/file.dng --vibe disposable --runs 7

Vibes: flashback_classic_v1 (default), disposable, point_shoot, rangefinder,
       monochrome — these drive which effects run (CA, grain, bloom, etc.).
"""
import os
# Must be set before any core.* import so core.config picks it up at module load.
os.environ.setdefault("LOFILOGIC_DEBUG_TIMING", "1")

import sys
import time
import argparse
import logging
import platform
import statistics

# Make the repo root importable when run as `python tools/bench_pipeline.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# INFO so the GPU adapter line ("✓ GPU pipeline ready: ...") is visible.
logging.basicConfig(level=logging.INFO, format="%(message)s")

from core.config import vibe_config_for, VIBE_PRESETS, ImageAdjustments  # noqa: E402
from core.processor import FlashbackProcessor                            # noqa: E402
from core.kernels import HAS_GPU                                         # noqa: E402
from core.gpu import gpu                                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Headless render-pipeline benchmark.")
    ap.add_argument("dng", help="Path to a DNG (or supported raw) to benchmark.")
    ap.add_argument("--vibe", default="flashback_classic_v1",
                    choices=sorted(VIBE_PRESETS.keys()),
                    help="Vibe preset to render with (default: flashback_classic_v1).")
    ap.add_argument("--runs", type=int, default=5,
                    help="Number of full-quality render passes to time (default: 5).")
    args = ap.parse_args()

    if not os.path.isfile(args.dng):
        print(f"✗ file not found: {args.dng}")
        return 1

    print("=" * 60)
    print(f"  platform : {platform.platform()}")
    print(f"  python   : {platform.python_version()}  ({platform.machine()})")
    print(f"  wgpu     : {'available' if HAS_GPU else 'NOT available (CPU fallbacks)'}")
    print(f"  vibe     : {args.vibe}")
    print(f"  image    : {os.path.basename(args.dng)}")
    print("=" * 60)

    vibe = vibe_config_for(args.vibe)
    processor = FlashbackProcessor(vibe=vibe, adjustments=ImageAdjustments())

    # The processor loads self.lut from the vibe's lut_ref, but uploading it to
    # the GPU buffer is normally the editor's job (ui/editor.py). Without this
    # the GPU LUT path returns None and falls back to slow CPU trilinear, so the
    # bench must mirror the editor to measure the real GPU LUT cost.
    if processor.lut is not None:
        gpu.upload_lut(processor.lut.table)
        print(f"  LUT uploaded to GPU buffer (size {processor.lut.table.shape})")
    else:
        print("  ⚠ no LUT resolved for this vibe — tone-curve path will run instead")

    # load_image emits its own load-stage timings (raw_develop, raw->ACEScg,
    # halation, ...) and triggers lazy GPU init (logs the adapter line).
    print("\n--- load (includes raw develop, highlight recovery, halation) ---")
    if processor.load_image(args.dng) is None:
        print("✗ load failed — see log above")
        return 1

    print(f"\n--- {args.runs} full-quality render passes ---")
    times_ms = []
    for i in range(args.runs):
        print(f"\n[run {i + 1}/{args.runs}]")
        t0 = time.time()
        processor.render_export()          # full render, downscale=False
        dt = (time.time() - t0) * 1000.0
        times_ms.append(dt)
        print(f"  >> full render wall: {dt:8.2f} ms")

    print("\n" + "=" * 60)
    print(f"  full render — median {statistics.median(times_ms):8.2f} ms"
          f"   min {min(times_ms):8.2f}   max {max(times_ms):8.2f}   (n={len(times_ms)})")
    print("=" * 60)
    print("\nFirst run is typically a cold outlier (shader/pipeline warm-up);"
          " compare the median across machines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
