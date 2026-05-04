#!/usr/bin/env python3
"""One-time FB calibration: sample the canonical Flashback HALD intermediate
TIFF and freeze the per-patch ACEScct values. Bundled with the GUI tool so
end users never need an FB DNG / their own calibration shoot.

Run this once after a verified FB HALD pass (matched natural_ratios + clean
intermediate export). Output: tools/fb_hald_samples.npz
"""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import core  # noqa: F401
sys.path.insert(0, str(HERE))
from build_lut_from_hald import load_tiff_float, sample_patches_fb, auto_sample_px

REPO = HERE.parent
SOURCE_TIFF = '/Users/julian/Pictures/Flashback_Output/colormatch/match-look-to-flashback/flashback_hald_intermediate.tif'
SOURCE_META = '/Users/julian/Pictures/Flashback_Output/colormatch/match-look-to-flashback/flashback_hald.hald_meta.json'
OUTPUT      = HERE / 'fb_hald_samples.npz'


def main():
    print(f'Loading FB intermediate TIFF: {SOURCE_TIFF}')
    img = load_tiff_float(SOURCE_TIFF)
    print(f'  shape: {img.shape[1]}×{img.shape[0]}')

    with open(SOURCE_META) as f:
        fb_meta = json.load(f)
    print(f'  HALD layout: n={fb_meta["n"]}, patch_px={fb_meta["patch_px"]}, '
          f'cols={fb_meta["cols"]}, rows={fb_meta["rows"]}')
    print(f'  FB canvas  : {fb_meta["img_w"]}×{fb_meta["img_h"]}')
    print(f'  FB natural_ratios used at inject: {fb_meta.get("natural_ratios")}')

    sp = auto_sample_px(fb_meta)
    print(f'  Sample window: {sp}px (auto-derived from patch_px={fb_meta["patch_px"]})')
    samples = sample_patches_fb(img, fb_meta, sample_px=sp, fb_meta=fb_meta)
    print(f'  Sampled {samples.shape[0]:,} patches')
    print(f'  ACEScct range  R:[{samples[:,0].min():.4f},{samples[:,0].max():.4f}]'
          f'  G:[{samples[:,1].min():.4f},{samples[:,1].max():.4f}]'
          f'  B:[{samples[:,2].min():.4f},{samples[:,2].max():.4f}]')

    np.savez(
        OUTPUT,
        samples=samples,
        n=fb_meta['n'],
        patch_px=fb_meta['patch_px'],
        cols=fb_meta['cols'],
        rows=fb_meta['rows'],
        natural_ratios_used=np.asarray(fb_meta.get('natural_ratios', [0.5156, 1.0, 0.6519]),
                                        dtype=np.float64),
    )
    print(f'\n✓ Wrote {OUTPUT.relative_to(REPO)}  ({samples.shape[0]} samples)')


if __name__ == '__main__':
    main()
