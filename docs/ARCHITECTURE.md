# Architecture

How a RAW file becomes a film-looking image, and where the code for each part lives.

## Goals

- A nice film look matters more than measured accuracy.
- RAWs are developed at half size (2×2 binning). That avoids demosaicing artifacts, reduces moiré
  and improves noise, and the effects are soft enough that more resolution wouldn't show.
- Made for applying a look to a whole roll, not for pixel-peeping.
- Keep it simple.
- The look comes from LUTs made in DaVinci Resolve. The ACEScct intermediate exists so a 16-bit
  TIFF can go to Resolve and back without banding.

## Overview

Every input is developed once into a cached **linear ACEScg** image. Sliders, effects, LUT and
export all work from that.

```
                     ┌─────────────────────────────────────────────┐
  Flashback V2 DNG ──┤                                             │
  Flashback V1 neg ──┤   develop  ──►  linear ACEScg intermediate  │  ◄── halation baked in here
  generic RAW      ──┤            (cached on load, per image)       │      (load time, once)
                     └──────────────────────┬──────────────────────┘
                                            │
                          per-frame render  │  (exposure, WB, tint, effects, LUT)
                                            ▼
                                     display sRGB / JPEG
```

Loading a file produces the intermediate and a quick downscaled preview; a background worker
then renders the full-resolution frame. See [`core/processor.py`](../core/processor.py).

## Developing

Three paths lead to ACEScg; after that everything is shared. The matrices are in
[`core/processor.py`](../core/processor.py).

### Flashback One35 V2 (DNG)

1. `rawpy.postprocess` with `user_wb=[1,1,1,1]`, `user_black=SENSOR_BLACK`, `half_size=True`,
   linear gamma, `output_color=raw`, 16-bit → normalised to `[0,1]`.
2. **Highlight recovery** (like darktable's "inpaint opposed") in raw space, before white balance:
   a clipped channel is rebuilt from the other two so blown highlights keep a plausible colour.
3. One matrix from white-balanced camera RGB to **ACEScg** (`FM1_WB_TO_ACESCG`: the calibrated
   ForwardMatrix combined with `XYZ_D50 → ACEScg`).

ISO and aperture are fixed on this camera, so EXIF `ExposureTime` fully describes its
autoexposure. It's used by the [exposure model](#exposure).

### Flashback One35 V1 (negative)

See [`core/v1_negative.py`](../core/v1_negative.py). The V1 can't write DNGs. Its "negative"
export is a headerless 8-bit RGGB mosaic plus a JSON sidecar, usually as a `.zip` per roll.

```
read uint8 mosaic → black-subtract (+ decode dither) → demosaic RGGB →
downscale to the V2 pixel-scale → highlight recovery → exposure trim →
ForwardMatrix (raw → XYZ_D50) → XYZ_D50 → ACEScg → V2 white-balance match
```

It's downscaled to the same 2072 px long edge as V2 files, so pixel-sized effects like grain and
blur look the same on both. The matrix and white point come from
[`tools/generate_matrices_v1.py`](../tools/generate_matrices_v1.py).

### Other RAW files

1. `rawpy.postprocess` with `output_color=sRGB`, the camera's `daylight_whitebalance` shifted to the
   Flashback reference Kelvin, `half_size` (full-size Markesteijn for X-Trans, then downscaled).
2. An exposure offset: the DNG's BaselineExposure if it has one, else a measured value per
   manufacturer, else 0.
3. Linear sRGB → **ACEScg**.

Not a per-camera calibration, so different cameras won't match exactly.

### Halation

**Halation** is applied once at load (`apply_halation` in [`core/effects.py`](../core/effects.py))
and stored in the intermediate. It's a slow, low-frequency effect that doesn't depend on the
sliders. Everything else runs on every render.

## Rendering

`FlashbackProcessor._render` renders from the intermediate in this order:

```
ACEScg intermediate
  → exposure · white balance · tint · push/pull        (linear ACEScg gain)
  → vignette → bloom → CNR                              (linear ACEScg, pre-LUT)
  → ACEScct encode → 3D LUT                             (the look)
  → chromatic aberration → edge softness → softness
       → grain → sharpen                                (display sRGB, post-LUT)
  → display sRGB
```

- **Before the LUT**, effects work in linear light. Vignette comes before bloom so the darkened
  edges glow less. Chroma noise reduction (CNR) works in Lab.
- **The LUT** is applied to ACEScct (a log encoding) with tetrahedral interpolation. Without a LUT,
  a tone curve in ProPhoto is used instead.
- **After the LUT**, effects work on the display image; CA and grain look right there.

The CPU versions are in [`core/effects.py`](../core/effects.py) and
[`core/kernels.py`](../core/kernels.py), the GPU versions in [`core/gpu.py`](../core/gpu.py) and
[`core/shaders/`](../core/shaders).

## Exposure

Exposure is a linear gain on the intermediate, but there are two kinds:

| Term | Source | Counteracted after the LUT? | Effect |
|------|--------|------------------------------|--------|
| User **Exposure** slider | per-image | no | changes output brightness |
| `base_exposure_offset_v2` | vibe | no | changes output brightness (LUT level match) |
| **Reverse-AE** × strength | vibe + EXIF | **yes** | shapes the film toe, brightness ~unchanged |
| Post-AE **boost** × strength | vibe | **yes** | shapes character, brightness ~unchanged |
| **Push / Pull** slider | per-image | **yes** | trades toe/highlight character, brightness ~unchanged |

The counteracted terms (`pre_lut_ev`) are applied before the LUT and undone after it
(`post_gain = 2^(-pre_lut_ev)`). They change where the image sits on the LUT's curve (toe vs
shoulder), not how bright the result is. The others change the output brightness.

Reverse-AE and the LUT-profiling TIFF export are for making LUTs and stay in the F12 panel.

## GPU

See [`core/gpu.py`](../core/gpu.py) and [`core/kernels.py`](../core/kernels.py).

- **`Frame`** holds an image as a float32 array, an `rgba32float` texture, or both, and converts
  only when needed. Consecutive GPU stages never go through numpy.
- **`run_resident`** uploads once, runs a list of `Frame → Frame` stages, and reads back once.
  With a LUT, the whole render is one such chain.
- **`_RenderArena`** reuses textures and uniform buffers between renders; allocating them every
  frame was the biggest interactive cost. It's per thread because the preview and thumbnail
  workers render at the same time.
- **Every GPU stage has a numpy/cv2 version**, used as the fallback without a GPU and as the
  reference in the tests.
- Without a usable GPU (none, or a software adapter like WARP or lavapipe) the app shows a
  "GPU not detected" banner and runs on the CPU.

Textures are `f32`, not `f16`: the pipeline is CPU-bound, so packing to half floats would cost more
than it saves. See `_TEX_FORMAT` in [`core/gpu.py`](../core/gpu.py).

## State

Two dataclasses in [`core/config.py`](../core/config.py):

- **`VibeConfig`**: all effect parameters of a vibe.
- **`ImageAdjustments`**: exposure, WB, tint, push/pull and rotation of one image.

The UI and the processor share the same instances; the next render picks up changes.

- **Vibes** start from `VIBE_PRESETS`: Disposable, Point & Shoot, Rangefinder, Monochrome and
  Flashback Classic (V1).
- **LUTs** are stored as `factory:<id>` or `user:<path>`, so a moved install still finds its own
  files. V1 negatives use a V1 variant of a factory LUT where there is one.
- Saved vibes are a versioned JSON file in the app data folder
  ([`core/vibe_state.py`](../core/vibe_state.py)), migrated once from the pre-1.5 format.
  Projects (`.lofi`) store the images and their settings with relative paths
  ([`core/project.py`](../core/project.py)).

## Making LUTs

1. Export a frame as a 16-bit ACEScct TIFF (F12 panel).
2. Grade it in DaVinci Resolve and export a `.cube`.
3. [`tools/`](../tools) builds colour charts from film/digital photo pairs to help with the grade.
4. Load the `.cube` as a user LUT, or ship it as a factory look.

## Module map

```
core/
  processor.py        develop and render
  config.py           constants, VibeConfig / ImageAdjustments, presets, unit conversions
  gpu.py              wgpu pipeline: Frame, resident stages, arena, adapter selection
  kernels.py          GPU-or-numpy kernels (LUT, blur, grain, ACEScct, colour transform)
  effects.py          effect functions / CPU oracles (halation, bloom, CA, vignette, CNR, …)
  shaders/            WGSL compute shaders, one per GPU stage
  v1_negative.py      V1 negative reader + develop
  dng_export.py       hand-rolled DNG writer (repackages the raw strip + Flashback metadata)
  camera_import.py    USB camera import into date-named folders
  project.py          .lofi save/load
  vibe_state.py       saved-vibe persistence + pre-1.5 migration
  export_naming.py    export filenames
  auto_exposure_reverse.py   reverse-AE gain from EXIF (profiling)

ui/
  editor.py           main window: loading, sliders, strip, export, shortcuts, drag & drop
  widgets.py          thumbnail strip/workers, zoomable view, vibe picker, render workers
  zen_overlay.py      full-screen gesture-driven Zen mode
  debug_panel.py      F12 advanced settings panel
  scrub_slider.py     the custom precision slider
  theme.py            design tokens + light/dark palettes
  native_chrome.py    macOS/Windows title-bar styling
  migration_notice.py post-migration summary dialog

tools/                colour charts, matrix calibration, benchmark (dev-only)
tests/                GPU vs numpy tests, shader compile test
```

See [DEVELOPMENT.md](DEVELOPMENT.md) to build, run and test.
