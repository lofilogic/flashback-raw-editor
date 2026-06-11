# Architecture

How LoFi Logic turns a RAW file into a film-looking image, and how the code is laid out.

- [Design philosophy](#design-philosophy)
- [The big picture](#the-big-picture)
- [Stage 1 — RAW to the ACEScg intermediate](#stage-1--raw-to-the-acescg-intermediate)
- [Stage 2 — the render pipeline](#stage-2--the-render-pipeline)
- [The exposure model](#the-exposure-model)
- [GPU-resident design](#gpu-resident-design)
- [Configuration & state](#configuration--state)
- [The LUT-authoring loop](#the-lut-authoring-loop)
- [Module map](#module-map)

---

## Design philosophy

- **Aesthetics over fidelity.** A pleasing, film-like look beats measurable accuracy.
- **Film-like low acuity is the point.** RAWs are developed at half-size (2×2 bin), this removes any demosaicing artifacts, reduces moiree and improves signal to noise ratio. The softness of the vibes is so soft, that more resolution barely affects the result.
- **Reward broad edits and batch processing**; don't encourage pixel-peeping.
- **Lean and pragmatic.** Prefer the simple thing that's actually worth it.
- **The look is LUT-driven.** LUTs are authored in DaVinci Resolve. A scene-referred ACEScct
  intermediate exists specifically so a 16-bit TIFF round-trip to Resolve survives without banding.

---

## The big picture

Every supported input converges on a single cached intermediate — a **linear ACEScg** image — and
everything downstream (sliders, effects, LUT, export) operates on that. Develop once, render many.

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

Loading a file produces the intermediate and an instant downscaled preview; a background worker
then renders the full-resolution frame. See [`core/processor.py`](../core/processor.py).

---

## Stage 1 — RAW to the ACEScg intermediate

There are three develop paths. They differ only in how they reach ACEScg; after that the pipeline
is shared. All matrices and the colour math live in [`core/processor.py`](../core/processor.py).

### Flashback One35 V2 (DNG) — the primary path

1. `rawpy.postprocess` with `user_wb=[1,1,1,1]`, `user_black=SENSOR_BLACK`, `half_size=True`,
   linear gamma, `output_color=raw`, 16-bit → normalised to `[0,1]`.
2. **Highlight recovery** (darktable-style "inpaint opposed"), in raw space, pre-white-balance:
   each clipped channel is reconstructed from the unclipped two so blown highlights keep plausible
   colour instead of skewing.
3. A single fused matmul takes white-balanced camera RGB → **ACEScg** (`FM1_WB_TO_ACESCG`,
   the calibrated ForwardMatrix folded with the `XYZ_D50 → ACEScg` transform).

The camera locks ISO and aperture, so its autoexposure is fully described by the EXIF
`ExposureTime` — this is read once and used by the [exposure model](#the-exposure-model).

### Flashback One35 V1 (negative) — see [`core/v1_negative.py`](../core/v1_negative.py)

The V1 can't write DNGs. It exports a *negative*: a headerless 8-bit RGGB Bayer dump plus a JSON
sidecar (geometry + metadata), usually delivered as a roll `.zip`.

```
read uint8 mosaic → black-subtract (+ decode dither) → demosaic RGGB →
downscale to the V2 pixel-scale → highlight recovery → exposure trim →
ForwardMatrix (raw → XYZ_D50) → XYZ_D50 → ACEScg → V2 white-balance match
```

It's downscaled to the **same 2072 px long edge** the V2 path produces, so every pixel-denominated
effect (grain tile, blur radii) transfers 1:1 and the two cameras share one look. The colour matrix
and white point are calibrated by [`tools/generate_matrices_v1.py`](../tools/generate_matrices_v1.py).

### Generic RAW (everything else) — best-effort

1. `rawpy.postprocess` with `output_color=sRGB`, the camera's `daylight_whitebalance` shifted to the
   Flashback reference Kelvin, `half_size` (full-size Markesteijn for X-Trans, then downscaled).
2. A small per-manufacturer EV boost nudges different bodies toward a similar mid-grey.
3. Linear sRGB → **ACEScg**.

This produces a good image, but it isn't a camera-specific calibration — different bodies won't land
in exactly the same place.

### Baking halation

For the Flashback paths, **halation** is applied to the intermediate at load time (see
`apply_halation` in [`core/effects.py`](../core/effects.py)) rather than per-frame, because it's a
low-frequency glow that doesn't need to recompute every slider move. It's the one effect baked into
the cached intermediate; everything else runs per-render.

---

## Stage 2 — the render pipeline

Each frame is rendered from the cached intermediate by `FlashbackProcessor._render`. Order matters —
this mirrors how a real camera/film system layers the same operations:

```
ACEScg intermediate
  → exposure · white balance · tint · push/pull        (linear ACEScg gain)
  → vignette → bloom → CNR                              (linear ACEScg, pre-LUT)
  → ACEScct encode → 3D LUT                             (the look)
  → chromatic aberration → edge softness → softness
       → grain → sharpen                                (display sRGB, post-LUT)
  → display sRGB
```

- **Pre-LUT** effects work in linear scene-referred light (vignette before bloom, so the glow is
  generated from the already-darkened perimeter, as real optics do). CNR (chroma noise reduction)
  runs in CIE Lab.
- **The LUT** is the look. Input is encoded to **ACEScct** (a log encoding) first; the LUT is applied
  via tetrahedral interpolation. With no LUT active, a tone-curve fallback path renders through
  ProPhoto instead.
- **Post-LUT** effects work on the display-referred (gamma-encoded) image, where lens artefacts like
  CA fringing and grain belong.

The CPU/oracle implementations of these effects live in [`core/effects.py`](../core/effects.py) and
[`core/kernels.py`](../core/kernels.py); their GPU twins live in [`core/gpu.py`](../core/gpu.py) and
[`core/shaders/`](../core/shaders).

---

## The exposure model

The intermediate is scene-referred, so exposure is just a linear gain — but the model distinguishes
two kinds of brightness change, which is the subtle part:

| Term | Source | Counteracted after the LUT? | Effect |
|------|--------|------------------------------|--------|
| User **Exposure** slider | per-image | no | changes output brightness |
| `base_exposure_offset_v2` | vibe | no | changes output brightness (LUT level match) |
| **Reverse-AE** × strength | vibe + EXIF | **yes** | shapes the film toe, brightness ~unchanged |
| Post-AE **boost** × strength | vibe | **yes** | shapes character, brightness ~unchanged |
| **Push / Pull** slider | per-image | **yes** | trades toe/highlight character, brightness ~unchanged |

The "counteracted" terms (`pre_lut_ev`) are applied *before* the LUT and then undone *after* it
(`post_gain = 2^(-pre_lut_ev)`), so they change **how the image travels through the LUT** — how much
sits in the toe vs the shoulder — without changing the final brightness. The non-counteracted terms
genuinely raise or lower the output.

Reverse-AE is a profiling tool; it and the LUT-profiling TIFF export are **not** user-facing.

---

## GPU-resident design

The pipeline keeps pixels on the GPU. See [`core/gpu.py`](../core/gpu.py) and
[`core/kernels.py`](../core/kernels.py).

- **`Frame`** is an image handle that lazily lives on the CPU (float32) or the GPU
  (`rgba32float` texture). It materialises the other side only at a real backend boundary, so two
  GPU stages in a row never round-trip through numpy.
- **`run_resident`** uploads once, runs a chain of `Frame → Frame` stages, and reads back once. With
  a LUT active, the *entire* render (pre-LUT effects, encode, LUT, post-LUT tail) runs as one resident
  chain — a single upload and a single readback per frame.
- **`_RenderArena`** is a thread-local bump allocator that reuses textures and uniform buffers across
  renders, killing per-frame allocation churn (the dominant interactive cost). It's thread-local
  because preview, thumbnail, and vibe-refresh workers render concurrently on the shared device.
- **Every GPU stage has a numpy/cv2 twin** that serves double duty: it's the *oracle* the parity
  tests check the shader against, and the *runtime fallback* when there's no usable GPU. Deleting the
  fallback would also delete the oracle — see [DEVELOPMENT.md → Testing](DEVELOPMENT.md#testing).
- **Graceful degradation.** Adapter selection flags software adapters (WARP, lavapipe, …) and
  init failures; the app surfaces a "GPU not detected" banner and falls back to the (slow) CPU path
  rather than crashing.

Why `f32` textures (not `f16`)? The pipeline is CPU-bound, so the half-float pack/unpack would cost
more CPU than the GPU bandwidth it saves, and `f32` keeps the resident path bit-matchable to the CPU
oracle. The full rationale is in the `_TEX_FORMAT` note in [`core/gpu.py`](../core/gpu.py).

---

## Configuration & state

Two dataclasses model the two layers of user-mutable state (see [`core/config.py`](../core/config.py)):

- **`VibeConfig`** — the "film stock" layer: every effect parameter that defines a look. One per vibe.
- **`ImageAdjustments`** — the per-image layer: exposure, WB, tint, push/pull, rotation.

Both are passed by reference; the UI mutates them and the next render picks up the change. The
processor never reads global state.

- **Vibes** are seeded from `VIBE_PRESETS` recipes. Five ship factory: Disposable, Point & Shoot,
  Rangefinder, Monochrome, Flashback Classic (V1).
- **LUTs** are referenced by a tagged string — `factory:<id>` or `user:<path>` — never a bare path,
  so a moved install can't load the wrong file. V1 negatives transparently swap in a V1-tuned variant
  of a factory look.
- **Persistence:** saved vibes live in a versioned JSON under the platform app-data dir
  ([`core/vibe_state.py`](../core/vibe_state.py)), with a one-time migration from the pre-1.5 schema.
  Projects (`.lofi`) store the image list + per-image settings with portable relative paths
  ([`core/project.py`](../core/project.py)).

---

## The LUT-authoring loop

The look is a 3D LUT, authored outside the app:

1. Export a frame as a **16-bit ACEScct TIFF** (advanced panel). ACEScct is scene-referred and log,
   so the round-trip survives without banding.
2. Grade it in **DaVinci Resolve** and export a `.cube`.
3. The tooling in [`tools/`](../tools) builds colour charts from film⇄digital comparison pairs to
   drive that grade.
4. Drop the `.cube` in as a `user:` LUT, or bundle it as a factory look.

---

## Module map

```
core/
  processor.py        develop paths + render pipeline (the spine)
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
  export_naming.py    deterministic export filenames
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

tools/                colour-chart building, matrix calibration, LUT/DNG utilities (dev-only)
tests/                parity tests (GPU vs numpy oracle) + shader-compile smoke test
```

See **[DEVELOPMENT.md](DEVELOPMENT.md)** to build, run, and test.
