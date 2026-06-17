# Changelog

## 1.6.0 — 2026-06-17

Broader camera support, smarter exposure for generic raws, and two memory fixes —
including one that could drive the app to many gigabytes and out of memory.

### Raw format support
- **Greatly expanded the import whitelist** — most formats libraw can decode now
  open (Canon, Nikon, Sony, Fujifilm, Olympus, Panasonic, Leica, Pentax, Hasselblad,
  Phase One, Kodak, Sigma, GoPro, and more). The decode itself is the gate: an
  unsupported or corrupt file surfaces as a clean miss instead of being pre-rejected.

### Exposure
- **Tiered exposure for non-Flashback raws.** DNGs read their embedded
  `BaselineExposure` (Tier 1); a few proprietary formats use a measured per-make
  residual (Tier 2); unknown bodies now apply **no** per-camera lift (Tier 3),
  which stops unmeasured raws from reading hot. A constant anchor re-aligns the
  libraw linear develop to the mid-grey the pipeline expects.
- **Exposure slider range widened to ±3 stops** (was ±2). Existing projects load
  unchanged — saved values keep their exact exposure.

### Fixes
- **Fixed a GPU memory leak that could exhaust system RAM.** The per-render texture
  arena pooled resources by image resolution and never released them, so browsing
  raws of differing sizes accumulated GPU memory without bound — on unified-memory
  Macs that is system RAM. Stale pools are now freed on resolution change.
- **Removing an image no longer leaves its thumbnail behind.** An LRU-cache `pop`
  raised instead of returning its default, aborting removal partway (backspace and
  drag-out both affected).
- **Developing overlay dims the whole window again.** The fade animated a child
  widget's window opacity, which composites wrong on macOS — the darkening
  collapsed to just the spinner and text mid-load. It now fades via a graphics
  effect that covers the full window.

### Under the hood
- Image/preview/thumbnail caches now share one **RAM-relative memory budget**
  (a quarter of system RAM, backing off live as other apps need memory) with
  LRU eviction across caches, so resident cache memory stays bounded.

## 1.5.0 — 2026-06-11

The largest release so far, and the first under the new name. Highlights: a full
colour-pipeline rewrite around ACEScg and the DNG dual-illuminant spec, a
GPU-resident render that's several times faster, support for the original ONE35
(V1) negatives, a reworked film look, and project files + camera import.

> **Upgrading from 1.1.2:** saved settings migrate automatically on first launch.
> Custom `.cube` LUTs are reset to the vibe's factory LUT (a pre-1.5 LUT was built
> against the old pipeline and would look ~2 stops over) — the original path is
> kept so you can re-import once regenerated. TIFF *import* is no longer supported;
> open the original DNG instead.

### Renamed to LoFi Logic
- The app is now **LoFi Logic** — "Flashback" is the camera; this is an independent
  editor for it. Preferences and saved vibes carry over on first launch.
- Project files use the **`.lofi`** extension (older `.fbproj` still opens) and open
  by double-click once associated (the installers register the type).

### Colour pipeline (v2)
- New working space: **linear ACEScg (AP1)**. Raw develops through DNG forward
  matrices to XYZ_D50, Bradford-adapts to D60, with per-channel Planckian white
  balance — replacing the old Rec.2020 chain.
- **Highlight recovery** (darktable-style "inpaint opposed") reconstructs clipped
  channels pre-white-balance, so blown highlights keep their hue.
- Chroma noise reduction moved to **Lab space**; bloom uses ACEScg luma weights.
- **Generic raw** (Canon, Nikon, Sony, Fujifilm, Olympus, Panasonic, …) reworked
  onto the v2 pipeline with a per-make baseline EV nudge so different bodies land
  near the same mid-grey. Still a best-effort match, not a per-camera calibration.

### Performance
- **GPU-resident render:** the whole effect chain runs on the GPU with a single
  upload and readback instead of bouncing each effect through the CPU. Full-quality
  renders dropped from ~2.3 s to ~0.3 s (Windows / RTX 3090) and ~0.48 s to ~0.11 s
  (Apple Silicon). The CPU path is kept as the reference oracle and an automatic
  fallback when no GPU is available.
- A per-render resource arena removes per-frame allocation churn; load-time halation
  is ~⅓ faster. Returning to an already-viewed image is now instant (cached preview).

### The film look
- **Chromatic aberration is now spectral** — a continuous purple→green fringe like
  real glass, in the physically-correct direction and orientation-invariant.
- **Bloom is generated after vignette**, so dimmed edges glow less and bloom
  concentrates where the image is actually bright.
- New **edge (corner) softness** effect (lens field-curvature look; off by default).
- Stronger chroma noise reduction at high settings; calmer Zen-mode exposure drag.

### Flashback ONE35 V1 support
- Opens **V1 negatives** — the headerless 8-bit raw the first-gen ONE35 exports.
  Drop a roll's `.zip` (or an extracted folder); frames develop to the same ACEScg
  intermediate as V2, so every slider, vibe, and LUT behaves identically.
- The **disposable** vibe uses a V1-tuned LUT so the look holds up on V1's flatter
  capture. Re-importing a roll never overwrites existing files.

### Editor
- **Project files**, **Open Recent**, and **Save / Save As** (Cmd/Ctrl+S).
- **Camera import:** connect a ONE35 and the roll is archived into dated folders,
  skipping anything already imported; import and export folders are separate and
  configurable.
- **Curation:** Delete/Backspace or drag a thumbnail off the strip to remove a frame
  (works in Zen mode too). Image lists always sort alphabetically.

### Advanced panel & export
- Every effect control now uses **user-facing units** (percent, pixels, EV); vignette
  feather is replaced by a signed **Vignette Curve**.
- LUTs are referenced by a **tagged id** (`factory:` / `user:`) so an in-place upgrade
  can't load the wrong file.
- **Stable, aesthetic export names** derived from the source — `FBV2_<frame>` /
  `FBV1_<roll>_<frame>` plus a per-vibe suffix — so the "already processed" check works.
- **ACEScct TIFF export** (for building LUTs in DaVinci Resolve) now exports at the
  app's standard exposure by default; reverse-AE is an opt-in for film-stock profiling.
- **Exported DNGs keep their exposure** (`ExposureTime` written to the standard Exif
  location, read by Adobe / Camera Raw); `CalibrationIlluminant1` corrected to D50.

### Install & platform
- **Parallel installs:** each release installs side by side under its own version,
  with versioned artifacts (`LoFiLogic-macOS-…`, `-Windows-Setup-…`, `-Linux-…`).
- The Windows uninstaller can optionally remove just this version's settings.
- **GPU diagnostics:** the app reports how the GPU resolved and shows a banner if it
  falls back to software / CPU (usually a missing `pip install` when run from source).

### Fixes
- Fixed a **Vulkan startup crash** ("Expression is not cached!") from the CNR shader;
  every shader is now compile-checked in CI against software Vulkan.
- Vignette no longer blackens extreme corners on some GPUs (corner-case NaN).
- **Sony ARW** files no longer render near-black (G2 white-balance sentinel preserved).
- Linux AppImage launcher corrected to run from its own bundled contents.

### Under the hood
- Internals split into **`VibeConfig`** (the film-stock layer) and **`ImageAdjustments`**
  (per-image edits), removing a class of mid-render slider races.
- **Test suite + CI** (pytest, plus a shader-compile job); GPL-3.0 license; new
  developer docs (architecture + build/test guide).

### Notes
- **Linux is best-effort** — the AppImage is built in CI but not yet hardware-tested;
  on Steam Deck / immutable systems use `--appimage-extract-and-run` if FUSE is missing.
