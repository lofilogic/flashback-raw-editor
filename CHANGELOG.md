# Changelog

## 1.5.0-beta — 2026-05-31

A large release. The headline is a full rewrite of the color pipeline
around the DNG dual-illuminant specification and ACEScg/AP1 as the
working space, plus highlight recovery. Generic raw support was
reworked, the editor gained project files and direct camera import,
the "Flashback Classic V1" vibe is back, and the internals were
restructured around a proper `VibeConfig` + `ImageAdjustments` split.
A test suite, CI, and GPL v3 license file landed along the way.

### Breaking changes

- **Vibe state format**: pre-1.5 `vibe_state.json` files are no longer
  loaded. The first launch will write a fresh dictionary of factory
  vibes.
- **TIFF import is no longer supported.** Open the file in 1.1.2 if
  you still need to round-trip TIFFs; an in-app dialog points there.
  TIFF was a v1-pipeline artifact and the v2 pipeline assumes raw
  input.
- **Pipeline removals** (visual: the same for typical edits, slightly
  different in extremes):
  - Lab highlight desaturation removed — unnecessary in the ACEScg
    working space.
  - Pre-LUT dither / anti-banding removed — the ACEScg → ACEScct
    encoding does not band.

### Color pipeline (v2)

- New working space: linear ACEScg (AP1, D60). Raw develops through
  forward matrices to XYZ_D50, then Bradford CAT to D60. Per-channel
  Planckian white balance replaces the v1 Rec.2020 chain.
- **Highlight recovery**: darktable-style "inpaint opposed", applied
  pre-WB. Clipped channels are reconstructed from neighbouring
  channels before the WB and color matrix run, so blown highlights
  retain hue and don't go magenta / cyan as they used to in extreme
  cases. Enabled by default; the fast (no-recovery) path remains
  available internally for the thumbnail worker.
- All v1 CCMs (`FLASHBACK_CCM`, `FLASHBACK_CCM2`, `IPHONE_CCM`),
  Rec.2020 conversion matrices, and related constants removed from
  `core/config.py`. v2 derives its matrices from the DNG spec.
- Bloom in the linear branch now uses ACEScg luminance weights
  (0.2722 / 0.6741 / 0.0537) instead of Rec.709 — green and blue
  were previously mis-weighted on AP1 primaries.
- CNR is now a Lab-space chroma denoiser with correct luma/chroma
  separation; the previous Rec.2020 version could fringe highlights.
- Generic raws now neutralise at `BASE_KELVIN` (5500 K / D55) before
  the WB slider, matching the Flashback path. Previously the slider
  behaved differently between the two raw types.

### Generic raw camera support (reworked, experimental)

Generic raw import (Fuji, Canon, Nikon, Sony, Olympus, Panasonic,
and non-Flashback DNGs) existed in 1.1.2; the path was rewritten on
top of the v2 pipeline for 1.5.0.

- `.raf`, `.cr2`, `.cr3`, `.nef`, `.arw`, `.orf`, `.rw2`, and
  non-Flashback `.dng` files develop via `rawpy` / libraw to linear
  sRGB → ACEScg, then feed the same downstream pipeline (LUT,
  sliders, grain, vignette, …) as Flashback files.
- DNG export is greyed out (with tooltip) for non-Flashback files;
  batch DNG export skips them and reports the count.
- **Still experimental**: the look is not at full parity with
  Flashback files. Generic raws use libraw's camera color matrices
  and the file's own daylight white balance, not the Flashback
  ColorChecker profile — colour response, especially in saturated
  areas, will differ.

### New vibe: Flashback Classic V1

The original Flashback look (`flashback_classic_v1`) is back as a
selectable vibe alongside `disposable`, `point_shoot`, `rangefinder`,
and `monochrome`. It uses the new V1 LUT and a chromatic-aberration
preset that leans on the new `ca_zoom_blur` control.

### Camera import & project files

- **Camera import**: connecting a Flashback camera offers an
  auto-export of the roll into dated subfolders of `Flashback_Output`
  (`YYYY-MM-DD/_RAW/<name>.dng`), skipping anything already imported.
  Detection is content-based (UNPROCESSED_JPG folder, `SN…_….dng`
  naming, or known volume label) and rejects volumes that hold more
  than the camera's 27-frame capacity, so a generic photo backup
  cannot trigger an import. Linux mount scan now descends both
  `/media/<user>/` and `/run/media`. When a roll is fully imported
  the dialog offers to open the on-disk copies instead.
- Each exported DNG is written with a freshly-generated embedded
  thumbnail, reusing the loaded image for the strip.
- **Project files (`.fbproj`)**: JSON format storing image paths,
  per-image settings, rotation, and active index. Paths are stored
  relative to the project file (POSIX-style) so projects survive
  being moved or shared; absolute paths from earlier saves still
  load. New menu items: Open Project, Save / Save As (Cmd+S / Cmd+
  Shift+S), and an Open Recent Project list backed by QSettings.
- **Curation**: Delete / Backspace removes the active image;
  thumbnails can also be dragged down past the bottom edge of the
  window. Removal works inside Zen Mode and auto-closes Zen when the
  last image is dropped.
- Image lists are now always sorted alphabetically, regardless of OS.
- **Camera import and export folders are now separate.** Camera
  import always archives rolls into the *camera import folder*;
  regular exports go to the *export folder*. Defaults for both are
  configurable in the advanced panel (F12 → Default Folders) and
  persist across sessions; mid-session changes to the export folder
  via the toolbar are intentionally one-off, so a one-time "export
  here" can't accidentally nest the next camera import inside a
  dated subfolder.

### Export

- New: **Export LUT Profile TIFF** in the advanced panel (F12).
- Exported JPGs now carry a short vibe suffix instead of
  `_processed`: `_disp`, `_ps`, `_rf`, `_mono`, or `_v1`, depending
  on which vibe was active when the file was rendered. The DNG
  export keeps `_clean.dng` (the raw is vibe-agnostic).
- DNG export: `CalibrationIlluminant1` now correctly tagged D50 (the
  Flashback ColorChecker matrix was generated under a 5000 K light).
  In-app rendering is unchanged; third-party DNG processors (ACR,
  Lightroom, libraw) interpolating between FM1/FM2 should see
  sub-ΔE 2 shifts on saturated colors.
- ONE35 V2 sensor geometry (4144×3088, 15 995 840-byte strip) lifted
  into named `core.config` constants.

### Effects & UI

- **Chromatic aberration**: new `ca_zoom_blur` multiplier (default
  1.0) dials the global zoom-blur pass independently of CA strength.
  Exposed in the debug panel and used by the new V1 vibe.
- CNR controls moved from the "Baked" group (which required an image
  reload) to the top of the "Real-time Effects" group, where they
  actually live. Halation remains the only truly baked effect.
- Zen-mode overlay extracted into `ui/zen_overlay.py`; `ui/editor.py`
  shrunk by ~170 lines.
- Thumbnail strip scrolling now uses pixel-accurate input on macOS
  touchpads (and hi-res mice elsewhere), so two-finger swipes scroll
  smoothly instead of jumping in 100 px steps. A classic notched
  mouse wheel still steps by one thumbnail per tick.

### Performance

- **Loading**, **navigation between images**, and **vibe switching**
  are all noticeably faster. Contributors include: color-space
  transforms collapsed into fused matrix chains (`raw → ACEScg`,
  `XYZ_D50 → ACEScg`, `wb-normalised camera RGB → ACEScg`), two
  per-frame passes removed (Lab highlight desaturation, pre-LUT
  dither — the ACEScg pipeline does not need them), and the
  background render now reads from per-processor `vibe` /
  `adjustments` references rather than a global, which removes a
  source of redundant re-renders when the UI was edited mid-flight.

### Architecture

- `DebugConfig` global replaced with two dataclasses:
  - **`VibeConfig`** — the "film stock" layer (halation, grain, LUT,
    …); persisted to `vibe_state.json`.
  - **`ImageAdjustments`** — per-image edits (exposure, WB, tint,
    push/pull, rotation, plus `active_vibe_id` for future faithful
    restoration).
  Each `FlashbackProcessor` owns references to both; background
  render threads read from `self.vibe` / `self.adjustments` rather
  than a global, eliminating a slider-edits-leak-into-in-flight-render
  race.
- `vibe_state.load_all()` now returns `dict[str, VibeConfig]`.

### Tools

- **`tools/curate_color_charts.py`** (new): interactive side-by-side
  viewer for one film/digital pair. Drag patches to reposition,
  right-click to delete, click empty space to add. Press `S` to write
  the chart TIFFs. Skips the CLI's hue and local-delta filters since
  curation replaces them by eye.
- **`tools/build_color_charts.py`**: outlier filter now uses K-NN
  local delta consensus (default k=15) in film-Lab space instead of
  a global median. The film→digital LUT is locally smooth but
  globally nonlinear, so a single median over-rejects saturated
  patches; local consensus respects the curvature. Default
  `--max-delta-mad` lowered from 3.0 to 2.5. Doc strings and labels
  corrected to ACEScct/AP1 (not Rec.2020).
- `tools/raw_whitebalance_analyzer` renamed to `.py` so editors and
  imports treat it as a Python file.

### Dev / infra

- **Test suite** (`pytest`, 37 tests) covering matrix consistency,
  WB direction, tone-curve endpoints, config dataclass roundtrips,
  preset completeness, and effect strength-zero contracts.
  `pip install -r requirements-dev.txt && pytest`.
- **GitHub Actions CI** runs `pytest` on every push and PR to main
  (Python 3.11, Ubuntu, headless PySide6).
- All ad-hoc `print()` calls (20+) across `core` and `ui` routed
  through stdlib `logging`. `main.py` installs a `basicConfig` so
  on-screen output looks unchanged but is now level-filterable and
  test-capturable.
- `load_image` failures now log a full traceback through `logging`
  (visible in bundled apps; previously they vanished silently).
- `DEBUG_TIMING` defaults off; developers opt in with
  `FLASHBACK_DEBUG_TIMING=1`. A `.vscode/launch.json` ships with
  both a normal and debug-timing launch config; `.gitignore`
  allowlists it while excluding personal `.vscode/settings.json`.
- **LICENSE**: canonical GPL v3 text now present in the repo
  (README already declared GPL v3).
- Several docstrings / timing labels corrected to describe the
  pipeline's actual color spaces (ACEScct as the log encoding of
  ACEScg, not Rec.2020; CA on display-sRGB, not on linear ACEScg).
