# Changelog

## Unreleased (targets 1.5.0-beta3)

### Performance: GPU-resident render pipeline

The render now keeps the image on the GPU for the whole pass. The entire
effect chain — vignette, bloom, chroma noise reduction, ACEScct encode, LUT,
chromatic aberration, edge softness, softness, grain, sharpen — runs as a
single GPU chain with one upload and one readback, instead of moving the
image between CPU and GPU for every effect. On the slower Windows / RTX 3090
reference machine a full-quality render dropped from ~2.3 s to ~0.3 s; on
Apple Silicon from ~0.48 s to ~0.11 s. The migration itself is output-neutral
(verified bit-exact against the previous CPU path); the CPU path is retained
as the oracle and as an automatic fallback when no usable GPU is present. The
look changes below are separate and intentional.

A follow-up pass cut the steady-state cost further: GPU textures and uniform
buffers are now drawn from a per-render arena instead of being reallocated
every frame (the allocation churn, not GPU compute, was the main remaining
interactive cost), and image loading is faster — load-time halation reuses the
same arena and blurs its glow at half resolution (imperceptible for a soft
glow), roughly a third quicker.

### Chromatic aberration is now spectral

CA was rebuilt as a spectral model: instead of three discrete red/green/blue
copies it integrates a continuous spectrum, giving a smooth purple→green
fringe like real glass.

- The fringe direction was corrected — a light→dark edge (going outward from
  the centre) fringes blue and a dark→light edge red, matching the scanned
  film reference and earlier betas.
- CA strength (and bloom) are now orientation-invariant: rotating a photo 90°
  no longer changes the look (both normalise to the long edge). Existing
  `ca_pixels` values are unchanged for landscape framing.
- The legacy CA sub-parameters (steps, blue blur, zoom blur) are no longer
  used by the spectral model.

### New: edge (corner) softness

An optional effect that softens the image toward the corners (lens
field-curvature look) while keeping the centre sharp. Off by default; tune
strength / blur / start-radius in the Advanced panel.

### Bloom is now applied after vignette

Bloom is generated from the vignetted image, as a real lens does: dimmed
perimeter highlights emit less glow, so bloom concentrates where the image is
actually bright instead of washing the darker edges.

### Stronger chroma noise reduction

The CNR range tolerance now scales with the amount, so high settings remove
substantially more chroma noise (the previous fixed value plateaued early)
while low settings stay edge-preserving. The strength slider range was
rescaled to match.

### New: Flashback ONE35 V1 negative support

The editor now opens **V1 "negatives"** — the headerless 8-bit raw the
first-generation ONE35 exports (it can't write DNGs). Each frame is a raw
Bayer dump plus a JSON sidecar, bundled per roll in a zip. V1 frames are
developed to the same ACEScg intermediate as V2 DNGs, so every slider,
halation, LUT, grain pass and every vibe behaves identically — a V1 roll
looks like a V2 roll.

- Import a roll by dropping or opening its `.zip`. The negatives are
  extracted into dated `Flashback_Output/<date>/_v1_imports/<roll>/`
  folders, with the date taken from the frame timestamps inside the zip,
  so an old roll exported later still lands on its shoot date.
- An already-extracted roll **folder** can be dragged straight in, and
  re-importing a roll (zip or folder) never overwrites existing files —
  matching the V2 camera import's skip-already-imported behaviour.
- The **disposable** vibe uses a V1-tuned LUT (`disposable_V1`) for V1
  negatives, so the disposable look holds up on V1's flatter capture; the
  other vibes share their V2 LUT. DNG export stays V2-only — a V1 negative
  has no raw DNG to round-trip.

### Cleaner, stable export filenames

Exports are renamed to short, sortable names derived only from the source
file, so the same image always produces the same name and the "already
processed" check keeps working:

- V2: `FBV2_<frame>` — the camera serial prefix is dropped, e.g.
  `FBV2_00042_disp.jpg`.
- V1: `FBV1_<roll>_<frame>` — a short tag for the roll plus the frame's
  sequence number, e.g. `FBV1_3f9c_00007_disp.jpg`.

Frame numbers are zero-padded to five digits. The per-vibe suffix
(`_disp`, `_ps`, `_rf`, `_mono`, `_v1`, or `_clean.dng`) is unchanged.

### Fixes

- Vignette no longer blackens the extreme corner pixels on some GPUs (a NaN
  from a corner-case power calculation).

## 1.5.0-beta2 — 2026-06-01

Changes since 1.5.0-beta. The full 1.5 changes are documented in the
1.5.0-beta entry below; this entry only covers the delta. Testers
upgrading from beta1 should read the "Notes for beta1 testers" at the
bottom — beta2 triggers a settings migration on first launch.

### Schema migration is now actually implemented

The 1.5.0-beta notes already described the vibe-state migration as a
breaking change, but beta1 didn't actually run it: the old
`vibe_state.json` was being silently read via tolerant deserialization,
so any field names that happened to overlap with the new schema leaked
through — often at the wrong scale. beta2 ships the real migration
work:

- Settings live in a new versioned file: `vibe_state_1_5_0.json` with
  a `schema_version: 2` envelope. Pre-1.5 `vibe_state.json` is left
  in place unmodified so a downgrade still finds the original file.
- One-shot migrator on first launch translates legacy values into the
  new schema. Bucket A (verbatim) covers the effect toggles and the
  pixel-unit fields. Bucket B (linear rescale) covers strength /
  color sliders that map onto the new percent units. Bucket C (reset
  to factory) covers thresholds, CNR, vignette feather, and CA
  strength — fields whose underlying coordinates moved with the
  pipeline rewrite.
- Custom user LUTs are reset to the vibe's factory LUT because a
  pre-1.5 `.cube` was authored against the old colour pipeline and
  would look ~2 stops over and colour-shifted under ACEScg. The
  original path is preserved in a new `legacy_user_lut` field on the
  vibe so you can re-import once you've regenerated the file.
- A non-blocking post-migration summary dialog explains what was
  rescaled, what was reset, and which custom LUTs were dropped.
  Dismissal persists across launches via the new envelope.

### Effect values now use user-facing units throughout

The advanced-settings panel is the most visible change. Each control
now has an explicit unit suffix and a range you can reason about:

- Strength controls (halation, sharpen, grain, vignette, bloom, CNR)
  are in percent.
- CA strength is in pixels (edge offset at the long edge of the
  frame), independent of image resolution.
- Threshold controls (halation, bloom) are in percent of the
  ACEScct-encoded dynamic range.
- Vignette feather is replaced by **Vignette Curve**, a signed
  slider from -100 to +100 where 0 is neutral, positive is softer
  falloff, negative is harder edge.
- Blur radii (halation, sharpen, softness, CA blue blur) stay in
  pixels, now explicitly labelled.

`VIBE_PRESETS` and the on-disk schema use the same units, so the
panel value, the saved value, and the preset definition are all
directly comparable.

### Tagged LUT references

Vibes store `lut_ref` instead of a raw `lut_path`. Values are either
`factory:<id>` (looked up in a registry against the current build's
asset directory) or `user:<absolute path>`. This kills the original
Windows bug where an upgrade in place could load the previous
version's LUT from the old install directory — a factory reference
always resolves against the install you're currently running. A
`user:` LUT whose file has gone missing falls back to the vibe's
factory LUT.

### Per-camera baseline exposure boost for generic raws

Non-Flashback raws are now boosted by a per-make EV value at develop
time so that the same exposure settings on different cameras land at a
roughly similar mid-grey in the developed output. Goal: within ~0.5 EV
of "matches ACR defaults", with our Fuji pipeline as the rough anchor.

Values are community ballpark (RawDigger Real ISO measurements,
DPReview studio comparisons, RawTherapee / darktable forum
consensus), not calibrated against an in-house reference — they will
be refined empirically. Notable entries: Fujifilm +1.0 EV, Apple
iPhone DNG +1.5 EV, Google Pixel DNG +2.0 EV, Pentax / Ricoh +0.5 EV,
Sigma +0.7 EV; Sony / Canon are the rough zero point. Fuji RAFs (a
proprietary container exifread can't parse) fall through to an
extension-based table.

### Sony ARW files no longer render black

`_wb_shift_to_kelvin` (introduced in the v2-pipeline-prep commit that
re-anchored generic raws at `BASE_KELVIN` before the WB slider) was
overwriting Sony's `daylight_whitebalance[3] = 0.0` sentinel with an
explicit G2 multiplier. libraw treats any non-zero G2 as an
independent multiplier rather than "G2 tracks G1", and the resulting
WB collapsed Sony output to near-black. The sentinel is now preserved.
The same path also guarded `raw.raw_pattern` against `None`, which
some compressed/lossless ARWs report.

### CA zoom-blur is now wired up

The `ca_zoom_blur` field existed in 1.5.0-beta but was silently
dropped at the call site. It now feeds the CA pass as expected,
which is what makes the disposable and flashback_classic_v1 presets
match their intended look.

### Parallel installs on every platform

Each release now installs into its own directory and registers as a
distinct entry in the OS, so two Flashback versions can coexist on
the same machine. Installing a newer release does NOT replace the
older one — if you want a clean upgrade, uninstall the old version
first.

- **Windows**: `C:\Program Files\Flashback One35 v2 <version>\`,
  per-version `AppId`, per-version Start-Menu group and uninstaller.
- **macOS**: `.app` bundle is `Flashback One35 <version>.app` with a
  per-version `CFBundleIdentifier`. Menu-bar display name carries
  the version too.
- Release artifact filenames are now versioned across all three
  platforms: `Flashback-macOS-<version>.dmg`,
  `Flashback-Windows-Setup-<version>.exe`,
  `Flashback-Linux-<version>.AppImage`.

### Windows uninstaller can remove this version's settings

The uninstaller asks once whether to also delete saved settings.
Defaults to "No" so a misclick can't lose data. If confirmed, only
the schema file the uninstalled version uses
(`vibe_state_1_5_0.json` for any 1.5.x release) is removed —
pre-1.5 `vibe_state.json` and any settings files belonging to a
parallel Flashback install are left alone.

### Notes for beta1 testers

- **First beta2 launch will migrate your beta1 settings.** Anything
  you tuned in beta1 goes through the same bucket logic as a fresh
  pre-1.5 migration: strength / colour sliders are rescaled into the
  new percent units; thresholds, CNR, vignette feather, and CA
  strength are reset to factory because the underlying numerical
  ranges changed in this release. The summary dialog lists what was
  touched.
- **Custom LUTs you imported in beta1 are reset to factory.** Same
  reason as the pre-1.5 case: any `.cube` predating the v2 ACEScg
  pipeline mis-targets. The original path is shown in the summary
  dialog so you can re-import once you've regenerated the file.
- **beta1 is not removed by beta2's installer.** The two installs
  are independent. If you want a clean state, uninstall beta1 first
  via Add/Remove Programs (Windows) or by dragging the old `.app`
  to Trash (macOS). Saved settings are shared between them only if
  you decline beta1's uninstaller "remove settings" prompt; if you
  uninstall beta1 and remove settings, beta2 will run as a fresh
  install with no migration.

## 1.5.0-beta — 2026-05-31

A large release. The headline is a full rewrite of the color pipeline
around the DNG dual-illuminant specification and ACEScg/AP1 as the
working space, plus highlight recovery. Generic raw support was
reworked, the editor gained project files and direct camera import,
the "Flashback Classic V1" vibe is back, and the internals were
restructured around a proper `VibeConfig` + `ImageAdjustments` split.
A test suite, CI, and GPL v3 license file landed along the way.

### Breaking changes

- **Vibe state format**: 1.5.0-beta read the pre-1.5 `vibe_state.json`
  with a tolerant deserializer that silently kept any field names that
  happened to overlap with the new schema — often at a wrong scale.
  This was a known issue and is replaced in 1.5.0-beta2 by a proper
  schema-versioned envelope, an explicit migrator, and a versioned
  on-disk filename (see the 1.5.0-beta2 entry for the full story).
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
