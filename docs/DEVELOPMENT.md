# Development

Build, run, test, and package LoFi Logic from source.

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running from source](#running-from-source)
- [Developer affordances](#developer-affordances)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Building & packaging](#building--packaging)
- [Releasing](#releasing)
- [Conventions](#conventions)

---

## Prerequisites

- **Python 3.11** (what CI builds and tests against).
- A C/Qt-capable platform: macOS, Windows, or Linux.
- On **Linux**, you'll likely need a few system libs for Qt/OpenGL and RAW decoding:
  ```bash
  sudo apt-get install -y libgl1 libglib2.0-0 libraw-dev
  ```
- A working **GPU + current drivers** are recommended but not required — without one the app runs on
  a slow numpy/cv2 CPU fallback (and shows a "GPU not detected" banner).

---

## Setup

```bash
git clone https://github.com/lofilogic/flashback-raw-editor
cd flashback-raw-editor

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt        # to run the app
pip install -r requirements-dev.txt    # to run the tests (adds pytest)
```

Requirements are pinned. `requirements.txt` runs the app, `requirements-build.txt` adds PyInstaller
for packaging, `requirements-dev.txt` adds the test tooling.

---

## Running from source

```bash
python main.py
```

The first render lazily initialises the GPU device (no startup cost). Logs go to stdout with the
`[module]` prefixes and ✓ / ⚠ / ✗ glyphs you'll see referenced throughout the code.

---

## Developer affordances

- **`F12`** — toggle the Advanced Settings panel: live controls for every vibe effect parameter,
  plus the LUT-profiling TIFF export and DNG profile name. This is the tuning surface; the main
  window stays deliberately minimal.
- **`LOFILOGIC_DEBUG_TIMING=1`** — print per-stage render timings to stdout. Off by default so user
  installs stay quiet. Example:
  ```bash
  LOFILOGIC_DEBUG_TIMING=1 python main.py
  ```

---

## Project layout

See the **[module map in ARCHITECTURE.md](ARCHITECTURE.md#module-map)** for a per-file breakdown of
`core/` and `ui/`. In short: `core/` is the headless image pipeline (develop + render + I/O), `ui/` is
the PySide6 app, `tools/` is dev-only colour calibration, and `tests/` is the parity suite.

---

## Testing

```bash
pytest                       # full suite
pytest tests/test_effects.py # one module
pytest -v                    # verbose (what CI runs)
```

The suite is built around **parity**: every stage that moved from CPU to GPU keeps its numpy/cv2
implementation as the reference *oracle*, and the GPU/WGSL implementation must reproduce it within a
tight tolerance (`1e-5`). `tests/parity_utils.py::assert_parity` is the single gate every stage test
uses, so the GPU path can't silently drift from the maths the CPU path defines.

This is why the CPU fallbacks aren't "overkill": **the fallback and the test oracle are the same
function.** Removing it would remove the only independent reference the shaders are checked against.

### What CI runs ([`.github/workflows/tests.yml`](../.github/workflows/tests.yml))

| Job | Environment | What it checks |
|-----|-------------|----------------|
| `pytest` | ubuntu-22.04, no Vulkan driver | the full suite on the **CPU path** (no GPU present) |
| `shader-compile` | ubuntu-22.04 + Mesa **lavapipe** | every WGSL shader compiles to SPIR-V via Naga |

The `shader-compile` job exists because Naga's SPIR-V codegen has caught crashes that only surfaced
on Vulkan (e.g. an "Expression is not cached!" panic) while D3D12/Metal stayed fine. Note that the
**GPU-vs-oracle parity comparison runs locally on a real GPU** (Metal/D3D12/Vulkan) — CI's `pytest`
job has no GPU, so it exercises the CPU oracle, not the shaders. Run `pytest` on a GPU machine before
shipping GPU changes.

---

## Building & packaging

Packaging is driven by **[`LoFiLogic.spec`](../LoFiLogic.spec)** (PyInstaller). The spec injects the
version from the `LOFILOGIC_VERSION` environment variable (set by CI from the git tag) into
`_version.py` at build time.

```bash
pip install -r requirements-build.txt
LOFILOGIC_VERSION=v1.5.0 pyinstaller LoFiLogic.spec
```

Per-platform wrappers, all run by CI ([`.github/workflows/build.yml`](../.github/workflows/build.yml)):

| Platform | Wrapper | Output |
|----------|---------|--------|
| macOS (Apple Silicon) | `create-dmg` | `.dmg` |
| Windows | Inno Setup ([`packaging/lofilogic.iss`](../packaging/lofilogic.iss)) | `.exe` installer |
| Linux | `appimagetool` (AppDir assembled in the workflow) | `.AppImage` |

A couple of platform gotchas are encoded in the build (and worth knowing before you touch it):

- **Linux/AppImage** strips the bundled `libstdc++`/`libgcc` so the host's newer runtime drives the
  GPU — otherwise Mesa's Vulkan driver (RADV) fails to load against the older bundled copies and the
  app silently falls back to CPU (this is the Steam Deck story).
- The build excludes the GL/GLES backend on Linux because its EGL init aborts the process on some
  setups; see the backend-selection note in [`core/gpu.py`](../core/gpu.py).

---

## Releasing

Releases are tag-driven. Pushing a `v*` tag triggers the build workflow, which builds all three
platforms and publishes a GitHub Release:

```bash
# update CHANGELOG.md with a "## <version> — <date>" section first
git tag v1.5.0
git push origin v1.5.0
```

- Release notes are auto-assembled: a downloads table plus the matching `CHANGELOG.md` section
  (extracted by version heading).
- Tags containing `-beta` or `-rc` are automatically marked as **pre-releases**.
- `workflow_dispatch` lets you trigger a build manually from the Actions tab without tagging.

---

## Conventions

A few things reviewers will look for — they keep the codebase the way it is:

- **Every new GPU stage needs a numpy/cv2 twin and a parity test.** The twin is the oracle and the
  no-GPU fallback both; the parity test (`assert_parity`, tol `1e-5`) keeps them in lockstep.
- **Comments explain *why*, not *what*.** The non-obvious rationale (a platform quirk, a colour-science
  choice, a perf trade-off) is what earns a comment.
- **Fail loud in diagnostics, degrade gracefully at runtime.** GPU/driver problems fall back to CPU
  rather than crashing, but log the cause. Don't add bare `except:` or silent `except … : pass` —
  log at `debug` at minimum.
- **UI styling goes through the theme tokens** in [`ui/theme.py`](../ui/theme.py) so light/dark both
  work (the F12 advanced panel is the one deliberate exception).
- **Keep the user-facing surface minimal.** Profiling-only features (reverse-AE, LUT-profiling TIFF
  export) stay in the advanced panel; don't productise them.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for how the pipeline fits together.
