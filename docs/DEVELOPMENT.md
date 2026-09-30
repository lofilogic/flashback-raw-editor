# Development

## Setup

Python 3.11 (what CI uses).

```bash
git clone https://github.com/lofilogic/flashback-raw-editor
cd flashback-raw-editor
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python main.py
```

On Linux you may need `libgl1 libglib2.0-0 libraw-dev`.

Without a working GPU the app falls back to a slow CPU path and shows a "GPU not detected" banner.

## Debugging

- `F12` opens the advanced panel: every vibe parameter, the LUT-profiling TIFF export, DNG profile
  name.
- `LOFILOGIC_DEBUG_TIMING=1 python main.py` prints per-stage render timings.
- `LOFILOGIC_FORCE_CPU=1` uses the CPU path even with a GPU.

## UI styling

Colours come from the tokens in [ui/theme.py](../ui/theme.py), so light and dark mode both work.
The F12 panel is the exception: it always uses a hard-coded dark style.

## Tests

```bash
pytest
```

Every GPU stage has a numpy/cv2 version. It's both the CPU fallback and the reference the shader is
tested against (`tests/parity_utils.py::assert_parity`, tolerance `1e-5`). A new GPU stage should
come with both.

CI has no GPU, so there the tests only cover the CPU path. A second CI job compiles all WGSL shaders
to SPIR-V on lavapipe, because Naga has crashed on Vulkan with shaders that were fine on Metal and
D3D12. Run `pytest` on a machine with a GPU before releasing shader changes.

## Building

```bash
pip install -r requirements-build.txt
LOFILOGIC_VERSION=v1.7.0 pyinstaller LoFiLogic.spec
```

CI ([build.yml](../.github/workflows/build.yml)) wraps the result in a `.dmg` (create-dmg), a Windows
installer (Inno Setup, [lofilogic.iss](../packaging/lofilogic.iss)) and an `.AppImage`.

Two Linux things:

- The AppImage drops the bundled `libstdc++`/`libgcc`. With them, Mesa's RADV driver fails to load
  and the app silently runs on the CPU (seen on Steam Deck).
- The GL backend is excluded because its EGL init aborts the process on some setups. See
  [core/gpu.py](../core/gpu.py).

## Releasing

Add a `## <version> — <date>` section to `CHANGELOG.md`, then:

```bash
git tag v1.7.0
git push origin v1.7.0
```

The tag builds all three platforms and publishes a GitHub Release with that changelog section.
Afterwards it points the README's download links at the new files and pushes that to `main`,
so pull after a release. Tags containing `-beta` or `-rc` become pre-releases and leave the
README alone.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pipeline works.
