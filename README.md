# Flashback One35 v2 Editor

A dedicated RAW processor and editor built specifically for the [Flashback One35 v2](https://www.flashbackcamera.com) camera.

![App Screenshot](docs/UI.png)

The Flashback One35 v2 shoots 35mm-style DNG files with a unique sensor and lens character. This app processes those RAWs through a custom color pipeline tuned for the camera — bypassing generic RAW processors to deliver results that match the Flashback's analogue aesthetic.

![Before / After](docs/before-after.gif)

---

## Download

Get the latest release for your platform from the [Releases](../../releases/latest) page.

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon) | ✓ Tested |
| Windows | ✓ Tested |
| Linux x86_64 | ⚠ Untested — binary provided, feedback welcome |

### macOS
1. Download `Flashback-macOS.zip` and unzip it
2. Move `Flashback One35 v2.app` to your Applications folder
3. On first launch, right-click → Open (macOS Gatekeeper requires this for unsigned apps)

### Windows
1. Download `Flashback-Windows.zip` and unzip it
2. Run `Flashback One35.exe` from the extracted folder

### Linux
1. Download `Flashback-Linux.zip` and unzip it
2. Make the binary executable: `chmod +x "Flashback One35"`
3. Run it: `./"Flashback One35"`

---

## Features

- **Custom color pipeline** — sensor-specific CCM, white balance, and LUT tuned for the One35 v2
- **Film emulation** — halation, chromatic aberration, softness, and grain
- **Real-time preview** — fast preview using Numba-accelerated processing
- **Batch export** — queue and process multiple images to JPEG in one go
- **Adjustment copy/paste** — copy settings from one image and paste to a selection
- **Zen mode** — hide controls for a clean full-screen view of your image
- **Drag & drop** — drop DNG files or folders directly onto the app
- **Camera detection** — auto-detects the One35 v2 when connected via USB

---

## How to Use

![UI Walkthrough](docs/UI.gif)

### Basic workflow
1. Open a folder of DNGs via the folder icon, drag & drop, or connect your camera
2. Browse images in the thumbnail strip at the bottom
3. Adjust **Exposure**, **White Balance**, and **Tint** with the sliders
4. Set your export folder and click **Process** to export JPEGs

### Controls

**Preview**
- `Scroll` — zoom in/out
- `Left-click + drag` — pan (when zoomed in)
- `Double-click` — fit to screen

**Thumbnail strip**
- `← →` — navigate images
- `Right-click` — queue image for batch processing
- `Shift + click` — select a range
- `Cmd/Ctrl + click` — select individual images

**Adjustments**
- `Double-click` any slider — reset to default
- `Cmd/Ctrl + C` — copy settings from current image
- `Cmd/Ctrl + V` — paste settings to selected images

**Zen mode**
- Click the zen mode button to hide all controls and show only the image — useful for evaluating your shots without distraction. Click again to return to the editor.

---

## License

GPL v3 — see [LICENSE](LICENSE) for details.
