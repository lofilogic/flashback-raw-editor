"""
RAW image processing engine.

FlashbackProcessor handles:
  1. RAW development → ACEScct intermediate (slow, once per image)
  2. Fast preview render: decode → WB/exposure → LUT → effects (~100-150ms)
  3. HQ preview: delegates to fast (placeholder for future expansion)
  4. Export render: full-quality with all effects at native resolution

export_image() handles JPEG and 16-bit TIFF output.
"""
import numpy as np
import rawpy
import colour
import cv2
import os
import time
import exifread
from pathlib import Path

from . import resource_path
from .config import (
    FLASHBACK_CCM, FLASHBACK_CCM2, SENSOR_BLACK, BASE_WB_SETTINGS, BASE_WB_SETTINGS2, BASE_EXPOSURE_OFFSET,
    DebugConfig, REC2020_FROM_SRGB,
    GRAIN_STRENGTH, SOFTNESS_SIGMA, SHARPEN_STRENGTH, SHARPEN_RADIUS,
    _timing_print,
)
from .kernels import (
    HAS_NUMBA,
    _numba_acescct_decode_core,
    _numba_acescct_encode_core,
    _rotate_90_clockwise_numba,
    _rotate_90_counterclockwise_numba,
    _apply_grain_numba,
)
from .effects import (
    apply_lut_fast,
    apply_chromatic_aberration,
    reduce_color_noise_chroma,
    apply_halation,
    apply_softness,
    apply_sharpen,
    add_blue_noise_dither,
)


def _is_flashback_dng(path: str) -> bool:
    """Return True if the file is a Flashback camera DNG (Make tag == 'Flashback')."""
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False, stop_tag='Image Make')
        make = str(tags.get('Image Make', '')).strip()
        return make.lower() == 'flashback'
    except Exception:
        return False


# =============================================================================
# IMAGE PROCESSOR CLASS
# =============================================================================

class FlashbackProcessor:
    """
    Handles image processing with fast preview and high-quality modes.

    Architecture:
    1. Load RAW → Preprocess to ACEScct intermediate (slow, once per image)
    2. Fast preview: Decode → WB/Exposure → Encode → Effects → LUT (fast)
    3. HQ preview: Fast preview + Grain + Sharpen (slower but complete)
    """

    def __init__(self, lut_path=None):
        """Initialize processor with LUT."""
        self.intermediate_acescct = None
        self.current_file = None
        self.preview_mode = "fast"  # "fast" or "hq"
        self.rotation = 0  # 0, 90, 180, 270 degrees
        self.is_full_res = False  # True when intermediate was built with half_size=False
        self.pixel_scale = 1.0    # Multiplier for pixel-sized effect params (2.0 when full-res)
        self.applied_rotation = 0  # Cumulative rotation baked into current intermediate
        self.grain_tiles = []  # Initialize grain tiles list

        # User-adjustable settings
        self.user_settings = {
            'exposure_ev': 0.0,   # -2 to +2 EV
            'wb_temp': 0,         # Temperature offset in Kelvin (-1000 to +1000)
            'tint': 0.0           # Tint offset (-10 to +10)
        }

        self._load_grain_tiles()

        # Load LUTs (preview for real-time, full for export)
        self.lut_preview = None
        self.lut_full = None

        if lut_path and os.path.exists(lut_path):
            try:
                self.lut_full = colour.read_LUT(lut_path)
                print(f"✓ Full LUT loaded: {self.lut_full.name} ({self.lut_full.table.shape})")

                self.lut_preview = self.lut_full
                print(f"✓ Using full LUT for preview ({self.lut_preview.table.shape})")

            except Exception as e:
                print(f"Warning: Could not load LUT: {e}")
        else:
            if lut_path:
                print(f"Warning: LUT not found at {lut_path}")
            else:
                print(f"Warning: No LUT path provided")

    def _load_grain_tiles(self):
        """Load pre-rendered grain tiles from assets/grain."""
        grain_dir = Path(resource_path("assets/grain"))

        if not grain_dir.exists():
            print(f"⚠ Grain directory not found: {grain_dir}")
            return

        tile_paths = sorted(grain_dir.glob("*.png")) + sorted(grain_dir.glob("*.jpg"))

        for path in tile_paths:
            try:
                tile = cv2.imread(str(path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
                tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                self.grain_tiles.append(tile)
                print(f"  ✓ Loaded grain tile: {path.name} ({tile.shape})")
            except Exception as e:
                print(f"  ⚠ Failed to load {path}: {e}")

        if self.grain_tiles:
            print(f"✓ Loaded {len(self.grain_tiles)} grain tiles")
        else:
            print(f"⚠ No grain tiles found in {grain_dir}, will generate procedurally")

    def generate_grain_layer(self, height, width, sigma=0.02):
        """Generate grain layer from pre-rendered tiles with random flipping for variation."""
        if not self.grain_tiles:
            grain = np.ones((height, width, 3), dtype=np.float32) * 0.5
            noise = np.random.normal(0, sigma, (height, width, 3)).astype(np.float32)
            return np.clip(grain + noise, 0, 1)

        grain = np.zeros((height, width, 3), dtype=np.float32)
        tile_h, tile_w = self.grain_tiles[0].shape[:2]

        for y in range(0, height, tile_h):
            for x in range(0, width, tile_w):
                tile = self.grain_tiles[np.random.randint(0, len(self.grain_tiles))].copy()

                if np.random.random() > 0.5:
                    tile = np.flip(tile, axis=1)
                if np.random.random() > 0.5:
                    tile = np.flip(tile, axis=0)

                h_end = min(y + tile_h, height)
                w_end = min(x + tile_w, width)
                tile_h_actual = h_end - y
                tile_w_actual = w_end - x

                grain[y:h_end, x:w_end] = tile[:tile_h_actual, :tile_w_actual]

        return grain

    def apply_grain_linear_light(self, image, strength=GRAIN_STRENGTH, scale=0.7):
        """Apply grain with pre-rendered tiles.

        scale > 1.0 generates the grain layer at reduced size then upscales,
        preserving apparent grain size across resolutions (at a small softness cost).
        """
        h, w = image.shape[:2]
        if scale > 1.0:
            gh = max(1, int(round(h / scale)))
            gw = max(1, int(round(w / scale)))
            grain = self.generate_grain_layer(gh, gw, sigma=strength)
            grain = cv2.resize(grain, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            grain = self.generate_grain_layer(h, w, sigma=strength)

        if HAS_NUMBA:
            return _apply_grain_numba(image, grain)
        else:
            return np.clip(image + (2.0 * grain) - 1.0, 0, 1)

    def rotate_clockwise(self):
        """Rotate image 90 degrees clockwise."""
        self.rotation = (self.rotation + 90) % 360
        return self._apply_rotation_and_render()

    def rotate_counterclockwise(self):
        """Rotate image 90 degrees counter-clockwise."""
        self.rotation = (self.rotation - 90) % 360
        return self._apply_rotation_and_render()

    def _apply_rotation_and_render(self):
        """Apply rotation to intermediate and re-render."""
        if self.intermediate_acescct is None:
            return None

        if self.rotation == 0:
            pass
        elif self.rotation == 90:
            self.intermediate_acescct = self._rotate_90(self.intermediate_acescct, clockwise=True)
        elif self.rotation == 180:
            self.intermediate_acescct = self._rotate_180(self.intermediate_acescct)
        elif self.rotation == 270:
            self.intermediate_acescct = self._rotate_90(self.intermediate_acescct, clockwise=False)

        self.applied_rotation = (self.applied_rotation + self.rotation) % 360
        self.rotation = 0
        return self.render_preview()

    def _rotate_90(self, img, clockwise=True):
        """Rotate 90 degrees using Numba or OpenCV."""
        if HAS_NUMBA:
            if clockwise:
                return _rotate_90_clockwise_numba(img)
            else:
                return _rotate_90_counterclockwise_numba(img)
        else:
            if clockwise:
                return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            else:
                return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _rotate_180(self, img):
        """Rotate 180 degrees using OpenCV."""
        return cv2.rotate(img, cv2.ROTATE_180)

    def get_rotation(self):
        """Get current rotation in degrees."""
        return self.rotation

    def _create_acescct_luts(self):
        """
        Precompute 1D lookup tables for ACEScct encode/decode.
        ACEScct is a per-channel operation, so 1D LUT works perfectly!
        This replaces colour-science's slow Python implementation.
        """
        lut_size = 65536

        acescct_vals = np.linspace(0, 1, lut_size, dtype=np.float32)

        linear_vals = np.where(
            acescct_vals < 0.155251141552511,
            (acescct_vals - 0.0729055341958355) / 10.5402377416545,
            np.power(2.0, acescct_vals * 17.52 - 9.72)
        ).astype(np.float32)

        self.acescct_decode_lut = linear_vals

        linear_input = np.linspace(0, 2.0, lut_size, dtype=np.float32)

        acescct_output = np.where(
            linear_input <= 0.0078125,
            10.5402377416545 * linear_input + 0.0729055341958355,
            (np.log2(np.maximum(linear_input, 1e-10)) + 9.72) / 17.52
        ).astype(np.float32)

        self.acescct_encode_lut = acescct_output
        self.acescct_encode_max = 2.0

        print(f"  ✓ ACEScct LUTs created (decode: {len(linear_vals)}, encode: {len(acescct_output)} entries)")

    def _fast_acescct_decode(self, acescct_img):
        """Wrapper to safely pass data to Numba core."""
        orig_shape = acescct_img.shape
        flat_input = acescct_img.ravel().astype(np.float32)
        out_buffer = np.empty_like(flat_input)

        _numba_acescct_decode_core(flat_input, out_buffer)

        return out_buffer.reshape(orig_shape)

    def _fast_acescct_encode(self, linear_img):
        """Wrapper to safely pass data to Numba core."""
        orig_shape = linear_img.shape
        flat_input = linear_img.ravel().astype(np.float32)
        out_buffer = np.empty_like(flat_input)

        _numba_acescct_encode_core(flat_input, out_buffer)

        return out_buffer.reshape(orig_shape)

    def load_image(self, dng_path, for_export=False, fast_mode=False, full_res=False):
        """
        Load and preprocess RAW image to ACEScct intermediate.
        This is the slow step, done once per image.

        Args:
            dng_path: Path to DNG file
            for_export: If True, apply halation (slower)
            fast_mode: If True, use LINEAR demosaic + clip highlights (10-20x faster, for thumbnails)
            full_res: If True, develop RAW at full sensor resolution (half_size=False).
                      Pixel-sized effect params are scaled by 2x to match preview look.
        Returns:
            preview image (numpy array)
        """
        total_start = time.time()

        self.is_full_res = full_res
        self.pixel_scale = 2.0 if full_res else 1.0
        self.applied_rotation = 0
        _half_size = not full_res

        _timing_print(f"\n{'='*60}")
        _timing_print(f"Loading: {os.path.basename(dng_path)} (fast={fast_mode}, full_res={full_res})")
        _timing_print(f"{'='*60}")

        self.current_file = dng_path
        profile = {}

        # ==========================================================
        # TIFF BYPASS
        # ==========================================================
        file_ext = str(self.current_file).lower().split('.')[-1]
        if file_ext in ['tif', 'tiff']:
            return self.load_intermediate_tiff(dng_path)

        # ==========================================================
        # STANDARD RAW PIPELINE
        # ==========================================================

        try:
            # Step 1: RAW development
            start = time.time()
            print("Developing RAW...")
            with rawpy.imread(dng_path) as raw:
                is_not_flashback = not _is_flashback_dng(dng_path)
                if fast_mode:
                    demosaic_fb = rawpy.DemosaicAlgorithm.LINEAR
                else:
                    demosaic_fb = rawpy.DemosaicAlgorithm.AHD
                highlight_mode = 1  # Clip: natural luma rolloff; chroma fixed pre-CCM below

                if is_not_flashback:
                    # Fuji RAW Development (Standard D65)
                    rgb_linear = raw.postprocess(
                        demosaic_algorithm=demosaic_fb,
                        use_camera_wb=False,
                        use_auto_wb=False,
                        user_wb=raw.daylight_whitebalance,
                        half_size=_half_size,
                        no_auto_bright=True,
                        bright=1,
                        highlight_mode=highlight_mode,
                        gamma=(1, 1),
                        output_bps=16,
                        output_color=rawpy.ColorSpace.sRGB
                    ).astype(np.float32) / 65535.0

                    h, w = rgb_linear.shape[:2]
                    img_srgb_lin = cv2.resize(rgb_linear, (int(w * 0.7), int(h * 0.7)), interpolation=cv2.INTER_AREA)

                    # Perceptual White Balance Shift
                    perceptual_gains = np.array([0.75, 0.95, 1.35], dtype=np.float32)
                    img_srgb_lin = img_srgb_lin * perceptual_gains
                    img_srgb_lin = np.clip(img_srgb_lin, 0.0, 1.0)

                    profile['raw_develop'] = (time.time() - start) * 1000
                    profile['color_matrix'] = 0.0

                else:
                    # --- FLASHBACK ONE35 V2 PIPELINE ---
                    rgb_linear = raw.postprocess(
                        demosaic_algorithm=demosaic_fb,
                        user_wb=BASE_WB_SETTINGS,
                        user_black=SENSOR_BLACK,
                        half_size=_half_size,
                        no_auto_bright=True,
                        bright=0.5,
                        highlight_mode=highlight_mode,
                        gamma=(1, 1),
                        output_bps=16,
                        output_color=rawpy.ColorSpace.raw,
                    ).astype(np.float32) / 65535.0

                    profile['raw_develop'] = (time.time() - start) * 1000
                    _timing_print(f"    RAW shape: {rgb_linear.shape}")

                    # Step 2: Apply CCM
                    start = time.time()
                    print("Applying color matrix...")
                    img_srgb_lin = (rgb_linear.reshape(-1, 3) @ FLASHBACK_CCM.T).reshape(rgb_linear.shape)
                    profile['color_matrix'] = (time.time() - start) * 1000

                    # Step 2b: Soft Lab desaturation in highlights
                    if DebugConfig.enable_highlight_desat:
                        img_srgb_lin = self._desaturate_highlights_lab(
                            img_srgb_lin,
                            threshold_L=DebugConfig.highlight_desat_threshold_L,
                            rolloff_L=DebugConfig.highlight_desat_rolloff_L,
                            sigma=DebugConfig.highlight_desat_sigma,
                        )
                    _timing_print(f"    After CCM: [{img_srgb_lin.min():.4f}, {img_srgb_lin.max():.4f}]")

            # Step 3: Convert to Rec.2020
            start = time.time()
            print("Converting to Rec.2020...")
            img_rec2020_lin = (img_srgb_lin.reshape(-1, 3) @ REC2020_FROM_SRGB).reshape(img_srgb_lin.shape)
            profile['rec2020'] = (time.time() - start) * 1000
            _timing_print(f"  After Rec.2020: [{img_rec2020_lin.min():.4f}, {img_rec2020_lin.max():.4f}]")
            _timing_print(f"    -> {profile['rec2020']:6.2f} ms")

            # Step 4: Apply base exposure offset
            start = time.time()
            print("Applying base exposure...")
            img_rec2020_lin *= BASE_EXPOSURE_OFFSET
            profile['exposure'] = (time.time() - start) * 1000
            _timing_print(f"  After exposure: [{img_rec2020_lin.min():.4f}, {img_rec2020_lin.max():.4f}]")
            _timing_print(f"    -> {profile['exposure']:6.2f} ms")

            # HALATION: Only applied during export (expensive blur operation)
            if DebugConfig.enable_halation and DebugConfig.halation_strength > 0:
                start = time.time()
                print("Baking in halation (export only)...")
                threshold = DebugConfig.halation_threshold
                strength = DebugConfig.halation_strength
                if is_not_flashback:
                    threshold = 0.6
                    strength = strength / 1.5
                img_rec2020_lin = apply_halation(
                    img_rec2020_lin,
                    threshold,
                    DebugConfig.halation_blur_radius * self.pixel_scale,
                    strength
                )
                profile['halation'] = (time.time() - start) * 1000
                _timing_print(f"  After halation: [{img_rec2020_lin.min():.4f}, {img_rec2020_lin.max():.4f}]")
                _timing_print(f"    -> {profile['halation']:6.2f} ms")

            # Step 5: Encode to ACEScct
            start = time.time()
            print("Encoding to ACEScct...")
            self.intermediate_acescct = self._fast_acescct_encode(
                np.maximum(1e-10, img_rec2020_lin)
            )

            # Ensure contiguous memory layout for fast access
            start = time.time()
            self.intermediate_acescct = np.ascontiguousarray(self.intermediate_acescct)
            profile['contiguous'] = (time.time() - start) * 1000

            # Apply color noise reduction (bake it into intermediate)
            if DebugConfig.enable_cnr and DebugConfig.cnr_sigma > 0:
                start = time.time()
                print("  Applying color noise reduction...")
                self.intermediate_acescct = reduce_color_noise_chroma(
                    self.intermediate_acescct, sigma=DebugConfig.cnr_sigma * self.pixel_scale
                )
                profile['cnr'] = (time.time() - start) * 1000
                _timing_print(f"    -> {profile['cnr']:6.2f} ms")

            # Render preview
            start = time.time()
            self.preview_mode = 'hq'
            result = self._render_preview()

            profile['render_preview'] = (time.time() - start) * 1000
            profile['total'] = (time.time() - total_start) * 1000

            _timing_print("\n=== LOAD_IMAGE TIMING BREAKDOWN ===")
            for key, value in profile.items():
                if key != 'total':
                    _timing_print(f"  {key:20s}: {value:6.2f} ms ({value/profile['total']*100:5.1f}%)")
            _timing_print(f"  {'TOTAL':20s}: {profile['total']:6.2f} ms")
            _timing_print("===================================\n")

            return result

        except Exception as e:
            print(f"✗ Error loading image: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_intermediate_tiff(self, tiff_path):
        """
        Loads an intermediate TIFF directly into the pipeline, bypassing RAW development.
        Expects the TIFF to be in the ACEScct color space.
        """
        start = time.time()
        _timing_print(f"\n{'='*60}")
        _timing_print(f"Loading Intermediate TIFF: {os.path.basename(tiff_path)}")
        _timing_print(f"{'='*60}")

        self.current_file = tiff_path

        try:
            img = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"OpenCV failed to read the file: {tiff_path}")

            if len(img.shape) == 3 and img.shape[2] >= 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img.dtype == np.uint16:
                img_float = img.astype(np.float32) / 65535.0
            elif img.dtype == np.uint8:
                img_float = img.astype(np.float32) / 255.0
            elif img.dtype == np.float16:
                img_float = img.astype(np.float32)
            else:
                img_float = img.astype(np.float32)

            self.intermediate_acescct = img_float
            self.preview_mode = 'hq'

            _timing_print(f"✓ Loaded intermediate TIFF in {(time.time() - start)*1000:.2f} ms")

            return self.render_preview()

        except Exception as e:
            print(f"✗ Failed to load intermediate TIFF: {e}")
            return None

    def update_setting(self, param, value):
        """
        Update a user setting and re-render.
        Automatically switches to fast mode.
        """
        if param in self.user_settings:
            self.user_settings[param] = value
            self.preview_mode = "fast"
            return self.render_preview()
        return None

    def request_hq_preview(self):
        """Switch to high-quality preview mode."""
        self.preview_mode = "hq"
        return self.render_preview()

    def render_preview(self):
        """Main render function — delegates to fast or HQ based on mode."""
        if self.intermediate_acescct is None:
            return None

        if self.preview_mode == "hq":
            return self._render_hq()
        else:
            return self._render_fast()

    def _render_preview(self):
        """Internal alias used during load_image."""
        return self.render_preview()

    def _render_fast(self):
        """
        Fast preview render. Applies full effect chain except halation.
        Target: ~100-150ms for responsive editing.
        """
        profile = {}
        total_start = time.time()

        img = self.intermediate_acescct.copy()
        profile['copy'] = time.time() - total_start

        orig_h, orig_w = img.shape[:2]

        is_fast_mode = getattr(self, 'preview_mode', 'hq') == 'fast'

        if is_fast_mode:
            img = cv2.resize(img, (orig_w // 3, orig_h // 3), interpolation=cv2.INTER_LINEAR)

        # Decode ACEScct to linear Rec.2020
        start = time.time()
        img_linear = self._fast_acescct_decode(img)
        profile['acescct_decode'] = time.time() - start

        # Apply white balance adjustment
        start = time.time()
        img_linear = self._apply_white_balance(
            img_linear,
            self.user_settings['wb_temp'],
            self.user_settings['tint']
        )
        profile['wb_tint'] = time.time() - start

        # Apply exposure adjustment
        start = time.time()
        exposure_mult = 2 ** self.user_settings['exposure_ev']
        img_linear *= exposure_mult
        profile['exposure'] = time.time() - start

        # Re-encode to ACEScct for LUT
        start = time.time()
        img_acescct = self._fast_acescct_encode(img_linear)
        profile['acescct_encode'] = time.time() - start

        # ANTI-BANDING: Add subtle dither before LUT to mask quantization
        if DebugConfig.enable_pre_lut_dither and DebugConfig.pre_lut_dither_strength > 0 and not is_fast_mode:
            img_acescct = add_blue_noise_dither(img_acescct, DebugConfig.pre_lut_dither_strength)

        # Apply LUT FIRST (before softness to avoid revealing LUT banding)
        start = time.time()
        if DebugConfig.enable_lut and self.lut_preview is not None:
            try:
                img_display = apply_lut_fast(img_acescct, self.lut_preview)
            except Exception as e:
                print(f"    [LUT ERROR]: {e}")
                img_display = img_acescct
        else:
            img_display = img_acescct
        profile['lut'] = time.time() - start

        # Chromatic aberration after LUT — skipped in fast mode for responsiveness
        start = time.time()
        if DebugConfig.enable_chromatic_aberration and DebugConfig.ca_strength > 0 and not is_fast_mode:
            img_display = apply_chromatic_aberration(img_display, DebugConfig.ca_strength, DebugConfig.ca_steps)
        profile['chromatic_aberration'] = time.time() - start

        # Apply softness AFTER LUT (will blur any banding artifacts from LUT)
        start = time.time()
        if DebugConfig.enable_softness and DebugConfig.softness_sigma > 0 and not is_fast_mode:
            img_display = apply_softness(img_display, DebugConfig.softness_sigma)
        profile['softness'] = time.time() - start

        # Grain in preview
        start = time.time()
        if DebugConfig.enable_grain and DebugConfig.grain_strength > 0 and not is_fast_mode:
            img_display = self.apply_grain_linear_light(img_display, DebugConfig.grain_strength)
        profile['grain'] = time.time() - start

        # Sharpen in preview
        start = time.time()
        if DebugConfig.enable_sharpen and DebugConfig.sharpen_strength > 0 and not is_fast_mode:
            img_display = apply_sharpen(img_display, DebugConfig.sharpen_strength, DebugConfig.sharpen_radius)
        profile['sharpen'] = time.time() - start

        profile['total'] = time.time() - total_start

        if is_fast_mode:
            img_display = cv2.resize(img_display, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        _timing_print("\n=== FAST RENDER PROFILE (preview - no halation) ===")
        for key, value in profile.items():
            _timing_print(f"  {key:20s}: {value*1000:6.2f} ms")
        _timing_print("===========================\n")

        return np.clip(img_display, 0, 1)

    def _render_hq(self):
        """
        Currently delegates to fast preview.
        Kept because the UI calls this mode explicitly — may be expanded later.
        """
        return self._render_fast()

    def render_export(self):
        """
        FULL QUALITY render for export.
        Includes all effects: halation, softness, grain, sharpen.
        No downscaling — native resolution output.
        """
        profile = {}
        total_start = time.time()

        img = self.intermediate_acescct.copy()

        start = time.time()
        img_linear = self._fast_acescct_decode(img)
        profile['acescct_decode'] = time.time() - start

        start = time.time()
        img_linear = self._apply_white_balance(
            img_linear,
            self.user_settings['wb_temp'],
            self.user_settings['tint']
        )
        profile['wb_tint'] = time.time() - start

        start = time.time()
        exposure_mult = 2 ** self.user_settings['exposure_ev']
        img_linear *= exposure_mult
        profile['exposure'] = time.time() - start

        start = time.time()
        img_acescct = self._fast_acescct_encode(img_linear)
        profile['acescct_encode'] = time.time() - start

        start = time.time()
        if DebugConfig.enable_softness and DebugConfig.softness_sigma > 0:
            img_acescct = apply_softness(img_acescct, DebugConfig.softness_sigma * self.pixel_scale)
        profile['softness'] = time.time() - start

        start = time.time()
        if self.lut_full is not None:
            try:
                img_display = apply_lut_fast(img_acescct, self.lut_full)
            except Exception as e:
                print(f"    [LUT ERROR]: {e}")
                img_display = img_acescct
        elif self.lut_preview is not None:
            img_display = apply_lut_fast(img_acescct, self.lut_preview)
        else:
            img_display = img_acescct
        profile['lut'] = time.time() - start

        start = time.time()
        if DebugConfig.enable_chromatic_aberration and DebugConfig.ca_strength > 0:
            img_display = apply_chromatic_aberration(img_display, DebugConfig.ca_strength, DebugConfig.ca_steps)
        profile['chromatic_aberration'] = time.time() - start

        start = time.time()
        if DebugConfig.enable_grain and DebugConfig.grain_strength > 0:
            img_display = self.apply_grain_linear_light(
                img_display, DebugConfig.grain_strength, scale=self.pixel_scale
            )
        profile['grain'] = time.time() - start

        start = time.time()
        if DebugConfig.enable_sharpen and DebugConfig.sharpen_strength > 0:
            img_display = apply_sharpen(
                img_display,
                DebugConfig.sharpen_strength,
                DebugConfig.sharpen_radius * self.pixel_scale,
            )
        profile['sharpen'] = time.time() - start

        profile['total'] = time.time() - total_start

        _timing_print("\n=== EXPORT RENDER PROFILE (full quality with all effects) ===")
        for key, value in profile.items():
            _timing_print(f"  {key:20s}: {value*1000:6.2f} ms")
        _timing_print("===========================\n")

        return np.clip(img_display, 0, 1)

    def _desaturate_highlights_lab(self, img_srgb_lin, threshold_L=80.0, rolloff_L=15.0, sigma=3.0):
        """
        Soft highlight desaturation in Lab space. Reduces a* and b* toward zero
        in bright regions without touching L* (perceptual lightness), so there is
        no blowout or luma shift.
        Handles gamma encode/decode internally; input and output are linear sRGB.
        """
        img_gamma = (np.clip(img_srgb_lin, 0.0, 1.0) ** (1.0 / 2.2)).astype(np.float32)
        img_lab = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2Lab)

        L = img_lab[:, :, 0]  # L* in [0, 100]
        mask = np.clip((L - threshold_L) / max(rolloff_L, 1e-6), 0.0, 1.0)
        if sigma > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)

        img_lab[:, :, 1] *= (1.0 - mask)  # a*
        img_lab[:, :, 2] *= (1.0 - mask)  # b*

        img_gamma_out = cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB)
        return np.clip(img_gamma_out ** 2.2, 0.0, 1.0).astype(np.float32)

    def _apply_white_balance(self, img_linear, temp_offset, tint_offset):
        """
        Apply white balance adjustment in linear space.
        temp_offset: Kelvin offset from neutral (-1000 to +1000)
        tint_offset: Magenta/Green shift (-10 to +10)
        """
        temp_factor = temp_offset / 1000.0

        r_mult = 1.0 + (temp_factor * 0.15)
        b_mult = 1.0 - (temp_factor * 0.15)
        g_mult = 1.0 - (tint_offset * 0.015)

        img_wb = img_linear.copy()
        img_wb[:, :, 0] *= r_mult
        img_wb[:, :, 1] *= g_mult
        img_wb[:, :, 2] *= b_mult

        return img_wb

    def get_settings(self):
        """Return current user settings."""
        return self.user_settings.copy()

    def set_settings(self, settings):
        """Load settings (for copy/paste)."""
        self.user_settings.update(settings)
        self.preview_mode = "hq"
        return self.render_preview()


# =============================================================================
# EXPORT
# =============================================================================

def export_image(processor, output_path, quality=95, as_tiff=False):
    """
    Export current image.
    If as_tiff is False: exports final 8-bit JPEG with all effects and LUT.
    If as_tiff is True: exports 16-bit intermediate ACEScct TIFF (no LUT).
    Uses OpenCV natively for TIFFs to ensure PyInstaller compatibility.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"✗ Cannot create output directory: {e}")
            return False

    try:
        ext = os.path.splitext(output_path)[1].lower()
        is_tiff_export = as_tiff or ext in ['.tif', '.tiff']

        if is_tiff_export:
            # TIFF EXPORT (OpenCV - PyInstaller Safe)
            img_array = processor.intermediate_acescct
            if img_array is None:
                print("✗ No image loaded to export")
                return False

            img_export = np.clip(img_array * 65535.0, 0, 65535).astype(np.uint16)
            img_bgr = cv2.cvtColor(img_export, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(output_path, img_bgr)

            if success:
                print(f"✓ Exported (OpenCV TIFF): {output_path}")
                return True
            else:
                print(f"✗ OpenCV failed to write TIFF to {output_path}")
                return False

        else:
            # JPEG EXPORT (Standard Fallback Chain)
            use_full_res = (
                DebugConfig.experimental_full_res_export
                and processor.current_file is not None
                and not processor.is_full_res
            )
            saved_state = None
            if use_full_res:
                root, ext_out = os.path.splitext(output_path)
                output_path = f"{root}_full{ext_out}"
                print("→ Experimental full-res reprocess for export…")
                saved_state = {
                    'intermediate': processor.intermediate_acescct,
                    'is_full_res': processor.is_full_res,
                    'pixel_scale': processor.pixel_scale,
                    'applied_rotation': processor.applied_rotation,
                    'preview_mode': processor.preview_mode,
                }
                try:
                    processor.load_image(processor.current_file, for_export=True, full_res=True)

                    # Re-apply any rotation the user had on the preview.
                    # Orientation offsets between half/full rawpy decode are left to the user
                    # to correct manually via the rotate button.
                    prev_rot = saved_state['applied_rotation']
                    if prev_rot:
                        processor.rotation = prev_rot
                        processor._apply_rotation_and_render()
                    processor.preview_mode = "hq"
                except Exception as e:
                    print(f"✗ Full-res reload failed, falling back to preview resolution: {e}")
                    # Restore preview state
                    processor.intermediate_acescct = saved_state['intermediate']
                    processor.is_full_res = saved_state['is_full_res']
                    processor.pixel_scale = saved_state['pixel_scale']
                    processor.applied_rotation = saved_state['applied_rotation']
                    processor.preview_mode = saved_state['preview_mode']
                    saved_state = None

            try:
                img_array = processor.render_export()
            finally:
                if saved_state is not None:
                    processor.intermediate_acescct = saved_state['intermediate']
                    processor.is_full_res = saved_state['is_full_res']
                    processor.pixel_scale = saved_state['pixel_scale']
                    processor.applied_rotation = saved_state['applied_rotation']
                    processor.preview_mode = saved_state['preview_mode']
            if img_array is None:
                print("✗ No image loaded to export")
                return False

            if img_array.dtype in [np.float32, np.float64]:
                img_8bit = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
            else:
                img_8bit = img_array

            # 1. Primary: PIL
            try:
                from PIL import Image
                pil_img = Image.fromarray(img_8bit, mode='RGB')
                pil_img.save(output_path, 'JPEG', quality=quality, optimize=True)
                print(f"✓ Exported (PIL): {output_path}")
                return True
            except ImportError:
                pass

            # 2. Fallback: imageio
            try:
                import imageio
                imageio.imwrite(output_path, img_8bit, format='JPEG', quality=quality)
                print(f"✓ Exported (imageio): {output_path}")
                return True
            except Exception as e:
                print(f"✗ imageio export failed: {e}")

                # 3. Last Resort: OpenCV
                try:
                    img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    print(f"✓ Exported (OpenCV JPEG): {output_path}")
                    return True
                except Exception as e2:
                    print(f"✗ OpenCV export also failed: {e2}")
                    raise

    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False
