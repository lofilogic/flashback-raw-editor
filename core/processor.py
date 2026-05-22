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
    FLASHBACK_CCM, FLASHBACK_CCM2, IPHONE_CCM, SENSOR_BLACK, BASE_WB_SETTINGS, BASE_WB_SETTINGS2, BASE_EXPOSURE_OFFSET,
    DebugConfig, REC2020_FROM_SRGB,
    GRAIN_STRENGTH, GRAIN_TILE_SCALE, GRAIN_HIGHLIGHT_BIAS, SOFTNESS_SIGMA, SHARPEN_STRENGTH, SHARPEN_RADIUS,
    _timing_print,
)
from .gpu import gpu
from .kernels import (
    acescct_decode,
    acescct_encode,
    rotate_90_clockwise,
    rotate_90_counterclockwise,
    apply_grain,
    gaussian_blur,
)
from .auto_exposure_reverse import extract_exposure_seconds, compute_reverse_gain
from .effects import (
    apply_lut_fast,
    apply_chromatic_aberration,
    reduce_color_noise_chroma,
    apply_halation,
    apply_softness,
    apply_sharpen,
    apply_vignette,
    apply_bloom,
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
    Handles image processing for preview and export.

    Architecture:
    1. Load RAW → Preprocess to ACEScct intermediate (slow, once per image)
    2. Preview render: Decode → WB/Exposure → Encode → Effects → LUT
       — pass downscale=True for responsive slider scrubbing
    3. Export render: full-quality with halation, softness, grain, sharpen
    """

    def __init__(self, lut_path=None):
        """Initialize processor with LUT."""
        self.intermediate_acescct = None
        self.current_file = None
        self.rotation = 0  # 0, 90, 180, 270 degrees
        self.grain_tiles = []

        # User-adjustable settings
        self.user_settings = {
            'exposure_ev': 0.0,   # -2 to +2 EV
            'wb_temp': 0,         # Temperature offset in Kelvin (-1000 to +1000)
            'tint': 0.0           # Tint offset (-10 to +10)
        }

        self._load_grain_tiles()

        self.lut = None

        if lut_path and os.path.exists(lut_path):
            try:
                self.lut = colour.read_LUT(lut_path)
                print(f"✓ LUT loaded: {self.lut.name} ({self.lut.table.shape})")
                gpu.upload_lut(self.lut.table)
                print(f"✓ LUT uploaded to GPU ({self.lut.table.shape})")
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
                if GRAIN_TILE_SCALE != 1.0:
                    new_h = max(1, int(round(tile.shape[0] * GRAIN_TILE_SCALE)))
                    new_w = max(1, int(round(tile.shape[1] * GRAIN_TILE_SCALE)))
                    tile = cv2.resize(tile, (new_w, new_h), interpolation=cv2.INTER_AREA)
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

    def apply_grain_linear_light(self, image, strength=GRAIN_STRENGTH):
        """Apply grain with pre-rendered tiles. `strength` scales the blend
        intensity in tile mode (in procedural fallback it's the noise sigma)."""
        h, w = image.shape[:2]
        grain = self.generate_grain_layer(h, w, sigma=strength)

        return apply_grain(image, grain, intensity=strength, highlight_bias=GRAIN_HIGHLIGHT_BIAS)

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

        self.rotation = 0
        return self.render_preview()

    def _rotate_90(self, img, clockwise=True):
        if clockwise:
            return rotate_90_clockwise(img)
        else:
            return rotate_90_counterclockwise(img)

    def _rotate_180(self, img):
        """Rotate 180 degrees using OpenCV."""
        return cv2.rotate(img, cv2.ROTATE_180)

    def get_rotation(self):
        """Get current rotation in degrees."""
        return self.rotation

    def _fast_acescct_decode(self, img):
        return acescct_decode(img)

    def _fast_acescct_encode(self, img):
        return acescct_encode(img)

    def load_image(self, dng_path, for_export=False, fast_mode=False):
        """
        Load and preprocess RAW image to ACEScct intermediate.
        This is the slow step, done once per image.

        Args:
            dng_path: Path to DNG file
            for_export: If True, apply halation (slower)
            fast_mode: If True, use LINEAR demosaic + clip highlights (10-20x faster, for thumbnails)
        Returns:
            preview image (numpy array)
        """
        total_start = time.time()

        _timing_print(f"\n{'='*60}")
        _timing_print(f"Loading: {os.path.basename(dng_path)} (fast={fast_mode})")
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
                    # Non-Flashback pipeline: daylight WB pre-applied, then a
                    # per-camera CCM fit via tools/match_camera.py maps the
                    # camera's raw RGB into Flashback-style linear sRGB so the
                    # same LUT lands consistently.
                    rgb_linear = raw.postprocess(
                        demosaic_algorithm=demosaic_fb,
                        use_camera_wb=False,
                        use_auto_wb=False,
                        user_wb=list(raw.daylight_whitebalance),
                        half_size=True,
                        no_auto_bright=True,
                        bright=1,
                        highlight_mode=highlight_mode,
                        gamma=(1, 1),
                        output_bps=16,
                        output_color=rawpy.ColorSpace.raw,
                    ).astype(np.float32) / 65535.0

                    profile['raw_develop'] = (time.time() - start) * 1000

                    start = time.time()
                    print("Applying iPhone CCM...")
                    img_srgb_lin = (rgb_linear.reshape(-1, 3) @ IPHONE_CCM.T).reshape(rgb_linear.shape)
                    img_srgb_lin = np.clip(img_srgb_lin, 0.0, 1.0)
                    profile['color_matrix'] = (time.time() - start) * 1000

                    # Soft Lab desaturation in highlights
                    if DebugConfig.enable_highlight_desat:
                        img_srgb_lin = self._desaturate_highlights_lab(
                            img_srgb_lin,
                            threshold_L=DebugConfig.highlight_desat_threshold_L - 19, # iPhone highlights are brighter, so shift threshold down a bit
                            rolloff_L=DebugConfig.highlight_desat_rolloff_L,
                            sigma=DebugConfig.highlight_desat_sigma,
                        )
                    _timing_print(f"    After CCM: [{img_srgb_lin.min():.4f}, {img_srgb_lin.max():.4f}]")

                else:
                    # --- FLASHBACK ONE35 V2 PIPELINE ---
                    rgb_linear = raw.postprocess(
                        demosaic_algorithm=demosaic_fb,
                        user_wb=BASE_WB_SETTINGS,
                        user_black=SENSOR_BLACK,
                        half_size=True,
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

            # Reverse the camera's autoexposure so absolute scene brightness is
            # preserved — bright scenes stay bright, dark scenes stay dark.
            if DebugConfig.enable_reverse_autoexposure and not is_not_flashback:
                exp_s = extract_exposure_seconds(dng_path)
                gain = compute_reverse_gain(exp_s, DebugConfig.reverse_autoexposure_t_ref)
                img_rec2020_lin *= gain
                _timing_print(f"  Reverse AE: ExposureTime={exp_s}s, T_ref={DebugConfig.reverse_autoexposure_t_ref}s, gain={gain:.3f}")

            # Static post-AE exposure boost — must match the value used to
            # train the LUT (tools/build_color_charts.py).
            if DebugConfig.enable_post_ae_exposure_boost:
                img_rec2020_lin *= 2.0 ** DebugConfig.post_ae_exposure_boost_ev
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
                    DebugConfig.halation_blur_radius,
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
                    self.intermediate_acescct, sigma=DebugConfig.cnr_sigma
                )
                profile['cnr'] = (time.time() - start) * 1000
                _timing_print(f"    -> {profile['cnr']:6.2f} ms")

            # Render preview
            start = time.time()
            result = self.render_preview(downscale=fast_mode)

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

            _timing_print(f"✓ Loaded intermediate TIFF in {(time.time() - start)*1000:.2f} ms")

            return self.render_preview()

        except Exception as e:
            print(f"✗ Failed to load intermediate TIFF: {e}")
            return None

    def render_preview(self, downscale=False):
        """Main render function. Pass downscale=True for fast slider scrub previews."""
        if self.intermediate_acescct is None:
            return None
        return self._render_fast(downscale=downscale)

    def _render_fast(self, downscale=False):
        """
        Fast preview render. Applies full effect chain except halation.
        Target: ~100-150ms for responsive editing.
        """
        profile = {}
        total_start = time.time()

        img = self.intermediate_acescct.copy()
        profile['copy'] = time.time() - total_start

        orig_h, orig_w = img.shape[:2]

        if downscale:
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

        # Vignette + bloom in linear light (lens effects before color grading)
        start = time.time()
        if DebugConfig.enable_bloom and DebugConfig.bloom_strength > 0 and not downscale:
            img_linear = apply_bloom(img_linear, DebugConfig.bloom_strength, DebugConfig.bloom_threshold, linear=True)
        profile['bloom'] = time.time() - start

        start = time.time()
        if DebugConfig.enable_vignette and DebugConfig.vignette_strength > 0 and not downscale:
            img_linear = apply_vignette(img_linear, DebugConfig.vignette_strength, DebugConfig.vignette_color_shift, DebugConfig.vignette_feather)
        profile['vignette'] = time.time() - start

        # Re-encode to ACEScct for LUT
        start = time.time()
        img_acescct = self._fast_acescct_encode(img_linear)
        profile['acescct_encode'] = time.time() - start

        # Apply LUT FIRST (before softness to avoid revealing LUT banding)
        start = time.time()
        if DebugConfig.enable_lut and self.lut is not None:
            try:
                img_display = apply_lut_fast(img_acescct, self.lut)
            except Exception as e:
                print(f"    [LUT ERROR]: {e}")
                img_display = img_acescct
        else:
            img_display = img_acescct
        profile['lut'] = time.time() - start

        # Chromatic aberration after LUT — skipped in fast mode for responsiveness
        start = time.time()
        if DebugConfig.enable_chromatic_aberration and DebugConfig.ca_strength > 0 and not downscale:
            img_display = apply_chromatic_aberration(img_display, DebugConfig.ca_strength, DebugConfig.ca_steps, DebugConfig.ca_blue_blur)
        profile['chromatic_aberration'] = time.time() - start

        # Dither AFTER CA so the noise pattern doesn't get streaked by CA's
        # radial scaling at the corners.
        if DebugConfig.enable_pre_lut_dither and DebugConfig.pre_lut_dither_strength > 0 and not downscale:
            img_display = add_blue_noise_dither(img_display, DebugConfig.pre_lut_dither_strength)

        # Apply softness AFTER LUT (will blur any banding artifacts from LUT)
        start = time.time()
        if DebugConfig.enable_softness and DebugConfig.softness_sigma > 0 and not downscale:
            img_display = apply_softness(img_display, DebugConfig.softness_sigma)
        profile['softness'] = time.time() - start

        # Grain in preview
        start = time.time()
        if DebugConfig.enable_grain and DebugConfig.grain_strength > 0 and not downscale:
            img_display = self.apply_grain_linear_light(img_display, DebugConfig.grain_strength)
        profile['grain'] = time.time() - start

        # Sharpen in preview
        start = time.time()
        if DebugConfig.enable_sharpen and DebugConfig.sharpen_strength > 0 and not downscale:
            img_display = apply_sharpen(img_display, DebugConfig.sharpen_strength, DebugConfig.sharpen_radius)
        profile['sharpen'] = time.time() - start

        profile['total'] = time.time() - total_start

        _timing_print("\n=== FAST RENDER PROFILE (preview - no halation) ===")
        for key, value in profile.items():
            _timing_print(f"  {key:20s}: {value*1000:6.2f} ms")
        _timing_print("===========================\n")

        # Return the small image as-is when downscaling — the caller (ZoomableImageWidget
        # via Qt SmoothTransformation) handles the upscale on the GPU, which is faster
        # than doing a cv2.resize + full-res numpy→QImage conversion here.
        return np.clip(img_display, 0, 1)

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

        # Vignette + bloom in linear light (lens effects before color grading)
        start = time.time()
        if DebugConfig.enable_bloom and DebugConfig.bloom_strength > 0:
            img_linear = apply_bloom(img_linear, DebugConfig.bloom_strength, DebugConfig.bloom_threshold, linear=True)
        profile['bloom'] = time.time() - start

        start = time.time()
        if DebugConfig.enable_vignette and DebugConfig.vignette_strength > 0:
            img_linear = apply_vignette(img_linear, DebugConfig.vignette_strength, DebugConfig.vignette_color_shift, DebugConfig.vignette_feather)
        profile['vignette'] = time.time() - start

        start = time.time()
        img_acescct = self._fast_acescct_encode(img_linear)
        profile['acescct_encode'] = time.time() - start

        start = time.time()
        if DebugConfig.enable_softness and DebugConfig.softness_sigma > 0:
            img_acescct = apply_softness(img_acescct, DebugConfig.softness_sigma)
        profile['softness'] = time.time() - start

        start = time.time()
        if self.lut is not None:
            try:
                img_display = apply_lut_fast(img_acescct, self.lut)
            except Exception as e:
                print(f"    [LUT ERROR]: {e}")
                img_display = img_acescct
        else:
            img_display = img_acescct
        profile['lut'] = time.time() - start

        start = time.time()
        if DebugConfig.enable_chromatic_aberration and DebugConfig.ca_strength > 0:
            img_display = apply_chromatic_aberration(img_display, DebugConfig.ca_strength, DebugConfig.ca_steps, DebugConfig.ca_blue_blur)
        profile['chromatic_aberration'] = time.time() - start

        start = time.time()
        if DebugConfig.enable_grain and DebugConfig.grain_strength > 0:
            img_display = self.apply_grain_linear_light(img_display, DebugConfig.grain_strength)
        profile['grain'] = time.time() - start

        start = time.time()
        if DebugConfig.enable_sharpen and DebugConfig.sharpen_strength > 0:
            img_display = apply_sharpen(
                img_display,
                DebugConfig.sharpen_strength,
                DebugConfig.sharpen_radius,
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
            mask = gaussian_blur(mask, sigma)

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
        """Update user_settings in-place. Does not render — call render_preview()
        separately if you need a fresh preview."""
        self.user_settings.update(settings)


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
            img_array = processor.render_export()
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
