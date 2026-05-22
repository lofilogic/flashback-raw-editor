"""
Flashback One35 raw processor — DNG-spec color pipeline.

Pipeline (Flashback DNG):
    rawpy.postprocess(user_wb=[1,1,1,1], user_black=SENSOR_BLACK,
                      gamma=(1,1), output_color=raw, output_bps=16)
      -> raw RGB (pre-WB)
    highlight recovery (darktable "inpaint opposed", pre-WB)
    raw_wb = raw / ASN_D50
    XYZ_D50 = FM1 @ raw_wb
    ACEScg = XYZ_D50_TO_ACESCG @ XYZ_D50          <- cached intermediate
    user WB + tint + exposure + push/pull (linear ACEScg)
    bloom / vignette (linear ACEScg)
    CNR in Lab space
    ACEScct encode -> LUT -> post-LUT effects -> display sRGB
"""
import os
import time
from pathlib import Path
import numpy as np
import rawpy
import cv2
import exifread
import colour

from . import resource_path
from .config import (
    SENSOR_BLACK, GRAIN_TILE_SCALE, GRAIN_HIGHLIGHT_BIAS, PUSH_PULL_RANGE_EV,
    _timing_print, DebugConfig,
)
from .dng_export import _PROFILE_TONE_CURVE
from .kernels import acescct_encode, apply_grain
from .auto_exposure_reverse import extract_exposure_seconds, compute_reverse_gain
from .effects import (
    apply_lut_fast,
    apply_chromatic_aberration,
    apply_halation,
    apply_softness,
    apply_sharpen,
    apply_vignette,
    apply_bloom,
    reduce_color_noise_chroma,
)


# =============================================================================
# COLOR MATRICES (DNG dual-illuminant)
# =============================================================================

# AsShotNeutral from real D50 grey-patch measurement (matches the asn
# embedded in DNGs by core/dng_export.py and used by Camera Raw at render
# time).
ASN_D50 = np.array([0.541, 1.0, 0.597], dtype=np.float32)

# ForwardMatrix1: camera_wb_rgb (raw / ASN) -> XYZ_D50.
# Calibrated under D55 daylight.
FM1 = np.array([
    [0.53086, 0.22116, 0.21219],
    [0.08570, 0.98930, -0.07500],
    [0.04526, -0.37228, 1.15192],
], dtype=np.float32)

FM1_RAW_TO_XYZ_D50 = FM1 / ASN_D50[np.newaxis, :]
ASN_INV = (1.0 / ASN_D50).astype(np.float32)


# =============================================================================
# ACES / DISPLAY MATRICES
# =============================================================================

# Bradford CAT D50 -> D60 (ACES adopted white).
BRADFORD_D50_TO_D60 = np.array([
    [ 0.98722400, -0.00611327,  0.01595330],
    [-0.00759836,  1.00186000,  0.00533002],
    [ 0.00307257, -0.00509595,  1.08168000],
], dtype=np.float32)

# XYZ_D60 -> ACEScg (AP1).
XYZ_D60_TO_ACESCG = np.array([
    [ 1.6410233797, -0.3248032942, -0.2364246952],
    [-0.6636628587,  1.6153315917,  0.0167563477],
    [ 0.0117218943, -0.0082844420,  0.9883948585],
], dtype=np.float32)

# Fused: XYZ_D50 -> ACEScg.
XYZ_D50_TO_ACESCG = (XYZ_D60_TO_ACESCG @ BRADFORD_D50_TO_D60).astype(np.float32)

# Fused: raw -> ACEScg (fast path, no highlight recovery).
RAW_TO_ACESCG = (XYZ_D50_TO_ACESCG @ FM1_RAW_TO_XYZ_D50).astype(np.float32)

# ACEScg -> linear sRGB.
ACESCG_TO_LINSRGB = np.array([
    [ 1.70505, -0.62179, -0.08326],
    [-0.13026,  1.14080, -0.01055],
    [-0.02400, -0.12897,  1.15297],
], dtype=np.float32)

_CS_PROPHOTO = colour.RGB_COLOURSPACES['ProPhoto RGB']
_CS_ACESCG   = colour.RGB_COLOURSPACES['ACEScg']
_CS_SRGB     = colour.RGB_COLOURSPACES['sRGB']
_CAT = 'CAT02'
ACESCG_TO_PROPHOTO = colour.RGB_to_RGB(
    np.eye(3, dtype=np.float32), _CS_ACESCG, _CS_PROPHOTO,
    chromatic_adaptation_transform=_CAT
).astype(np.float32)
PROPHOTO_TO_LINSRGB = colour.RGB_to_RGB(
    np.eye(3, dtype=np.float32), _CS_PROPHOTO, _CS_SRGB,
    chromatic_adaptation_transform=_CAT
).astype(np.float32)


# =============================================================================
# USER WB + EXPOSURE
# =============================================================================

# Slider zero = D55, matching FM1's calibration illuminant.
BASE_KELVIN = 5500.0

_XYZ_TO_AP1_PURE = XYZ_D60_TO_ACESCG.astype(np.float32)


def _planckian_xyz(cct: float) -> np.ndarray:
    if cct >= 4000.0:
        xy = np.asarray(colour.temperature.CCT_to_xy_CIE_D(cct))
    else:
        xy = np.asarray(colour.temperature.CCT_to_xy_Kang2002(cct))
    return np.asarray(colour.xy_to_XYZ(xy), dtype=np.float32)


def _kelvin_to_acescg_gain(target_k: float,
                           base_k: float = BASE_KELVIN) -> np.ndarray:
    """Per-channel ACEScg gain to shift white balance. G is normalized to 1."""
    base_ap1   = _XYZ_TO_AP1_PURE @ _planckian_xyz(base_k)
    target_ap1 = _XYZ_TO_AP1_PURE @ _planckian_xyz(target_k)
    gain = base_ap1 / target_ap1
    return (gain / gain[1]).astype(np.float32)


def _tint_to_acescg_gain(tint_offset: float) -> np.ndarray:
    """Green-magenta correction. Positive = magenta (decreases G)."""
    g_mult = 1.0 / (1.0 + tint_offset * 0.018)
    return np.array([1.0, g_mult, 1.0], dtype=np.float32)


# =============================================================================
# DISPLAY TRANSFORM
# =============================================================================

def _srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)


def _srgb_eotf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, np.power((x + a) / (1 + a), 2.4))


def _build_tone_curve_lut(curve_pairs: list, size: int = 1024) -> np.ndarray:
    pts = np.array(curve_pairs, dtype=np.float32).reshape(-1, 2)
    xs, ys = pts[:, 0], pts[:, 1]
    domain = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return np.interp(domain, xs, ys).astype(np.float32)


_TONE_CURVE_LUT = _build_tone_curve_lut(_PROFILE_TONE_CURVE, size=4096)


def _apply_tone_curve(x: np.ndarray) -> np.ndarray:
    n = _TONE_CURVE_LUT.shape[0]
    idx = np.clip(x * (n - 1), 0, n - 1).astype(np.int32)
    return _TONE_CURVE_LUT[idx]


# =============================================================================
# HIGHLIGHT RECOVERY (darktable "inpaint opposed")
# =============================================================================

def _recover_highlights(rgb_raw: np.ndarray, asn: np.ndarray,
                        threshold: float = 0.95) -> np.ndarray:
    """Highlight recovery operating in raw space (pre-WB).

    For each clipped channel, estimates the lost value as the cube of the
    average of the cube-roots of the other two channels, with a global
    chrominance correction to preserve the scene's local color cast.
    """
    asn_inv = (1.0 / asn).astype(rgb_raw.dtype)
    rgb_wb  = rgb_raw * asn_inv

    clipped = rgb_raw >= threshold
    if not clipped.any():
        return rgb_wb.astype(np.float32)

    rgb_cbrt = np.cbrt(np.maximum(0.0, rgb_wb))
    R, G, B  = rgb_cbrt[..., 0], rgb_cbrt[..., 1], rgb_cbrt[..., 2]
    refavg = np.stack([
        ((G + B) * 0.5) ** 3,
        ((R + B) * 0.5) ** 3,
        ((R + G) * 0.5) ** 3,
    ], axis=-1).astype(rgb_wb.dtype)

    chrominance    = np.zeros(3, dtype=rgb_wb.dtype)
    clipped_any_u8 = clipped.any(axis=-1).astype(np.uint8)
    if clipped_any_u8.any():
        kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        near_clip = cv2.dilate(clipped_any_u8, kernel).astype(bool)
        for c in range(3):
            sample = near_clip & ~clipped[..., c]
            if sample.sum() > 30:
                chrominance[c] = float(
                    (rgb_wb[..., c] - refavg[..., c])[sample].mean()
                )

    target = refavg + chrominance.reshape(1, 1, 3)
    out = np.where(clipped, np.maximum(rgb_wb, target), rgb_wb)
    return out.astype(np.float32)


# =============================================================================
# UTILITIES
# =============================================================================

def _is_flashback_dng(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False, stop_tag='Image Make')
        return str(tags.get('Image Make', '')).strip().lower() == 'flashback'
    except Exception:
        return False


# =============================================================================
# PROCESSOR
# =============================================================================

class FlashbackProcessor:
    """Raw processor for Flashback DNGs."""

    def __init__(self, lut_path=None):
        self.intermediate_acescg = None
        self.current_file = None
        self._rev_gain = 1.0
        self.rotation = 0
        self.user_settings = {
            'exposure_ev': 0.0,
            'wb_temp': 0,
            'tint': 0.0,
            'push_pull_ev': 0.0,
        }
        self.highlight_mode = 1
        self.rawpy_bright = 1.0
        self.enable_highlight_recovery = True

        self.lut = None
        path = lut_path or DebugConfig.lut_path
        if path and os.path.exists(path):
            try:
                self.lut = colour.read_LUT(path)
                print(f"[processor] LUT loaded: {self.lut.name} ({self.lut.table.shape})")
            except Exception as e:
                print(f"[processor] Could not load LUT {path}: {e}")

        self.grain_tiles = []
        self._load_grain_tiles()

    # ---- grain ----------------------------------------------------------------

    def _load_grain_tiles(self):
        grain_dir = Path(resource_path("assets/grain"))
        if not grain_dir.exists():
            return
        for path in sorted(grain_dir.glob("*.png")) + sorted(grain_dir.glob("*.jpg")):
            try:
                tile = cv2.imread(str(path), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
                tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                if GRAIN_TILE_SCALE != 1.0:
                    nh = max(1, int(round(tile.shape[0] * GRAIN_TILE_SCALE)))
                    nw = max(1, int(round(tile.shape[1] * GRAIN_TILE_SCALE)))
                    tile = cv2.resize(tile, (nw, nh), interpolation=cv2.INTER_AREA)
                self.grain_tiles.append(tile)
            except Exception:
                pass

    def _generate_grain_layer(self, height, width, sigma):
        if not self.grain_tiles:
            grain = np.full((height, width, 3), 0.5, dtype=np.float32)
            return np.clip(grain + np.random.normal(0, sigma, (height, width, 3)).astype(np.float32), 0, 1)
        out = np.zeros((height, width, 3), dtype=np.float32)
        th, tw = self.grain_tiles[0].shape[:2]
        for y in range(0, height, th):
            for x in range(0, width, tw):
                tile = self.grain_tiles[np.random.randint(0, len(self.grain_tiles))].copy()
                if np.random.random() > 0.5:
                    tile = np.flip(tile, axis=1)
                if np.random.random() > 0.5:
                    tile = np.flip(tile, axis=0)
                he = min(y + th, height); we = min(x + tw, width)
                out[y:he, x:we] = tile[:he - y, :we - x]
        return out

    @staticmethod
    def _grain_highlight_bias(film_ev_driver):
        if not film_ev_driver:
            return GRAIN_HIGHLIGHT_BIAS
        frac   = min(abs(film_ev_driver) / PUSH_PULL_RANGE_EV, 1.0)
        target = 1.0 if film_ev_driver < 0 else 0.0
        return float(GRAIN_HIGHLIGHT_BIAS + (target - GRAIN_HIGHLIGHT_BIAS) * frac)

    def _apply_grain(self, image, strength, highlight_bias=None):
        h, w = image.shape[:2]
        grain = self._generate_grain_layer(h, w, sigma=strength)
        if highlight_bias is None:
            highlight_bias = GRAIN_HIGHLIGHT_BIAS
        return apply_grain(image, grain, intensity=strength,
                           highlight_bias=highlight_bias)

    # ---- public surface -------------------------------------------------------

    def get_settings(self):
        return self.user_settings.copy()

    def set_settings(self, settings):
        self.user_settings.update(settings)

    def rotate_clockwise(self):
        self.rotation = (self.rotation + 90) % 360
        return self._apply_rotation_and_render()

    def rotate_counterclockwise(self):
        self.rotation = (self.rotation - 90) % 360
        return self._apply_rotation_and_render()

    def get_rotation(self):
        return self.rotation

    @property
    def intermediate_acescct(self):
        return self.intermediate_acescg

    @intermediate_acescct.setter
    def intermediate_acescct(self, value):
        self.intermediate_acescg = value

    def _render_fast(self, downscale=False):
        return self._render(downscale=downscale)

    # ---- pipeline -------------------------------------------------------------

    @staticmethod
    def _reverse_ae_gain_for(path):
        if not (path and DebugConfig.enable_reverse_autoexposure):
            return 1.0
        exp_s = extract_exposure_seconds(path)
        return float(compute_reverse_gain(
            exp_s, DebugConfig.reverse_autoexposure_t_ref))

    def _bake_halation(self, acescg):
        if not (DebugConfig.enable_halation and DebugConfig.halation_strength > 0):
            return acescg
        return apply_halation(
            acescg,
            DebugConfig.halation_threshold,
            DebugConfig.halation_blur_radius,
            DebugConfig.halation_strength,
        )

    def load_image(self, dng_path, for_export=False, fast_mode=False):
        total_start = time.time()
        _timing_print(f"\n{'='*60}")
        _timing_print(f"[processor] Loading: {os.path.basename(dng_path)} (fast={fast_mode})")
        _timing_print(f"{'='*60}")

        self.current_file = dng_path

        if os.path.splitext(dng_path)[1].lower() in ('.tif', '.tiff'):
            print("[processor] TIFF import is not supported. Open the original DNG instead.")
            return None

        if not _is_flashback_dng(dng_path):
            print("[processor] Non-Flashback DNG — generic pipeline not yet implemented.")
            return None

        try:
            t0 = time.time()
            with rawpy.imread(dng_path) as raw:
                demosaic = (rawpy.DemosaicAlgorithm.LINEAR if fast_mode
                            else rawpy.DemosaicAlgorithm.AHD)
                rgb = raw.postprocess(
                    demosaic_algorithm=demosaic,
                    user_wb=[1.0, 1.0, 1.0, 1.0],
                    use_camera_wb=False,
                    use_auto_wb=False,
                    user_black=SENSOR_BLACK,
                    half_size=True,
                    no_auto_bright=True,
                    bright=self.rawpy_bright,
                    highlight_mode=self.highlight_mode,
                    gamma=(1, 1),
                    output_bps=16,
                    output_color=rawpy.ColorSpace.raw,
                ).astype(np.float32) / 65535.0
            _timing_print(f"  raw_develop: {(time.time()-t0)*1000:6.2f} ms  "
                          f"shape={rgb.shape}  range=[{rgb.min():.4f},{rgb.max():.4f}]")

            t0 = time.time()
            if self.enable_highlight_recovery:
                rgb_wb = _recover_highlights(rgb, ASN_D50)
                flat   = rgb_wb.reshape(-1, 3)
                xyz    = (flat @ FM1.T).reshape(rgb_wb.shape)
                acescg = (xyz.reshape(-1, 3) @ XYZ_D50_TO_ACESCG.T).reshape(xyz.shape)
            else:
                flat   = rgb.reshape(-1, 3)
                acescg = (flat @ RAW_TO_ACESCG.T).reshape(rgb.shape)
            _timing_print(f"  raw->ACEScg: {(time.time()-t0)*1000:6.2f} ms  "
                          f"range=[{acescg.min():.4f},{acescg.max():.4f}]")

            self._rev_gain = self._reverse_ae_gain_for(dng_path)

            if for_export:
                acescg = self._bake_halation(acescg)

            self.intermediate_acescg = np.ascontiguousarray(acescg, dtype=np.float32)

            result = self.render_preview(downscale=fast_mode)
            _timing_print(f"  TOTAL load: {(time.time()-total_start)*1000:6.2f} ms\n")
            return result

        except Exception as e:
            print(f"[processor] load failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def render_preview(self, downscale=False):
        if self.intermediate_acescg is None:
            return None
        return self._render(downscale=downscale)

    def render_export(self):
        return self._render(downscale=False)

    def _render(self, downscale=False):
        t0  = time.time()
        cfg = DebugConfig
        img = self.intermediate_acescg
        if downscale:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // 3, h // 3), interpolation=cv2.INTER_LINEAR)

        push_pull_ev = float(self.user_settings.get('push_pull_ev', 0.0))

        f        = float(cfg.reverse_ae_strength)
        rev_gain = self._rev_gain if cfg.enable_reverse_autoexposure else 1.0
        rev_ev   = float(np.log2(rev_gain)) if rev_gain > 0 else 0.0
        boost_ev = float(cfg.post_ae_exposure_boost_ev) if cfg.enable_post_ae_exposure_boost else 0.0
        pre_lut_ev = f * rev_ev + f * boost_ev + push_pull_ev

        wb   = _kelvin_to_acescg_gain(BASE_KELVIN + self.user_settings['wb_temp'])
        tint = _tint_to_acescg_gain(self.user_settings['tint'])
        ev   = float(2.0 ** (self.user_settings['exposure_ev'] + cfg.base_exposure_offset_v2 + pre_lut_ev))
        gain = (wb * tint * ev).astype(np.float32)
        if not np.allclose(gain, 1.0):
            img = img * gain

        if not downscale:
            if cfg.enable_bloom and cfg.bloom_strength > 0:
                img = apply_bloom(img, cfg.bloom_strength,
                                  cfg.bloom_threshold, linear=True)
            if cfg.enable_vignette and cfg.vignette_strength > 0:
                img = apply_vignette(img, cfg.vignette_strength,
                                     cfg.vignette_color_shift,
                                     cfg.vignette_feather)

        if cfg.enable_lut and self.lut is not None:
            if cfg.enable_cnr and cfg.cnr_sigma > 0:
                img = reduce_color_noise_chroma(img, sigma=cfg.cnr_sigma)
            img_acescct = acescct_encode(np.maximum(img, 1e-10))
            try:
                img_display = apply_lut_fast(img_acescct, self.lut)
            except Exception as e:
                print(f"[processor] LUT error: {e}")
                img_display = np.clip(img_acescct, 0, 1)
        else:
            flat     = img.reshape(-1, 3)
            prophoto = (flat @ ACESCG_TO_PROPHOTO).reshape(img.shape)
            prophoto = _apply_tone_curve(np.clip(prophoto, 0.0, 1.0))
            lin_srgb = (prophoto.reshape(-1, 3) @ PROPHOTO_TO_LINSRGB).reshape(prophoto.shape)
            img_display = _srgb_oetf(np.clip(lin_srgb, 0.0, 1.0))

        if not downscale:
            if cfg.enable_chromatic_aberration and cfg.ca_strength > 0:
                img_display = apply_chromatic_aberration(
                    img_display, cfg.ca_strength, cfg.ca_steps, cfg.ca_blue_blur)
            if cfg.enable_softness and cfg.softness_sigma > 0:
                img_display = apply_softness(img_display, cfg.softness_sigma)
            if cfg.enable_grain and cfg.grain_strength > 0:
                grain_driver = f * rev_ev + push_pull_ev
                img_display  = self._apply_grain(
                    img_display, cfg.grain_strength,
                    highlight_bias=self._grain_highlight_bias(grain_driver))
            if cfg.enable_sharpen and cfg.sharpen_strength > 0:
                img_display = apply_sharpen(
                    img_display, cfg.sharpen_strength, cfg.sharpen_radius)

        post_gain = 2.0 ** (-pre_lut_ev)
        if not np.isclose(post_gain, 1.0):
            lin = _srgb_eotf(img_display) * post_gain
            img_display = _srgb_oetf(np.clip(lin, 0.0, 1.0))

        _timing_print(f"  render: {(time.time()-t0)*1000:6.2f} ms")
        return np.clip(img_display, 0.0, 1.0)

    # ---- rotation -------------------------------------------------------------

    def _apply_rotation_and_render(self):
        if self.intermediate_acescg is None:
            return None
        if self.rotation == 90:
            self.intermediate_acescg = np.ascontiguousarray(
                np.rot90(self.intermediate_acescg, k=-1))
        elif self.rotation == 180:
            self.intermediate_acescg = np.ascontiguousarray(
                np.rot90(self.intermediate_acescg, k=2))
        elif self.rotation == 270:
            self.intermediate_acescg = np.ascontiguousarray(
                np.rot90(self.intermediate_acescg, k=1))
        self.rotation = 0
        return self.render_preview()


# =============================================================================
# EXPORT
# =============================================================================

def export_image(processor, output_path, quality=95, as_tiff=False,
                 lut_profiling=False):
    """Export JPEG or LUT-profiling TIFF from the current processor state.

    Standard export produces a JPEG. TIFF export (lut_profiling=True, via the
    advanced panel) encodes the ACEScg intermediate as ACEScct with full
    reverse-AE + boost applied, suitable for LUT training in DaVinci Resolve.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    ext     = os.path.splitext(output_path)[1].lower()
    is_tiff = as_tiff or ext in ('.tif', '.tiff')

    if is_tiff:
        img = processor.intermediate_acescg
        if img is None:
            return False
        if lut_profiling:
            rev_gain = processor._rev_gain if DebugConfig.enable_reverse_autoexposure else 1.0
            if not np.isclose(rev_gain, 1.0):
                img = img * rev_gain
            if DebugConfig.enable_post_ae_exposure_boost:
                img = img * (2.0 ** DebugConfig.post_ae_exposure_boost_ev)
        if DebugConfig.enable_cnr and DebugConfig.cnr_sigma > 0:
            img = reduce_color_noise_chroma(img, sigma=DebugConfig.cnr_sigma)
        img_acescct = acescct_encode(np.maximum(img, 1e-10))
        img16 = np.clip(img_acescct * 65535.0, 0, 65535).astype(np.uint16)
        bgr   = cv2.cvtColor(img16, cv2.COLOR_RGB2BGR)
        return bool(cv2.imwrite(output_path, bgr))

    img = processor.render_export()
    if img is None:
        return False
    img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(img8, mode='RGB').save(output_path, 'JPEG',
                                               quality=quality, optimize=True)
        return True
    except Exception:
        bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
        return bool(cv2.imwrite(output_path, bgr,
                                [cv2.IMWRITE_JPEG_QUALITY, quality]))
