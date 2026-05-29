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

Pipeline (generic raw — non-Flashback):
    rawpy.postprocess(user_wb=daylight_whitebalance, output_color=sRGB,
                      gamma=(1,1), half_size=True, output_bps=16)
      -> linear sRGB (libraw applies camera matrix + daylight WB)
    linear sRGB -> ACEScg via LINSRGB_TO_ACESCG    <- cached intermediate
    same render pipeline from here (LUT, sliders, grain, etc.)
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
from .auto_exposure_reverse import compute_reverse_gain
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

# Fused: wb-normalised camera RGB (= raw / ASN) -> ACEScg.
# Used in the highlight-recovery path to replace the two-step
# rgb_wb -> XYZ -> ACEScg chain with a single matmul.
FM1_WB_TO_ACESCG = (XYZ_D50_TO_ACESCG @ FM1).astype(np.float32)

# ACEScg -> linear sRGB.
ACESCG_TO_LINSRGB = np.array([
    [ 1.70505, -0.62179, -0.08326],
    [-0.13026,  1.14080, -0.01055],
    [-0.02400, -0.12897,  1.15297],
], dtype=np.float32)

# linear sRGB -> ACEScg (for generic raw files developed via rawpy sRGB output).
LINSRGB_TO_ACESCG = np.linalg.inv(ACESCG_TO_LINSRGB).astype(np.float32)

# XYZ -> linear sRGB (IEC 61966-2-1 / D65 primaries).
_XYZ_TO_LINSRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float32)

# Generic raw WB target — matches Flashback's BASE_KELVIN so both paths share
# the same neutral point before the WB slider.
_GENERIC_RAW_TARGET_K = 5500.0
_GENERIC_DAYLIGHT_K   = 6504.0  # CIE D65 standard

# Approximate D65 Bayer WB for cameras whose raw file lacks daylight_whitebalance.
_GENERIC_DAYLIGHT_WB_FALLBACK = [2.0, 1.0, 1.6, 1.0]

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


def _wb_shift_to_kelvin(daylight_wb: list, target_k: float,
                         daylight_k: float = _GENERIC_DAYLIGHT_K) -> list:
    """Shift Bayer WB multipliers from daylight_k to target_k.

    Uses Planckian XYZ ratios in linear sRGB space as a sensor-agnostic proxy.
    The camera's own daylight_whitebalance stays the ground truth; only the CCT
    delta is applied on top.
    """
    rgb_dl = np.clip(_XYZ_TO_LINSRGB @ _planckian_xyz(daylight_k), 1e-6, None)
    rgb_tg = np.clip(_XYZ_TO_LINSRGB @ _planckian_xyz(target_k),   1e-6, None)
    scale  = rgb_dl / rgb_tg
    scale /= scale[1]                   # G is the Bayer reference channel
    wb     = list(daylight_wb)
    wb[0]  = float(wb[0] * scale[0])   # R
    wb[2]  = float(wb[2] * scale[2])   # B
    if len(wb) > 3:
        wb[3] = wb[1]                  # G2 tracks G
    return wb


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

def _read_dng_exif(path: str) -> tuple:
    """Single exifread pass — returns (is_flashback, exposure_seconds).

    Replaces the previous pattern of calling _is_flashback_dng and
    extract_exposure_seconds separately (2-3 file opens → 1).
    """
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        make = str(tags.get('Image Make', '')).strip().lower()
        is_flashback = (make == 'flashback')
        exp_s = None
        tag = tags.get('Image ExposureTime')
        if tag is not None:
            from fractions import Fraction
            val = tag.values[0]
            exp_s = float(Fraction(val.num, val.den))
        return is_flashback, exp_s
    except Exception:
        return False, None


# =============================================================================
# PROCESSOR
# =============================================================================

class FlashbackProcessor:
    """Raw processor for Flashback DNGs and generic camera raws."""

    def __init__(self, lut_path=None):
        self.intermediate_acescg = None
        self.current_file = None
        self.is_flashback_file = False
        self._rev_gain = 1.0
        self._rev_gain_unconditional = 1.0
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

    # ---- generic raw pipeline ------------------------------------------------

    def _develop_generic_raw(self, path: str) -> np.ndarray:
        """Develop a non-Flashback raw file to ACEScg using libraw's camera matrix.

        Uses rawpy's built-in camera profile (from DNG metadata or libraw database)
        with fixed daylight WB, producing linear sRGB which is then converted to
        ACEScg. All subsequent processing (LUT, sliders, grain, etc.) is identical
        to the Flashback path.
        """
        t0 = time.time()
        with rawpy.imread(path) as raw:
            daylight_wb = list(raw.daylight_whitebalance or [])
            if not daylight_wb or all(v == 0.0 for v in daylight_wb):
                daylight_wb = list(_GENERIC_DAYLIGHT_WB_FALLBACK)
                _timing_print(f"  [generic] daylight_whitebalance missing — using D65 fallback")
            fixed_wb = _wb_shift_to_kelvin(daylight_wb, _GENERIC_RAW_TARGET_K)
            _timing_print(f"  [generic] WB shifted D65->{_GENERIC_RAW_TARGET_K:.0f}K: "
                          f"[{fixed_wb[0]:.4f}, {fixed_wb[1]:.4f}, {fixed_wb[2]:.4f}]")

            rgb = raw.postprocess(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                user_wb=fixed_wb,
                use_camera_wb=False,
                use_auto_wb=False,
                half_size=True,
                no_auto_bright=True,
                bright=self.rawpy_bright,
                highlight_mode=1,
                gamma=(1, 1),
                output_bps=16,
                output_color=rawpy.ColorSpace.sRGB,
            ).astype(np.float32) / 65535.0
        _timing_print(f"  raw_develop (generic): {(time.time()-t0)*1000:6.2f} ms  "
                      f"shape={rgb.shape}  range=[{rgb.min():.4f},{rgb.max():.4f}]")

        t0 = time.time()
        acescg = (rgb.reshape(-1, 3) @ LINSRGB_TO_ACESCG.T).reshape(rgb.shape)
        _timing_print(f"  linSRGB->ACEScg: {(time.time()-t0)*1000:6.2f} ms  "
                      f"range=[{acescg.min():.4f},{acescg.max():.4f}]")
        return acescg

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

    def _render_fast(self, downscale=False):
        return self._render(downscale=downscale)

    # ---- pipeline -------------------------------------------------------------

    def _bake_halation(self, acescg):
        if not (DebugConfig.enable_halation and DebugConfig.halation_strength > 0):
            return acescg
        return apply_halation(
            acescg,
            DebugConfig.halation_threshold,
            DebugConfig.halation_blur_radius,
            DebugConfig.halation_strength,
        )

    def load_image(self, dng_path):
        total_start = time.time()
        _timing_print(f"\n{'='*60}")
        _timing_print(f"[processor] Loading: {os.path.basename(dng_path)}")
        _timing_print(f"{'='*60}")

        self.current_file = dng_path

        if os.path.splitext(dng_path)[1].lower() in ('.tif', '.tiff'):
            print("[processor] TIFF import is not supported. Open the original DNG instead.")
            return None

        is_flashback, exp_s = _read_dng_exif(dng_path)
        self.is_flashback_file = is_flashback

        try:
            if is_flashback:
                t0 = time.time()
                with rawpy.imread(dng_path) as raw:
                    # half_size=True performs 2x2 binning and skips demosaicing entirely,
                    # so demosaic_algorithm has no effect — always pass LINEAR as a no-op.
                    rgb = raw.postprocess(
                        demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
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
                    acescg = (rgb_wb.reshape(-1, 3) @ FM1_WB_TO_ACESCG.T).reshape(rgb_wb.shape)
                else:
                    acescg = (rgb.reshape(-1, 3) @ RAW_TO_ACESCG.T).reshape(rgb.shape)
                _timing_print(f"  raw->ACEScg: {(time.time()-t0)*1000:6.2f} ms  "
                              f"range=[{acescg.min():.4f},{acescg.max():.4f}]")

                # exp_s already read from the single EXIF pass above
                self._rev_gain = (float(compute_reverse_gain(exp_s, DebugConfig.reverse_autoexposure_t_ref))
                                  if (exp_s and DebugConfig.enable_reverse_autoexposure) else 1.0)
                self._rev_gain_unconditional = float(compute_reverse_gain(exp_s, DebugConfig.reverse_autoexposure_t_ref)) if exp_s else 1.0

                acescg = self._bake_halation(acescg)
            else:
                acescg = self._develop_generic_raw(dng_path)
                self._rev_gain = 1.0
                self._rev_gain_unconditional = 1.0

            self.intermediate_acescg = np.ascontiguousarray(acescg, dtype=np.float32)

            # Always return a fast downscaled preview so the UI is responsive
            # immediately. The caller is responsible for queuing a full-quality
            # background render via RenderWorker.
            result = self.render_preview(downscale=True)
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
            rev_gain = processor._rev_gain_unconditional
            if not np.isclose(rev_gain, 1.0):
                img = img * rev_gain
            base_ev = DebugConfig.base_exposure_offset_v2
            if not np.isclose(base_ev, 0.0):
                img = img * (2.0 ** base_ev)
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
