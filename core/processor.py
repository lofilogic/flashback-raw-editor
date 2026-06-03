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
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import rawpy
import cv2
import exifread
import colour

from . import resource_path

log = logging.getLogger(__name__)
from .config import (
    SENSOR_BLACK, GRAIN_TILE_SCALE, GRAIN_HIGHLIGHT_BIAS, PUSH_PULL_RANGE_EV,
    BASE_KELVIN, GENERIC_DAYLIGHT_K, GENERIC_DAYLIGHT_WB_FALLBACK,
    PROFILE_TONE_CURVE,
    VibeConfig, ImageAdjustments,
    pct, ca_pixels_to_scale, vignette_curve_to_power,
    cnr_pct_to_sigma, vignette_color_pct_to_shift,
    stops_above_mid_grey_to_acescct,
    resolve_lut_ref,
    _timing_print,
)
from .kernels import acescct_encode, apply_grain, encode_then_lut, run_resident
from .gpu import gpu
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


@contextmanager
def _timed(label: str):
    """Log wall-time for a single render stage.

    Diagnostic only: the actual printing is gated inside _timing_print by the
    FLASHBACK_DEBUG_TIMING env flag, so this is a no-op (beyond two time reads)
    in normal runs and never touches the rendered output.
    """
    t0 = time.time()
    try:
        yield
    finally:
        _timing_print(f"    [{label}] {(time.time()-t0)*1000:6.2f} ms")


# =============================================================================
# COLOR MATRICES (DNG dual-illuminant)
# =============================================================================

# AsShotNeutral from real D50 grey-patch measurement (matches the asn
# embedded in DNGs by core/dng_export.py and used by Camera Raw at render
# time).
ASN_D50 = np.array([0.541, 1.0, 0.597], dtype=np.float32)

# ForwardMatrix1: camera_wb_rgb (raw / ASN) -> XYZ_D50.
# Calibrated under D50 daylight (matches CalibrationIlluminant1 in the
# DNGs we emit and the --illuminant d50 flag used to derive the matrix).
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

_XYZ_TO_AP1_PURE = XYZ_D60_TO_ACESCG.astype(np.float32)


def _planckian_xyz(cct: float) -> np.ndarray:
    if cct >= 4000.0:
        xy = np.asarray(colour.temperature.CCT_to_xy_CIE_D(cct))
    else:
        xy = np.asarray(colour.temperature.CCT_to_xy_Kang2002(cct))
    return np.asarray(colour.xy_to_XYZ(xy), dtype=np.float32)


def _wb_shift_to_kelvin(daylight_wb: list, target_k: float,
                         daylight_k: float = GENERIC_DAYLIGHT_K) -> list:
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
        # Sony ARW carries G2=0.0 as a sentinel meaning "G2 tracks G1"; any
        # non-zero G2 is interpreted by libraw as an independent multiplier
        # and collapses the WB to near-black. Preserve the sentinel.
        if daylight_wb[3] != 0.0:
            wb[3] = wb[1]
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


_TONE_CURVE_LUT = _build_tone_curve_lut(PROFILE_TONE_CURVE, size=4096)


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


# Per-make exposure boost (EV) for non-Flashback raws developed via libraw.
# Goal: the same exposure settings on each camera produce a similar mid-grey
# in the developed output, with our Fuji pipeline as the rough anchor.
#
# Values are community ballpark — within ~0.5 EV of "matches ACR defaults",
# distilled from RawDigger's Real ISO measurements (Iliah Borg / LibRaw),
# DPReview studio comparisons, and RawTherapee/darktable forum consensus.
# They are NOT calibrated against the in-house Fuji reference and should be
# refined empirically once we measure mid-grey on each body.
_BOOST_EV_BY_MAKE = {
    'sony':                          0.00,
    'nikon':                         0.20,
    'nikon corporation':             0.20,
    'canon':                         0.00,
    'fujifilm':                      1.00,
    'fuji':                          1.00,
    'olympus':                       0.30,
    'olympus corporation':           0.30,
    'olympus imaging corp.':         0.30,
    'om digital solutions':          0.30,
    'panasonic':                     0.20,
    'leica':                         0.30,
    'leica camera ag':               0.30,
    'pentax':                        0.50,
    'ricoh':                         0.50,
    'ricoh imaging company, ltd.':   0.50,
    'sigma':                         0.70,
    'hasselblad':                    0.20,
    'phase one':                     0.20,
    'apple':                         1.50,   # iPhone ProRAW / LR Camera DNG
    'google':                        2.00,   # Pixel HDR+ DNG
    'dji':                           0.30,
}

# Used only when Make can't be read from EXIF — primarily Fuji RAF, which
# is a proprietary container exifread can't parse.
_BOOST_EV_BY_EXT = {
    '.raf': 1.00,
}


def _read_generic_raw_boost_ev(path: str) -> float:
    """Return the per-file exposure boost (EV) for a non-Flashback raw.

    Priority:
      1. EXIF ``Make`` → per-make table.
      2. File extension → per-extension fallback (for raws exifread can't parse).
      3. 0.0 if unknown.
    """
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        make = str(tags.get('Image Make', '')).strip().lower()
        if make and make in _BOOST_EV_BY_MAKE:
            ev = _BOOST_EV_BY_MAKE[make]
            log.info("[processor] baseline boost for make=%r: %+.2f EV", make, ev)
            return ev
        if make:
            log.info("[processor] no baseline-boost entry for make=%r", make)
    except Exception:
        log.exception("[processor] EXIF read failed")
    ext = os.path.splitext(path)[1].lower()
    ev = _BOOST_EV_BY_EXT.get(ext, 0.0)
    log.info("[processor] baseline boost for ext=%r: %+.2f EV", ext, ev)
    return ev


# =============================================================================
# PROCESSOR
# =============================================================================

class FlashbackProcessor:
    """Raw processor for Flashback DNGs and generic camera raws.

    Owns:
      * vibe         — the active VibeConfig (film-stock settings)
      * adjustments  — the per-image ImageAdjustments (sliders + rotation)

    Both are passed by reference; the UI mutates them directly and the
    next render picks up the new values. The processor never reads
    global state.
    """

    def __init__(self, vibe: VibeConfig = None, adjustments: ImageAdjustments = None):
        self.vibe = vibe if vibe is not None else VibeConfig()
        self.adjustments = adjustments if adjustments is not None else ImageAdjustments()

        self.intermediate_acescg = None
        self.current_file = None
        self.is_flashback_file = False
        self._rev_gain = 1.0
        self._rev_gain_unconditional = 1.0
        self.highlight_mode = 1
        self.rawpy_bright = 1.0
        self.enable_highlight_recovery = True

        self.lut = None
        path, origin = resolve_lut_ref(self.vibe.lut_ref)
        if path:
            try:
                self.lut = colour.read_LUT(path)
                log.info("[processor] LUT loaded (%s): %s (%s)", origin, self.lut.name, self.lut.table.shape)
            except Exception as e:
                log.error("[processor] Could not load LUT %s: %s", path, e)
        elif self.vibe.lut_ref:
            # Ref was set but resolved to nothing. The editor handles the
            # user-facing notice; here we just log so the cause is visible.
            log.warning("[processor] LUT ref %r could not be resolved (origin=%s)",
                        self.vibe.lut_ref, origin)

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

    def _apply_grain(self, image, strength, highlight_bias=None, grain_layer=None):
        h, w = image.shape[:2]
        if grain_layer is not None:
            grain = grain_layer
        else:
            with _timed("grain:generate"):
                grain = self._generate_grain_layer(h, w, sigma=strength)
        if highlight_bias is None:
            highlight_bias = GRAIN_HIGHLIGHT_BIAS
        with _timed("grain:blend"):
            out = apply_grain(image, grain, intensity=strength,
                              highlight_bias=highlight_bias)
        return out

    def _resident_post_lut_stages(self, v, shape, grain_driver):
        """Build the post-LUT resident tail (softness -> grain -> sharpen) as a
        list of Frame->Frame stages, matching the per-op order and parameters in
        _render's per-op block.

        Returns (stages, grain_layer). The grain layer is generated here (CPU,
        random tiles) when grain is enabled so it exists regardless of which path
        ultimately runs. Chromatic aberration is intentionally excluded: it is a
        cv2 stage that breaks residency and runs on the CPU before this tail.
        """
        stages = []
        if v.enable_softness and v.softness_sigma > 0:
            sigma = v.softness_sigma
            stages.append(lambda fr, s=sigma: gpu.softness_frame(fr, s))
        grain_layer = None
        if v.enable_grain and v.grain_strength_pct > 0:
            h, w = shape[:2]
            g_strength = pct(v.grain_strength_pct)
            grain_layer = self._generate_grain_layer(h, w, sigma=g_strength)
            g_bias = self._grain_highlight_bias(grain_driver)
            stages.append(lambda fr, g=grain_layer, i=g_strength, b=g_bias:
                          gpu.grain_frame(fr, g, i, highlight_bias=b))
        if v.enable_sharpen and v.sharpen_strength_pct > 0:
            sh_strength = pct(v.sharpen_strength_pct)
            sh_radius = v.sharpen_radius
            stages.append(lambda fr, s=sh_strength, r=sh_radius: gpu.sharpen_frame(fr, s, r))
        return stages, grain_layer

    # ---- generic raw pipeline ------------------------------------------------

    def _develop_generic_raw(self, path: str) -> np.ndarray:
        """Develop a non-Flashback raw file to ACEScg using libraw's camera matrix.

        Uses rawpy's built-in camera profile (DNG metadata or libraw database)
        with use_camera_wb=True so libraw applies pre_mul × cam_mul correctly
        (see body comment); the camera WB is then undone and replaced with a
        fixed BASE_KELVIN WB post-develop, so the downstream pipeline behaves
        as if the file had been shot at the Flashback daylight reference.
        Output is linear sRGB → converted to ACEScg before returning.
        """
        t0 = time.time()
        boost_ev = _read_generic_raw_boost_ev(path)
        boost_gain = float(2.0 ** boost_ev)
        with rawpy.imread(path) as raw:
            # X-Trans uses a 6x6 CFA; libraw's half_size 2x2 binning misaligns
            # the pattern and produces color aliasing. Detect via raw_pattern
            # shape and take a full-size Markesteijn demosaic, then downscale.
            # Some Sony ARWs (compressed/lossless variants) report raw_pattern
            # as None; Sony has no X-Trans, so None means Bayer.
            is_xtrans = raw.raw_pattern is not None and raw.raw_pattern.shape != (2, 2)

            daylight_wb = list(raw.daylight_whitebalance or [])
            if not daylight_wb or all(v == 0.0 for v in daylight_wb):
                daylight_wb = list(GENERIC_DAYLIGHT_WB_FALLBACK)
                _timing_print(f"  [generic] daylight_whitebalance missing — using D65 fallback")
            # Target BASE_KELVIN so the generic path lands at the same neutral
            # point as the Flashback path before the WB slider takes over.
            fixed_wb = _wb_shift_to_kelvin(daylight_wb, BASE_KELVIN)
            _timing_print(f"  [generic] WB shifted D65->{BASE_KELVIN:.0f}K: "
                          f"[{fixed_wb[0]:.4f}, {fixed_wb[1]:.4f}, {fixed_wb[2]:.4f}]")
            if boost_ev != 0.0:
                _timing_print(f"  [generic] baseline exposure boost: {boost_ev:+.2f} EV "
                              f"(gain {boost_gain:.3f})")

            rgb = raw.postprocess(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                user_wb=fixed_wb,
                use_camera_wb=False,
                use_auto_wb=False,
                half_size=not is_xtrans,
                no_auto_bright=True,
                bright=self.rawpy_bright,
                highlight_mode=2,
                gamma=(1, 1),
                output_bps=16,
                output_color=rawpy.ColorSpace.sRGB,
            ).astype(np.float32) / 65535.0
            # Apply baseline boost in float space — libraw's `bright` parameter
            # interacts with its auto-brightness state machine and is unreliable
            # with no_auto_bright=True + linear gamma. Multiplying the float
            # output is a clean, predictable linear gain; values above 1.0 will
            # be reined back in by the highlight rolloff downstream.
            if boost_gain != 1.0:
                rgb *= boost_gain
        _timing_print(f"  raw_develop (generic{', x-trans' if is_xtrans else ''}): "
                      f"{(time.time()-t0)*1000:6.2f} ms  "
                      f"shape={rgb.shape}  range=[{rgb.min():.4f},{rgb.max():.4f}]")

        if is_xtrans:
            t0 = time.time()
            h, w = rgb.shape[:2]
            rgb = cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            _timing_print(f"  x-trans downscale to half: {(time.time()-t0)*1000:6.2f} ms  "
                          f"shape={rgb.shape}")

        t0 = time.time()
        acescg = (rgb.reshape(-1, 3) @ LINSRGB_TO_ACESCG.T).reshape(rgb.shape)
        _timing_print(f"  linSRGB->ACEScg: {(time.time()-t0)*1000:6.2f} ms  "
                      f"range=[{acescg.min():.4f},{acescg.max():.4f}]")
        return acescg

    # ---- public surface -------------------------------------------------------

    def get_settings(self) -> dict:
        """Return a dict copy of the current per-image adjustments.

        Returns a dict (not the ImageAdjustments instance) so the UI can
        merge in extra fields like 'auto_tint' without touching the
        canonical dataclass.
        """
        return self.adjustments.to_dict()

    def set_settings(self, adjustments):
        """Update adjustments. Accepts either an ImageAdjustments instance or
        a partial dict; unknown keys ignored."""
        if isinstance(adjustments, ImageAdjustments):
            self.adjustments = adjustments
        elif isinstance(adjustments, dict):
            for k, v in adjustments.items():
                if hasattr(self.adjustments, k):
                    setattr(self.adjustments, k, v)

    def rotate_clockwise(self):
        self.adjustments.rotation = (self.adjustments.rotation + 90) % 360
        return self._apply_rotation_and_render()

    def rotate_counterclockwise(self):
        self.adjustments.rotation = (self.adjustments.rotation - 90) % 360
        return self._apply_rotation_and_render()

    def get_rotation(self):
        return self.adjustments.rotation

    def _render_fast(self, downscale=False):
        return self._render(downscale=downscale)

    # ---- pipeline -------------------------------------------------------------

    def _bake_halation(self, acescg):
        if not (self.vibe.enable_halation and self.vibe.halation_strength_pct > 0):
            return acescg
        return apply_halation(
            acescg,
            stops_above_mid_grey_to_acescct(self.vibe.halation_threshold_stops),
            self.vibe.halation_blur_radius,
            pct(self.vibe.halation_strength_pct),
        )

    def load_image(self, dng_path):
        total_start = time.time()
        _timing_print(f"\n{'='*60}")
        _timing_print(f"[processor] Loading: {os.path.basename(dng_path)}")
        _timing_print(f"{'='*60}")

        self.current_file = dng_path

        if os.path.splitext(dng_path)[1].lower() in ('.tif', '.tiff'):
            log.warning("[processor] TIFF import is not supported. Open the original DNG instead.")
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
                self._rev_gain = (float(compute_reverse_gain(exp_s, self.vibe.reverse_autoexposure_t_ref))
                                  if (exp_s and self.vibe.enable_reverse_autoexposure) else 1.0)
                self._rev_gain_unconditional = float(compute_reverse_gain(exp_s, self.vibe.reverse_autoexposure_t_ref)) if exp_s else 1.0

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
            # Returning None lets the caller (UI) decide how to surface the
            # failure; the full traceback is logged for diagnostics. The
            # exception is intentionally swallowed because load_image is the
            # user-facing critical path and a half-loaded image is worse than
            # a clean miss — we just need to make sure it can't fail silently.
            log.exception("[processor] load failed for %s: %s", dng_path, e)
            return None

    def render_preview(self, downscale=False):
        if self.intermediate_acescg is None:
            return None
        return self._render(downscale=downscale)

    def render_export(self):
        return self._render(downscale=False)

    def _render(self, downscale=False):
        t0  = time.time()
        v   = self.vibe          # film-stock parameters
        a   = self.adjustments   # per-image sliders
        img = self.intermediate_acescg
        if downscale:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // 3, h // 3), interpolation=cv2.INTER_LINEAR)

        push_pull_ev = float(a.push_pull_ev)

        f        = float(v.reverse_ae_strength)
        rev_gain = self._rev_gain if v.enable_reverse_autoexposure else 1.0
        rev_ev   = float(np.log2(rev_gain)) if rev_gain > 0 else 0.0
        boost_ev = float(v.post_ae_exposure_boost_ev) if v.enable_post_ae_exposure_boost else 0.0
        pre_lut_ev = f * rev_ev + f * boost_ev + push_pull_ev

        wb   = _kelvin_to_acescg_gain(BASE_KELVIN + a.wb_temp)
        tint = _tint_to_acescg_gain(a.tint)
        ev   = float(2.0 ** (a.exposure_ev + v.base_exposure_offset_v2 + pre_lut_ev))
        gain = (wb * tint * ev).astype(np.float32)
        if not np.allclose(gain, 1.0):
            img = img * gain

        if not downscale:
            if v.enable_bloom and v.bloom_strength_pct > 0:
                with _timed("bloom"):
                    img = apply_bloom(img, pct(v.bloom_strength_pct),
                                      stops_above_mid_grey_to_acescct(v.bloom_threshold_stops),
                                      linear=True)
            if v.enable_vignette and v.vignette_strength_pct > 0:
                with _timed("vignette"):
                    img = apply_vignette(img, pct(v.vignette_strength_pct),
                                         vignette_color_pct_to_shift(v.vignette_color_pct),
                                         vignette_curve_to_power(v.vignette_curve))

        grain_driver = f * rev_ev + push_pull_ev
        tail_done = False
        # Resident post-LUT tail (softness -> grain -> sharpen), built once and
        # shared by the fused chain and the CA path below. grain_layer is also
        # reused by the per-op fallback, so one grain layer is generated per
        # render regardless of which path runs.
        post_tail, grain_layer = (([], None) if downscale
                                  else self._resident_post_lut_stages(v, img.shape, grain_driver))

        if v.enable_lut and self.lut is not None:
            if v.enable_cnr and v.cnr_amount_pct > 0:
                with _timed("CNR"):
                    img = reduce_color_noise_chroma(img, sigma=cnr_pct_to_sigma(v.cnr_amount_pct))
            img_max = np.maximum(img, 1e-10)
            ca_on = v.enable_chromatic_aberration and v.ca_pixels > 0

            img_display = None
            # No CA: fuse the entire back half — ACEScct-encode -> LUT -> tail —
            # into one resident chain (single upload, single readback). With CA
            # on, the tail still goes resident, just after the CPU CA stage (see
            # the post-LUT block below). Any GPU miss returns None -> fallback.
            if not downscale and not ca_on:
                with _timed("encode+LUT+tail (resident)"):
                    img_display = run_resident(
                        img_max, [gpu.encode_frame, gpu.lut_frame, *post_tail])
                tail_done = img_display is not None

            if img_display is None:
                # Resident encode->LUT (single upload/readback), falling back to
                # the CPU encode + LUT when no GPU/LUT is available.
                with _timed("encode+LUT (resident)"):
                    img_display = encode_then_lut(img_max)
                if img_display is None:
                    with _timed("ACEScct encode"):
                        img_acescct = acescct_encode(img_max)
                    try:
                        img_display = apply_lut_fast(img_acescct, self.lut)
                    except Exception as e:
                        log.error("[processor] LUT error: %s", e)
                        img_display = np.clip(img_acescct, 0, 1)
        else:
            flat     = img.reshape(-1, 3)
            prophoto = (flat @ ACESCG_TO_PROPHOTO).reshape(img.shape)
            prophoto = _apply_tone_curve(np.clip(prophoto, 0.0, 1.0))
            lin_srgb = (prophoto.reshape(-1, 3) @ PROPHOTO_TO_LINSRGB).reshape(prophoto.shape)
            img_display = _srgb_oetf(np.clip(lin_srgb, 0.0, 1.0))

        # Post-LUT tail. Skipped when the fused chain already ran it (tail_done).
        # Otherwise CA runs on the CPU (cv2 warpAffine) and the softness/grain/
        # sharpen tail runs as one resident sub-chain (single upload/readback),
        # falling back to the per-op path on a GPU miss.
        if not downscale and not tail_done:
            if v.enable_chromatic_aberration and v.ca_pixels > 0:
                ca_scale = ca_pixels_to_scale(v.ca_pixels, img_display.shape[1])
                img_display = apply_chromatic_aberration(
                    img_display, ca_scale, v.ca_steps, v.ca_blue_blur,
                    zoom_blur=pct(v.ca_zoom_blur_pct))
            resident_tail = None
            if post_tail:
                with _timed("tail (resident)"):
                    resident_tail = run_resident(img_display, post_tail)
            if resident_tail is not None:
                img_display = resident_tail
            else:
                if v.enable_softness and v.softness_sigma > 0:
                    with _timed("softness"):
                        img_display = apply_softness(img_display, v.softness_sigma)
                if v.enable_grain and v.grain_strength_pct > 0:
                    img_display = self._apply_grain(
                        img_display, pct(v.grain_strength_pct),
                        highlight_bias=self._grain_highlight_bias(grain_driver),
                        grain_layer=grain_layer)
                if v.enable_sharpen and v.sharpen_strength_pct > 0:
                    with _timed("sharpen"):
                        img_display = apply_sharpen(
                            img_display, pct(v.sharpen_strength_pct), v.sharpen_radius)

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
        rot = self.adjustments.rotation
        if rot == 90:
            self.intermediate_acescg = np.ascontiguousarray(
                np.rot90(self.intermediate_acescg, k=-1))
        elif rot == 180:
            self.intermediate_acescg = np.ascontiguousarray(
                np.rot90(self.intermediate_acescg, k=2))
        elif rot == 270:
            self.intermediate_acescg = np.ascontiguousarray(
                np.rot90(self.intermediate_acescg, k=1))
        self.adjustments.rotation = 0
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
        vibe = processor.vibe
        if lut_profiling:
            rev_gain = processor._rev_gain_unconditional
            if not np.isclose(rev_gain, 1.0):
                img = img * rev_gain
            base_ev = vibe.base_exposure_offset_v2
            if not np.isclose(base_ev, 0.0):
                img = img * (2.0 ** base_ev)
        if vibe.enable_cnr and vibe.cnr_amount_pct > 0:
            img = reduce_color_noise_chroma(img, sigma=cnr_pct_to_sigma(vibe.cnr_amount_pct))
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
