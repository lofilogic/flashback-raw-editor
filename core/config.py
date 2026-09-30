"""
Constants, presets and the two state dataclasses.

VibeConfig        effect parameters that make up a vibe. Persisted by
                  core.vibe_state, edited in the F12 panel.
ImageAdjustments  per-image sliders + rotation. Saved in projects.
"""
from dataclasses import dataclass, asdict, fields, replace
import math as _math
import os as _os

# =============================================================================
# RAW PIPELINE CONSTANTS
# =============================================================================

SENSOR_BLACK = 64

# ONE35 V2 sensor geometry. The DNG exporter copies the raw strip verbatim,
# so these must match the source. The byte count is the fallback when
# StripByteCounts is missing (10-bit packed: w*h*10/8).
SENSOR_WIDTH = 4144
SENSOR_HEIGHT = 3088
SENSOR_RAW_STRIP_BYTES = 15995840

# WB slider zero. D55, the ForwardMatrix1 calibration illuminant; the
# generic-raw path targets it too so both land on the same neutral.
BASE_KELVIN = 5500.0

# D65, the reference for libraw's daylight_whitebalance.
GENERIC_DAYLIGHT_K = 6504.0

# Fallback Bayer WB for cameras whose raw file lacks daylight_whitebalance.
GENERIC_DAYLIGHT_WB_FALLBACK = [2.0, 1.0, 1.6, 1.0]

# Profile tone curve, (in, out) pairs. Written to exported DNGs (tag 50940)
# and used for rendering when no LUT is active.
PROFILE_TONE_CURVE = [
    0.0, 0.0, 0.02, 0.02, 0.06, 0.10, 0.20, 0.42,
    0.40, 0.70, 0.78, 0.95, 1.0, 1.0,
]

# =============================================================================
# EXPOSURE PIPELINE TUNING
# =============================================================================

# Render-time lift (EV) added to the user exposure. Not counteracted after
# the LUT, so it raises output brightness. Bridges the gap between the level
# the LUTs were trained on and the camera-metered intermediate.
BASE_EXPOSURE_OFFSET_V2 = 2.0

# Generic raws only: lift (EV) baked in at develop. libraw's linear develop
# maps raw white to 1.0, which puts mid-grey ~2 stops below the Flashback
# develop that BASE_EXPOSURE_OFFSET_V2 was tuned against.
GENERIC_RAW_ANCHOR_EV = 2.0

# Linear boost after reverse-AE, before the ACEScct encode. Must match
# tools/build_color_charts.py or the LUT sees a different input range than it
# was trained on.
POST_AE_EXPOSURE_BOOST_EV = 2.0

# How much of reverse-AE + boost to apply. 0 = camera-metered, 1 = full
# effect through the LUT. 0.3 adds some film character without drifting far
# from the metered brightness.
REVERSE_AE_STRENGTH = 0.3

# Push/pull range in EV each way. Shifts exposure into the LUT and undoes it
# afterwards, so brightness stays put and only the toe/shoulder changes. Also
# drives grain highlight bias.
PUSH_PULL_RANGE_EV = 2.0

# =============================================================================
# EFFECT DEFAULTS
# =============================================================================

# Defaults in user-facing units (see VibeConfig); the helpers below convert
# them for the effect functions.
CA_PIXELS = 5.0            # blue offset in px at the long edge
HALATION_THRESHOLD_STOPS = 4.5   # EV above middle grey
HALATION_BLUR_RADIUS = 8.0 # px
HALATION_STRENGTH_PCT = 75.0
# Halo colour. 100% = the red-orange of colour-negative back-reflection,
# 0% = colourless, >100% heads toward a CineStill halo. Only the depth of the
# red changes, not the hue (see halation_scale_tint).
HALATION_WARMTH_PCT = 120.0
SOFTNESS_SIGMA = 0.5       # px
# Corner softness: radial defocus toward the corners (field curvature).
# Separate from the global softness blur.
EDGE_SOFTNESS_STRENGTH_PCT = 60.0   # 0–100 → max sharp→blur blend at the corners
EDGE_SOFTNESS_SIGMA = 3.0           # px, blur radius of the soft copy
EDGE_SOFTNESS_START_PCT = 40.0      # 0–100 → radius (as % of corner) where softness begins
GRAIN_STRENGTH_PCT = 50.0
GRAIN_TILE_SCALE = 0.8     # <1.0 makes grain finer (tiles render denser); >1.0 makes it chunkier.
GRAIN_HIGHLIGHT_BIAS = 0.3 # 1.0 = grain biased to highlights, 0.0 = shadows, 0.5 = flat.
SHARPEN_STRENGTH_PCT = 50.0
SHARPEN_RADIUS = 1.0       # px
CNR_AMOUNT_PCT = 20.0      # sigma 4
CNR_DESPIKE_PCT = 60.0        # chroma firefly/outlier clamp; 0 = off
CNR_DESPIKE_BIAS_PCT = 75.0  # 0 = symmetric, 100 = green (-a*) only
VIGNETTE_STRENGTH_PCT = 50.0
VIGNETTE_COLOR_PCT = 25.0
VIGNETTE_CURVE = 0.0       # -100…+100, higher = more feathered (softer)
BLOOM_STRENGTH_PCT = 30.0
BLOOM_THRESHOLD_STOPS = 3.0      # EV above middle grey

# Internal maxima: percent fields map 0–100 onto 0–MAX.
_CNR_SIGMA_MAX = 20.0
# Despike clamp range in Lab a*/b*: loose at low amounts, tight at 100.
_CNR_DESPIKE_T_HI = 40.0
_CNR_DESPIKE_T_LO = 4.0
_VIGNETTE_COLOR_MAX = 0.2


# =============================================================================
# UNIT CONVERSIONS  (user-facing values  →  internal effect scalars)
# =============================================================================

# Long edge of the V2 half-size develop. Pixel-sized effects are tuned at this
# size and scaled by long_edge / WORKING_LONG_EDGE for other inputs.
WORKING_LONG_EDGE = 2072

# Experimental JPEG input. Applied at load in this order: tone curve,
# exposure, contrast, saturation, shadow saturation.
#
# The tone curve undoes the camera's tone mapping. Matched by hand in
# Photoshop: ACEScct export of a JPEG curved onto its RAW twin. 0–255.
JPEG_TONE_CURVE = [(0, 0), (39, 45), (136, 121), (158, 143), (181, 193), (200, 222), (255, 255)]
JPEG_EXPOSURE_EV = 0.0   # stops
JPEG_CONTRAST = 1.0      # power around 18% grey in linear; <1 flattens
JPEG_SATURATION = 0.9    # 1 = unchanged, 0 = monochrome
JPEG_SHADOW_SATURATION = 0.5  # saturation at -5 EV and below; 1 = off
JPEG_SHADOW_SAT_END_EV = 1.0 # stops vs 18% grey where it has faded to none
JPEG_SPATIAL_MULT = 1   # extra effect-size factor on top of the long-edge scale
JPEG_VIGNETTE_MULT = 3.0  # camera JPEGs arrive lens-corrected; RAWs keep theirs


def ca_pixels_to_scale(pixels: float, long_edge: int) -> float:
    """Pixel offset at the long half-edge → radial CA scale.

    Normalised by the long edge so rotating the frame doesn't change the fringe.
    """
    if long_edge <= 0:
        return 0.0
    return float(pixels) / (float(long_edge) / 2.0)


def pct(value: float) -> float:
    """Percent → fraction."""
    return float(value) / 100.0


def vignette_curve_to_power(curve: float) -> float:
    """-100…+100 curve → cosine falloff exponent. 0 is neutral, positive is
    softer."""
    return float(2.0 ** (-float(curve) / 50.0))


def cnr_pct_to_sigma(amount_pct: float) -> float:
    return pct(amount_pct) * _CNR_SIGMA_MAX


def cnr_sigma_color(sigma: float) -> float:
    """Bilateral range sigma for chroma NR.

    Scales with the spatial sigma; with a fixed range sigma, strength plateaued
    because the bilateral kept preserving chroma near edges. The floor of 15
    stops colour bleeding across edges at low settings. Shared by the CPU and
    GPU paths.
    """
    return max(15.0, float(sigma) * 3.0)


def cnr_despike_thresholds(amount_pct: float, bias_pct: float) -> tuple:
    """Lab clamp limits ``(thr_green, thr_other)`` for the despike prepass.

    Each chroma channel is clamped to median ± thr of its 3x3 neighbourhood,
    which removes single-pixel colour spikes. The bilateral can't do that: it
    treats a spike as an edge and keeps it.

    thr_green applies to a* below the median, thr_other to everything else.
    bias widens thr_other, so 100% only touches green. (0, 0) means off.
    """
    amt = pct(amount_pct)
    if amt <= 0.0:
        return (0.0, 0.0)
    thr_green = _CNR_DESPIKE_T_HI - amt * (_CNR_DESPIKE_T_HI - _CNR_DESPIKE_T_LO)
    bias = min(0.999, max(0.0, pct(bias_pct)))
    thr_other = thr_green / (1.0 - bias)
    return (thr_green, thr_other)


def vignette_color_pct_to_shift(color_pct: float) -> float:
    return pct(color_pct) * _VIGNETTE_COLOR_MAX


# 18% middle grey, the reference point for the threshold-in-stops scale.
_MID_GREY_LINEAR = 0.18


def stops_above_mid_grey_to_acescct(stops: float) -> float:
    """Stops above 18% grey → ACEScct value, for the bloom/halation masks.

    Ignores the ACEScct toe, which only matters below about -4.5 stops.
    """
    linear = _MID_GREY_LINEAR * (2.0 ** float(stops))
    return float((_math.log2(max(linear, 1e-10)) + 9.72) / 17.52)

# =============================================================================
# HALATION SCALE MODEL
# =============================================================================
#
# Halation is three scales blurred at growing radii, summed and
# screen-blended. Outer scales are redder, since back-reflected light passes
# the upper dye layers and orange mask twice.
#
# Per scale: (radius_mult, thresh_offset, weight, green_frac, blue_frac, kind)
#   radius_mult    multiple of halation_blur_radius
#   thresh_offset  added to the ACEScct threshold; outer scales only catch
#                  the brightest sources
#   weight         share of the summed glow
#   green/blue     relative to red at warmth 100%
#   kind           'disc' = defocused copy of the highlights (the defined
#                  halo), 'exp' = diffuse scatter tail
HALATION_SCALES = (
    (1.0, 0.0,  1.00, 0.45, 0.12, 'disc'),  # core
    (2.5, 0.10, 0.18, 0.28, 0.05, 'exp'),   # near scatter
    (5.0, 0.20, 0.07, 0.16, 0.02, 'exp'),   # far scatter
)


def halation_scale_tint(green_frac: float, blue_frac: float, weight: float,
                        warmth_pct: float):
    """RGB tint for one scale, weight included. Green and blue are
    frac ** (warmth/100), so 0% is neutral and >100% is redder."""
    exp = max(warmth_pct, 0.0) / 100.0
    return (weight, weight * (green_frac ** exp), weight * (blue_frac ** exp))


# =============================================================================
# DEBUG / TIMING
# =============================================================================

# Per-stage timing prints, enabled with LOFILOGIC_DEBUG_TIMING=1.
DEBUG_TIMING = _os.environ.get('LOFILOGIC_DEBUG_TIMING', '').lower() in ('1', 'true', 'yes')


def _timing_print(msg):
    if DEBUG_TIMING:
        print(msg)


# =============================================================================
# VIBE CONFIG (the "film stock" layer)
# =============================================================================

@dataclass
class VibeConfig:
    """Effect parameters for one vibe. Usually built by vibe_config_for()."""
    # ---- effect toggles ----
    enable_halation: bool = True
    enable_chromatic_aberration: bool = True
    enable_softness: bool = True
    enable_edge_softness: bool = False
    enable_grain: bool = True
    enable_sharpen: bool = True
    enable_cnr: bool = True
    enable_lut: bool = True
    enable_vignette: bool = True
    enable_bloom: bool = True

    # ---- effect parameters (user-facing units) ----
    # Thresholds are in stops above 18% grey.
    halation_threshold_stops: float = HALATION_THRESHOLD_STOPS  # EV above mid grey
    halation_blur_radius: float = HALATION_BLUR_RADIUS         # px
    halation_strength_pct: float = HALATION_STRENGTH_PCT       # 0–300
    halation_warmth_pct: float = HALATION_WARMTH_PCT           # 0–300, 100 = physical
    ca_pixels: float = CA_PIXELS                                # edge px @ long edge
    softness_sigma: float = SOFTNESS_SIGMA                      # px
    edge_softness_strength_pct: float = EDGE_SOFTNESS_STRENGTH_PCT  # 0–100
    edge_softness_sigma: float = EDGE_SOFTNESS_SIGMA            # px
    edge_softness_start_pct: float = EDGE_SOFTNESS_START_PCT    # 0–100 (% of corner radius)
    grain_strength_pct: float = GRAIN_STRENGTH_PCT              # 0–200
    sharpen_strength_pct: float = SHARPEN_STRENGTH_PCT          # 0–500
    sharpen_radius: float = SHARPEN_RADIUS                      # px
    cnr_amount_pct: float = CNR_AMOUNT_PCT                      # 0–100
    cnr_despike_pct: float = CNR_DESPIKE_PCT                    # 0–100 (chroma firefly clamp)
    cnr_despike_bias_pct: float = CNR_DESPIKE_BIAS_PCT          # 0 sym … 100 green-only
    vignette_strength_pct: float = VIGNETTE_STRENGTH_PCT        # 0–100
    vignette_color_pct: float = VIGNETTE_COLOR_PCT              # 0–100
    vignette_curve: float = VIGNETTE_CURVE                      # -100…+100
    bloom_strength_pct: float = BLOOM_STRENGTH_PCT              # 0–100
    bloom_threshold_stops: float = BLOOM_THRESHOLD_STOPS        # EV above mid grey

    # ---- reverse-AE (advanced) ----
    enable_reverse_autoexposure: bool = False
    reverse_autoexposure_t_ref: float = 1e-3
    enable_post_ae_exposure_boost: bool = False
    post_ae_exposure_boost_ev: float = POST_AE_EXPOSURE_BOOST_EV
    reverse_ae_strength: float = REVERSE_AE_STRENGTH

    # ---- pipeline tuning ----
    base_exposure_offset_v2: float = BASE_EXPOSURE_OFFSET_V2

    # ---- LUT + DNG metadata ----
    # "" (no LUT), "factory:<id>" (see FACTORY_LUTS) or "user:<abs path>".
    # Factory ids keep saved vibes working when the install moves.
    lut_ref: str = ''
    dng_profile_name: str = 'Flashback Standard'

    # Pre-1.5 custom LUT path, kept by the migrator so the user can find and
    # re-import it. Display only; cleared on the next LUT import.
    legacy_user_lut: str = ''

    # ---- serialization ----
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'VibeConfig':
        """Build a VibeConfig from a dict; unknown keys ignored, types coerced."""
        kwargs = {}
        known = {f.name: f.type for f in fields(cls)}
        for name, t in known.items():
            if name in d:
                try:
                    kwargs[name] = t(d[name]) if t is not bool else bool(d[name])
                except (TypeError, ValueError):
                    pass  # leave default
        return cls(**kwargs)

    def copy(self) -> 'VibeConfig':
        return replace(self)


# =============================================================================
# IMAGE ADJUSTMENTS (the per-image layer)
# =============================================================================

@dataclass
class ImageAdjustments:
    """The four main-window sliders + rotation for one image.

    active_vibe_id is stored per image although the UI has one global vibe,
    so per-image vibes wouldn't need a schema change.
    """
    exposure_ev: float = 0.0
    wb_temp: float = 0.0
    tint: float = 0.0
    push_pull_ev: float = 0.0
    rotation: int = 0
    active_vibe_id: str = ''   # filled in by the editor when an image loads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ImageAdjustments':
        kwargs = {}
        known = {f.name: f.type for f in fields(cls)}
        for name, t in known.items():
            if name in d:
                try:
                    kwargs[name] = t(d[name])
                except (TypeError, ValueError):
                    pass
        return cls(**kwargs)

    def copy(self) -> 'ImageAdjustments':
        return replace(self)


# =============================================================================
# LUTS
# =============================================================================

# Paths relative to the install root, resolved through resource_path.

FACTORY_LUTS = {
    'disposable':           'assets/luts/disposable.cube',
    'disposable_v1':        'assets/luts/disposable_V1.cube',
    'flashback_classic_v1': 'assets/luts/V1.cube',
    'point_shoot':          'assets/luts/pointandshoot.cube',
    'rangefinder':          'assets/luts/rangefinder.cube',
    'monochrome':           'assets/luts/monochrome.cube',
}

# Prefixes for VibeConfig.lut_ref.
LUT_REF_FACTORY = 'factory:'
LUT_REF_USER = 'user:'

# V1 negatives are flatter than V2 DNGs, so some looks have a V1 variant.
# Swapped in at render time only; never saved, never applied to user LUTs.
_V1_LUT_OVERRIDES = {
    LUT_REF_FACTORY + 'disposable': LUT_REF_FACTORY + 'disposable_v1',
}


def effective_lut_ref(base_ref: str, is_v1: bool) -> str:
    """The LUT ref to render with: the V1 variant for V1 negatives if one
    exists, otherwise base_ref."""
    if is_v1:
        return _V1_LUT_OVERRIDES.get(base_ref, base_ref)
    return base_ref


def resolve_lut_ref(ref: str):
    """Tagged LUT ref → (absolute_path, origin), origin being 'factory',
    'user' or None. The path is None if the file is missing."""
    # Local import: core/__init__.py imports this module.
    from . import resource_path
    if not ref:
        return None, None
    if ref.startswith(LUT_REF_FACTORY):
        fid = ref[len(LUT_REF_FACTORY):]
        rel = FACTORY_LUTS.get(fid)
        if not rel:
            return None, 'factory'
        abs_path = resource_path(rel)
        return (abs_path if _os.path.exists(abs_path) else None), 'factory'
    if ref.startswith(LUT_REF_USER):
        path = ref[len(LUT_REF_USER):]
        return (path if _os.path.exists(path) else None), 'user'
    return None, None


# =============================================================================
# VIBE PRESETS
# =============================================================================

# User-facing units. ca_pixels is at the 2072 px working size.
VIBE_PRESETS = {
    'disposable':           {'enable_ca': True,  'ca_pixels': 8.0, 'softness': 0.5, 'sharpness_pct': 200.0, 'sharpen_radius': 0.5, 'grain_pct': 120.0, 'vignette_pct': 10.0, 'vignette_curve':  66.0, 'bloom_pct': 15.0, 'lut': 'factory:disposable'},
    'flashback_classic_v1': {'enable_ca': True,  'ca_pixels':  5.0, 'softness': 0.3, 'sharpness_pct':  80.0, 'sharpen_radius': 0.5, 'grain_pct': 200.0, 'vignette_pct': 10.0, 'vignette_curve':  66.0, 'bloom_pct':  3.0, 'lut': 'factory:flashback_classic_v1', 'base_exposure_offset_v2': 0.0},
    'point_shoot':          {'enable_ca': True,  'ca_pixels':  2.0, 'softness': 0.3, 'sharpness_pct':  50.0, 'sharpen_radius': 1.0, 'grain_pct':  80.0, 'vignette_pct': 10.0, 'vignette_curve':   0.0, 'bloom_pct': 10.0, 'lut': 'factory:point_shoot'},
    'rangefinder':          {'enable_ca': False, 'ca_pixels':  0.0, 'softness': 0.1, 'sharpness_pct':  80.0, 'sharpen_radius': 1.0, 'grain_pct':  50.0, 'vignette_pct':  5.0, 'vignette_curve':   0.0, 'bloom_pct':  5.0, 'lut': 'factory:rangefinder'},
    'monochrome':           {'enable_ca': False, 'ca_pixels':  0.0, 'softness': 0.1, 'sharpness_pct':  80.0, 'sharpen_radius': 1.0, 'grain_pct': 150.0, 'vignette_pct': 20.0, 'vignette_curve':   0.0, 'bloom_pct':  5.0, 'lut': 'factory:monochrome'},
}

# Export filename suffix: {basename}_{suffix}.jpg. Unknown ids use 'edit'.
VIBE_EXPORT_SUFFIX = {
    'disposable':           'disp',
    'point_shoot':          'ps',
    'rangefinder':          'rf',
    'monochrome':           'mono',
    'flashback_classic_v1': 'v1',
}


def vibe_config_for(vibe_id: str) -> VibeConfig:
    """Fresh VibeConfig from a preset. Unlisted fields keep their defaults."""
    cfg = VibeConfig()
    preset = VIBE_PRESETS[vibe_id]
    cfg.enable_chromatic_aberration = preset['enable_ca']
    cfg.ca_pixels                   = preset['ca_pixels']
    cfg.softness_sigma              = preset['softness']
    cfg.sharpen_strength_pct        = preset['sharpness_pct']
    cfg.sharpen_radius              = preset['sharpen_radius']
    cfg.grain_strength_pct          = preset['grain_pct']
    cfg.vignette_strength_pct       = preset['vignette_pct']
    cfg.vignette_curve              = preset.get('vignette_curve', VIGNETTE_CURVE)
    cfg.bloom_strength_pct          = preset['bloom_pct']
    cfg.lut_ref                     = preset['lut']
    cfg.base_exposure_offset_v2     = preset.get('base_exposure_offset_v2', BASE_EXPOSURE_OFFSET_V2)
    return cfg


# Used by the debug panel to detect changes from factory.
VIBE_FIELD_NAMES = tuple(f.name for f in fields(VibeConfig))
