"""
Application-wide constants, dataclasses, and runtime configuration.

Two dataclasses model the two layers of user-mutable state:

  VibeConfig         — the "film stock" layer. Effect parameters that
                       define a vibe (halation, grain, LUT, etc.).
                       One instance per active vibe. Persisted via
                       core.vibe_state. Edited only in the debug panel.

  ImageAdjustments   — the per-image layer. Exposure, WB, tint,
                       push/pull, rotation, plus the id of the vibe
                       this image was last edited under. Travels with
                       the image; gets saved in a project.

Everything that used to be DebugConfig.X is now a field on VibeConfig.
"""
from dataclasses import dataclass, asdict, fields, replace
import os as _os

# =============================================================================
# RAW PIPELINE CONSTANTS
# =============================================================================

SENSOR_BLACK = 64

# Native ONE35 V2 sensor geometry. The DNG exporter writes the raw strip
# verbatim, so these must match the source file's ImageWidth/ImageLength.
# SENSOR_RAW_STRIP_BYTES is the fallback strip length used when StripByteCounts
# is missing from the source EXIF (10-bit packed: w*h*10/8).
SENSOR_WIDTH = 4144
SENSOR_HEIGHT = 3088
SENSOR_RAW_STRIP_BYTES = 15995840

# Slider zero for the WB knob. Matches the Flashback ForwardMatrix1's
# calibration illuminant (D55). The generic-raw path also targets this
# Kelvin so both paths land at the same neutral point.
BASE_KELVIN = 5500.0

# CIE D65 — the reference illuminant for libraw's daylight_whitebalance.
GENERIC_DAYLIGHT_K = 6504.0

# Fallback Bayer WB for cameras whose raw file lacks daylight_whitebalance.
GENERIC_DAYLIGHT_WB_FALLBACK = [2.0, 1.0, 1.6, 1.0]

# v2 profile tone curve, used by the DNG exporter (tag 50940) AND by the
# fallback render path when no LUT is active. Pairs of (input, output).
PROFILE_TONE_CURVE = [
    0.0, 0.0, 0.02, 0.02, 0.06, 0.10, 0.20, 0.42,
    0.40, 0.70, 0.78, 0.95, 1.0, 1.0,
]

# =============================================================================
# EXPOSURE PIPELINE TUNING
# =============================================================================

# v2 pipeline: constant render-time exposure lift (EV). Applied alongside
# user exposure_ev and NOT counteracted post-LUT, so it genuinely raises
# output brightness. Tune to compensate for the gap between the LUT's
# training input level and the clean camera-metered intermediate.
BASE_EXPOSURE_OFFSET_V2 = 2.0

# Static linear-space boost applied AFTER reverse-AE and BEFORE ACEScct encode.
# Must match the value used by tools/build_color_charts.py when sampling the
# digital chart, otherwise the LUT's input domain at runtime won't match what
# colormatch saw at training time.
POST_AE_EXPOSURE_BOOST_EV = 2.0

# Fraction of the full reverse-AE + boost effect applied at slider zero.
# 0.0 = camera-metered look (AE fully preserved), 1.0 = old behavior (full
# reverse-AE + boost visible through the LUT). ~0.3 gives a mild film character
# while keeping brightness close to the camera-metered original.
REVERSE_AE_STRENGTH = 0.3

# "Push / Pull" slider extent, in EV (each direction). Pulling
# left scales the pre-LUT exposure down by 2^pp and counteracts it post-LUT
# (brightness ~unchanged, film toe more pronounced); pushing right does the
# opposite. Also drives grain highlight-bias.
PUSH_PULL_RANGE_EV = 2.0

# =============================================================================
# EFFECT DEFAULTS
# =============================================================================

CHROMATIC_ABERRATION_STRENGTH = 0.005
CHROMATIC_ABERRATION_STEPS = 4
CHROMATIC_ABERRATION_BLUE_BLUR = 0.3
CHROMATIC_ABERRATION_ZOOM_BLUR = 1.0  # multiplier on the global zoom-blur pass inside CA
HALATION_THRESHOLD = 0.65
HALATION_BLUR_RADIUS = 4.0
HALATION_STRENGTH = 0.5
SOFTNESS_SIGMA = 0.5
GRAIN_STRENGTH = 0.01
GRAIN_TILE_SCALE = 0.8     # <1.0 makes grain finer (tiles render denser); >1.0 makes it chunkier.
GRAIN_HIGHLIGHT_BIAS = 0.3 # 1.0 = grain biased to highlights, 0.0 = shadows, 0.5 = flat.
SHARPEN_STRENGTH = 0.5
SHARPEN_RADIUS = 1.0
CNR_SIGMA = 2.0
VIGNETTE_STRENGTH = 0.5
VIGNETTE_COLOR_SHIFT = 0.05
VIGNETTE_FEATHER = 1.0
BLOOM_STRENGTH = 0.3
BLOOM_THRESHOLD = 0.65

# =============================================================================
# DEBUG / TIMING
# =============================================================================

# Per-effect timing prints. Off by default; opt in via the FLASHBACK_DEBUG_TIMING
# env var ("1" / "true" / "yes") so user installs stay quiet.
DEBUG_TIMING = _os.environ.get('FLASHBACK_DEBUG_TIMING', '').lower() in ('1', 'true', 'yes')


def _timing_print(msg):
    """Print timing/debug messages. Controlled by DEBUG_TIMING flag."""
    if DEBUG_TIMING:
        print(msg)


# =============================================================================
# VIBE CONFIG (the "film stock" layer)
# =============================================================================

@dataclass
class VibeConfig:
    """All effect parameters that define a vibe.

    One instance per active vibe. Persisted via core.vibe_state.
    Constructed empty (all factory defaults) and then either tweaked by
    the user or seeded from a VIBE_PRESETS recipe via vibe_config_for().
    """
    # ---- effect toggles ----
    enable_halation: bool = True
    enable_chromatic_aberration: bool = True
    enable_softness: bool = True
    enable_grain: bool = True
    enable_sharpen: bool = True
    enable_cnr: bool = True
    enable_lut: bool = True
    enable_vignette: bool = True
    enable_bloom: bool = True

    # ---- effect parameters ----
    halation_threshold: float = HALATION_THRESHOLD
    halation_blur_radius: float = HALATION_BLUR_RADIUS
    halation_strength: float = HALATION_STRENGTH
    ca_strength: float = CHROMATIC_ABERRATION_STRENGTH
    ca_steps: int = CHROMATIC_ABERRATION_STEPS
    ca_blue_blur: float = CHROMATIC_ABERRATION_BLUE_BLUR
    ca_zoom_blur: float = CHROMATIC_ABERRATION_ZOOM_BLUR
    softness_sigma: float = SOFTNESS_SIGMA
    grain_strength: float = GRAIN_STRENGTH
    sharpen_strength: float = SHARPEN_STRENGTH
    sharpen_radius: float = SHARPEN_RADIUS
    cnr_sigma: float = CNR_SIGMA
    vignette_strength: float = VIGNETTE_STRENGTH
    vignette_color_shift: float = VIGNETTE_COLOR_SHIFT
    vignette_feather: float = VIGNETTE_FEATHER
    bloom_strength: float = BLOOM_STRENGTH
    bloom_threshold: float = BLOOM_THRESHOLD

    # ---- reverse-AE (advanced) ----
    enable_reverse_autoexposure: bool = False
    reverse_autoexposure_t_ref: float = 1e-3
    enable_post_ae_exposure_boost: bool = False
    post_ae_exposure_boost_ev: float = POST_AE_EXPOSURE_BOOST_EV
    reverse_ae_strength: float = REVERSE_AE_STRENGTH

    # ---- pipeline tuning ----
    base_exposure_offset_v2: float = BASE_EXPOSURE_OFFSET_V2

    # ---- LUT + DNG metadata ----
    lut_path: str = ''
    dng_profile_name: str = 'Flashback Standard'

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
    """Per-image user adjustments: the four main-window sliders + rotation.

    Travels with the image and is persisted in projects. active_vibe_id
    records which vibe the image was last edited under; for now the UI
    keeps a single global active vibe, but every image stores its own id
    so future per-image vibes (or project reloading) work without a
    schema change.
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
# VIBE PRESETS — recipes that seed a VibeConfig
# =============================================================================

VIBE_PRESETS = {
    'disposable':           {'enable_ca': True,  'ca_strength': 0.010, 'ca_zoom_blur': 1.5, 'softness': 0.3, 'sharpness': 2.0, 'sharpen_radius': 0.5, 'grain': 1.2, 'vignette': 0.10, 'vignette_feather': 0.4, 'bloom': 0.10, 'lut': 'assets/luts/disposable.cube'},
    'flashback_classic_v1': {'enable_ca': True,  'ca_strength': 0.005, 'ca_zoom_blur': 4.0, 'softness': 0.3, 'sharpness': 0.8, 'sharpen_radius': 0.5, 'grain': 2.0, 'vignette': 0.10, 'vignette_feather': 0.4, 'bloom': 0.03, 'lut': 'assets/luts/V1.cube', 'base_exposure_offset_v2': 0.0},
    'point_shoot':          {'enable_ca': True,  'ca_strength': 0.002, 'softness': 0.3, 'sharpness': 0.5, 'sharpen_radius': 1.0, 'grain': 0.8, 'vignette': 0.10, 'vignette_feather': 1.0, 'bloom': 0.10, 'lut': 'assets/luts/pointandshoot.cube'},
    'rangefinder':          {'enable_ca': False, 'ca_strength': 0.0,   'softness': 0.1, 'sharpness': 0.8, 'sharpen_radius': 1.0, 'grain': 0.5, 'vignette': 0.05, 'vignette_feather': 1.0, 'bloom': 0.05, 'lut': 'assets/luts/rangefinder.cube'},
    'monochrome':           {'enable_ca': False, 'ca_strength': 0.0,   'softness': 0.1, 'sharpness': 0.8, 'sharpen_radius': 1.0, 'grain': 1.5, 'vignette': 0.20, 'vignette_feather': 1.0, 'bloom': 0.05, 'lut': 'assets/luts/monochrome.cube'},
}

# Short, file-name-safe suffix per vibe — appended to exported JPGs as
# {basename}_{suffix}.jpg so users can tell at a glance which look produced
# which file. Unknown vibe ids fall back to 'edit'.
VIBE_EXPORT_SUFFIX = {
    'disposable':           'disp',
    'point_shoot':          'ps',
    'rangefinder':          'rf',
    'monochrome':           'mono',
    'flashback_classic_v1': 'v1',
}


def vibe_config_for(vibe_id: str) -> VibeConfig:
    """Construct a fresh VibeConfig from a preset recipe.

    All non-preset fields keep their factory defaults. The preset
    dictionary uses short keys (enable_ca, ca_strength, softness, ...);
    we map those onto the dataclass field names.
    """
    cfg = VibeConfig()  # all factory defaults
    preset = VIBE_PRESETS[vibe_id]
    cfg.enable_chromatic_aberration = preset['enable_ca']
    cfg.ca_strength                 = preset['ca_strength']
    cfg.softness_sigma              = preset['softness']
    cfg.sharpen_strength            = preset['sharpness']
    cfg.sharpen_radius              = preset['sharpen_radius']
    cfg.grain_strength              = preset['grain']
    cfg.vignette_strength           = preset['vignette']
    cfg.vignette_feather            = preset.get('vignette_feather', 1.0)
    cfg.bloom_strength              = preset['bloom']
    cfg.lut_path                    = preset['lut']
    cfg.base_exposure_offset_v2     = preset.get('base_exposure_offset_v2', BASE_EXPOSURE_OFFSET_V2)
    cfg.ca_zoom_blur                = preset.get('ca_zoom_blur', CHROMATIC_ABERRATION_ZOOM_BLUR)
    return cfg


# Names of every VibeConfig field — used by the debug panel to detect
# "modified from factory" state.
VIBE_FIELD_NAMES = tuple(f.name for f in fields(VibeConfig))
