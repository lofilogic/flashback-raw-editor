"""
Application-wide constants, effect defaults, and runtime configuration.
"""
import numpy as np
import colour

# =============================================================================
# COLOR PIPELINE CONSTANTS
# =============================================================================

FLASHBACK_CCM = np.array([
    [ 3.8045148 , -0.40716213,  0.03187762],
    [-0.45492041,  0.73636414,  0.02067507],
    [ 0.11892583, -0.55000283,  2.91999937]
])

FLASHBACK_CCM2 = np.array([
    [0.706563, 0.061847, 0.231589],
    [-0.186467, 1.036193, 0.150275],
    [-0.167809, -0.001273, 1.169082],
])

# Fitted via tools/match_camera.py against a ColorChecker shot, mapping
# iPhone raw (with daylight_whitebalance pre-applied) to Flashback-style
# linear sRGB so the same LUT lands in roughly the same place.
IPHONE_CCM = np.array([
    [ 0.59425110, -0.27924676,  0.02515316],
    [-0.03373678,  0.50158660,  0.05856543],
    [ 0.03639984, -0.34033317,  0.89340804],
])

SENSOR_BLACK = 64
BASE_WB_SETTINGS = [0.5, 1.0, 0.61, 1.0]
BASE_WB_SETTINGS2 = [2.0333, 1.0000, 1.6796, 1.0000]
BASE_EXPOSURE_OFFSET = 1

# Static linear-space boost applied AFTER reverse-AE and BEFORE ACEScct encode.
# Must match the value used by tools/build_color_charts.py when sampling the
# digital chart, otherwise the LUT's input domain at runtime won't match what
# colormatch saw at training time.
POST_AE_EXPOSURE_BOOST_EV = 3.5

# Color spaces (initialized once at import time)
CS_SRGB = colour.RGB_COLOURSPACES['sRGB']
CS_REC2020 = colour.RGB_COLOURSPACES['ITU-R BT.2020']

# Precomputed sRGB → Rec.2020 3×3 matrix (avoids colour-science on every load)
REC2020_FROM_SRGB = colour.RGB_to_RGB(
    np.eye(3, dtype=np.float32), CS_SRGB, CS_REC2020
).astype(np.float32)

# =============================================================================
# EFFECT DEFAULTS
# =============================================================================

CHROMATIC_ABERRATION_STRENGTH = 0.005
CHROMATIC_ABERRATION_STEPS = 4
CHROMATIC_ABERRATION_BLUE_BLUR = 0.3
HALATION_THRESHOLD = 0.55
HALATION_THRESHOLD_FUJI = 0.7
HALATION_BLUR_RADIUS = 4
HALATION_STRENGTH = 0.5
SOFTNESS_SIGMA = 0.5
GRAIN_STRENGTH = 0.01
GRAIN_BLUR_SIGMA = 0.1
GRAIN_TILE_SCALE = 0.8 # <1.0 makes grain finer (tiles render denser); >1.0 makes it chunkier.
GRAIN_HIGHLIGHT_BIAS = 0.3 # 1.0 = grain biased to highlights (matches scanned negative film density), 0.0 = biased to shadows (noise-floor look), 0.5 = flat.
SHARPEN_STRENGTH = 0.5
SHARPEN_RADIUS = 1.0
CNR_SIGMA = 2.0
HIGHLIGHT_DESAT_THRESHOLD_L = 58.0   # Lab L* at which desaturation begins (0-100)
HIGHLIGHT_DESAT_ROLLOFF_L   = 10.0   # width of ramp in L* units
HIGHLIGHT_DESAT_SIGMA       = 10.0    # spatial Gaussian blur on the mask
DITHER_STRENGTH = 0.005
VIGNETTE_STRENGTH = 0.5
VIGNETTE_COLOR_SHIFT = 0.05
VIGNETTE_FEATHER = 1.0
BLOOM_STRENGTH = 0.3
BLOOM_THRESHOLD = 0.05

# =============================================================================
# VIBE PRESETS
# =============================================================================

VIBE_PRESETS = {
    'disposable':  {'enable_ca': True,  'ca_strength': 0.010, 'softness': 0.6, 'sharpness': 2.0, 'sharpen_radius': 0.5, 'grain': 1.2, 'vignette': 0.10, 'vignette_feather': 0.4, 'bloom': 0.10, 'lut': 'assets/luts/disposable_full_smoothed.cube'},
    'point_shoot': {'enable_ca': True,  'ca_strength': 0.002, 'softness': 0.5, 'sharpness': 0.5, 'sharpen_radius':  1.0, 'grain': 0.8, 'vignette': 0.10, 'vignette_feather': 1.0, 'bloom': 0.10, 'lut': 'assets/luts/pointandshoot.cube'},
    'rangefinder': {'enable_ca': False, 'ca_strength': 0.0,   'softness': 0.2, 'sharpness': 0.8, 'sharpen_radius':  1.0, 'grain': 0.5, 'vignette': 0.05, 'vignette_feather': 1.0, 'bloom': 0.05, 'lut': 'assets/luts/rangefinder.cube'},
    'monochrome':  {'enable_ca': False, 'ca_strength': 0.0,   'softness': 0.2, 'sharpness': 0.8, 'sharpen_radius':  1.0, 'grain': 1.5, 'vignette': 0.20, 'vignette_feather': 1.0, 'bloom': 0.05, 'lut': 'assets/luts/monochrome.cube'},
}

# =============================================================================
# DEBUG / TIMING
# =============================================================================

# Set to True to print per-effect timing to console (development only)
DEBUG_TIMING = True

def _timing_print(msg):
    """Print timing/debug messages. Controlled by DEBUG_TIMING flag."""
    if DEBUG_TIMING:
        print(msg)

# =============================================================================
# RUNTIME DEBUG CONFIGURATION
# =============================================================================

class DebugConfig:
    """
    Runtime configuration for the debug panel.
    Class-level attributes act as global toggles and tunable parameters.
    """
    # Toggles
    enable_halation = True
    enable_chromatic_aberration = True
    enable_softness = True
    enable_grain = True
    enable_sharpen = True
    enable_cnr = True
    enable_lut = True
    enable_pre_lut_dither = True
    enable_vignette = True
    enable_bloom = True
    enable_reverse_autoexposure = False

    # Reference exposure time in seconds — the "middleground". Shots with a
    # shorter ExposureTime get boosted; longer get cut. Tune empirically
    # against your film scans.
    reverse_autoexposure_t_ref = 1e-3

    # Static post-AE exposure boost (linear gain in EV). Must match the value
    # used to build the LUT — see POST_AE_EXPOSURE_BOOST_EV above.
    enable_post_ae_exposure_boost = False
    post_ae_exposure_boost_ev = POST_AE_EXPOSURE_BOOST_EV

    # Parameters (initialized from constants above)
    halation_threshold = HALATION_THRESHOLD
    halation_blur_radius = HALATION_BLUR_RADIUS
    halation_strength = HALATION_STRENGTH
    ca_strength = CHROMATIC_ABERRATION_STRENGTH
    ca_steps = CHROMATIC_ABERRATION_STEPS
    ca_blue_blur = CHROMATIC_ABERRATION_BLUE_BLUR
    softness_sigma = SOFTNESS_SIGMA
    grain_strength = GRAIN_STRENGTH
    sharpen_strength = SHARPEN_STRENGTH
    sharpen_radius = SHARPEN_RADIUS
    cnr_sigma = CNR_SIGMA
    enable_highlight_desat = True
    highlight_desat_threshold_L = HIGHLIGHT_DESAT_THRESHOLD_L
    highlight_desat_rolloff_L   = HIGHLIGHT_DESAT_ROLLOFF_L
    highlight_desat_sigma       = HIGHLIGHT_DESAT_SIGMA
    pre_lut_dither_strength = DITHER_STRENGTH
    vignette_strength = VIGNETTE_STRENGTH
    vignette_color_shift = VIGNETTE_COLOR_SHIFT
    vignette_feather = VIGNETTE_FEATHER
    bloom_strength = BLOOM_STRENGTH
    bloom_threshold = BLOOM_THRESHOLD

    # Path to the active LUT (relative for bundled, absolute for user-loaded).
    # Empty means: fall back to whatever the processor loaded at construction.
    lut_path = ''

    @classmethod
    def reset(cls):
        """Reset all parameters to module defaults.
        Note: if new parameters are added, this method must also be updated.
        """
        cls.enable_halation = True
        cls.enable_chromatic_aberration = True
        cls.enable_softness = True
        cls.enable_grain = True
        cls.enable_sharpen = True
        cls.enable_cnr = True
        cls.enable_lut = True
        cls.enable_pre_lut_dither = True
        cls.enable_vignette = True
        cls.enable_bloom = True
        cls.enable_reverse_autoexposure = False
        cls.reverse_autoexposure_t_ref = 1e-3
        cls.enable_post_ae_exposure_boost = False
        cls.post_ae_exposure_boost_ev = POST_AE_EXPOSURE_BOOST_EV

        cls.halation_threshold = HALATION_THRESHOLD
        cls.halation_blur_radius = HALATION_BLUR_RADIUS
        cls.halation_strength = HALATION_STRENGTH
        cls.ca_strength = CHROMATIC_ABERRATION_STRENGTH
        cls.ca_steps = CHROMATIC_ABERRATION_STEPS
        cls.ca_blue_blur = CHROMATIC_ABERRATION_BLUE_BLUR
        cls.softness_sigma = SOFTNESS_SIGMA
        cls.grain_strength = GRAIN_STRENGTH
        cls.sharpen_strength = SHARPEN_STRENGTH
        cls.sharpen_radius = SHARPEN_RADIUS
        cls.cnr_sigma = CNR_SIGMA
        cls.enable_highlight_desat = True
        cls.highlight_desat_threshold_L = HIGHLIGHT_DESAT_THRESHOLD_L
        cls.highlight_desat_rolloff_L   = HIGHLIGHT_DESAT_ROLLOFF_L
        cls.highlight_desat_sigma       = HIGHLIGHT_DESAT_SIGMA
        cls.pre_lut_dither_strength = DITHER_STRENGTH
        cls.vignette_strength = VIGNETTE_STRENGTH
        cls.vignette_color_shift = VIGNETTE_COLOR_SHIFT
        cls.vignette_feather = VIGNETTE_FEATHER
        cls.bloom_strength = BLOOM_STRENGTH
        cls.bloom_threshold = BLOOM_THRESHOLD
        cls.lut_path = ''


# =============================================================================
# VIBE STATE SCHEMA
# =============================================================================

# Ordered list of (field_name, type) tuples — every field that participates
# in vibe state. Used by serialization, factory_state_for, and the panel sync.
VIBE_FIELDS = [
    ('enable_halation', bool),
    ('enable_chromatic_aberration', bool),
    ('enable_softness', bool),
    ('enable_grain', bool),
    ('enable_sharpen', bool),
    ('enable_cnr', bool),
    ('enable_lut', bool),
    ('enable_pre_lut_dither', bool),
    ('enable_highlight_desat', bool),
    ('enable_vignette', bool),
    ('enable_bloom', bool),
    ('halation_threshold', float),
    ('halation_blur_radius', float),
    ('halation_strength', float),
    ('ca_strength', float),
    ('ca_steps', int),
    ('ca_blue_blur', float),
    ('softness_sigma', float),
    ('grain_strength', float),
    ('sharpen_strength', float),
    ('sharpen_radius', float),
    ('cnr_sigma', float),
    ('highlight_desat_threshold_L', float),
    ('highlight_desat_rolloff_L', float),
    ('highlight_desat_sigma', float),
    ('pre_lut_dither_strength', float),
    ('vignette_strength', float),
    ('vignette_color_shift', float),
    ('vignette_feather', float),
    ('bloom_strength', float),
    ('bloom_threshold', float),
    ('lut_path', str),
]

# Captured at import time, before any user code mutates DebugConfig — this is
# the "fresh-install" baseline shared across all vibes for non-vibe-specific
# fields (halation, CNR, dither, etc.).
_FACTORY_BASE = {name: getattr(DebugConfig, name) for name, _ in VIBE_FIELDS}


def factory_state_for(vibe_id: str) -> dict:
    """Return the bundled factory state for `vibe_id` — all VIBE_FIELDS values
    the app would use on a fresh install. Vibe-specific fields come from
    VIBE_PRESETS; everything else is the global baseline.
    """
    state = dict(_FACTORY_BASE)
    preset = VIBE_PRESETS[vibe_id]
    state['enable_chromatic_aberration'] = preset['enable_ca']
    state['ca_strength'] = preset['ca_strength']
    state['softness_sigma'] = preset['softness']
    state['sharpen_strength'] = preset['sharpness']
    state['sharpen_radius'] = preset['sharpen_radius']
    state['grain_strength'] = preset['grain']
    state['vignette_strength'] = preset['vignette']
    state['vignette_feather'] = preset.get('vignette_feather', 1.0)
    state['bloom_strength'] = preset['bloom']
    state['lut_path'] = preset['lut']
    return state


def snapshot_debug_config() -> dict:
    """Capture the current DebugConfig as a vibe-state dict (only VIBE_FIELDS)."""
    return {name: getattr(DebugConfig, name) for name, _ in VIBE_FIELDS}


def apply_state_to_debug_config(state: dict) -> None:
    """Write a vibe-state dict into DebugConfig, coercing types and ignoring unknown keys."""
    for name, t in VIBE_FIELDS:
        if name in state:
            try:
                setattr(DebugConfig, name, t(state[name]))
            except (TypeError, ValueError):
                pass
