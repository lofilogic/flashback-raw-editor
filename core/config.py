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
HALATION_THRESHOLD = 0.6
HALATION_THRESHOLD_FUJI = 0.7
HALATION_BLUR_RADIUS = 4
HALATION_STRENGTH = 0.5
SOFTNESS_SIGMA = 0.5
GRAIN_STRENGTH = 0.01
GRAIN_BLUR_SIGMA = 0.7
SHARPEN_STRENGTH = 0.5
SHARPEN_RADIUS = 1.0
CNR_SIGMA = 2.0
HIGHLIGHT_DESAT_THRESHOLD_L = 60.0   # Lab L* at which desaturation begins (0-100)
HIGHLIGHT_DESAT_ROLLOFF_L   = 10.0   # width of ramp in L* units
HIGHLIGHT_DESAT_SIGMA       = 10.0    # spatial Gaussian blur on the mask
DITHER_STRENGTH = 0.005

# =============================================================================
# VIBE PRESETS
# =============================================================================

VIBE_PRESETS = {
    'disposable':  {'enable_ca': True,  'ca_strength': 0.010, 'softness': 0.5, 'sharpness': 0.2, 'sharpen_radius': 10.0, 'grain': 0.020, 'lut': 'assets/luts/disposable_smoothed.cube'},
    'point_shoot': {'enable_ca': True,  'ca_strength': 0.002, 'softness': 0.5, 'sharpness': 0.5, 'sharpen_radius':  1.0, 'grain': 0.010, 'lut': 'assets/luts/pointandshoot.cube'},
    'rangefinder': {'enable_ca': False, 'ca_strength': 0.0,   'softness': 0.2, 'sharpness': 0.8, 'sharpen_radius':  1.0, 'grain': 0.007, 'lut': 'assets/luts/rangefinder.cube'},
    'monochrome':  {'enable_ca': False, 'ca_strength': 0.0,   'softness': 0.2, 'sharpness': 0.8, 'sharpen_radius':  1.0, 'grain': 0.020, 'lut': 'assets/luts/monochrome.cube'},
}

# =============================================================================
# DEBUG / TIMING
# =============================================================================

# Set to True to print per-effect timing to console (development only)
DEBUG_TIMING = False

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
        cls.enable_reverse_autoexposure = False
        cls.reverse_autoexposure_t_ref = 1e-3
        cls.enable_post_ae_exposure_boost = False
        cls.post_ae_exposure_boost_ev = POST_AE_EXPOSURE_BOOST_EV

        cls.halation_threshold = HALATION_THRESHOLD
        cls.halation_blur_radius = HALATION_BLUR_RADIUS
        cls.halation_strength = HALATION_STRENGTH
        cls.ca_strength = CHROMATIC_ABERRATION_STRENGTH
        cls.ca_steps = CHROMATIC_ABERRATION_STEPS
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
