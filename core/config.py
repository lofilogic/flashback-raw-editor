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

# Color spaces (initialized once at import time)
CS_SRGB = colour.RGB_COLOURSPACES['sRGB']
CS_REC2020 = colour.RGB_COLOURSPACES['ITU-R BT.2020']

# =============================================================================
# EFFECT DEFAULTS
# =============================================================================

CHROMATIC_ABERRATION_STRENGTH = 0.005
CHROMATIC_ABERRATION_STEPS = 4
HALATION_THRESHOLD = 0.8
HALATION_THRESHOLD_FUJI = 0.8
HALATION_BLUR_RADIUS = 5
HALATION_STRENGTH = 0.7
SOFTNESS_SIGMA = 0.5
GRAIN_STRENGTH = 0.01
GRAIN_BLUR_SIGMA = 0.7
SHARPEN_STRENGTH = 0.5
SHARPEN_RADIUS = 1.0
CNR_SIGMA = 0.7
HIGHLIGHT_DESAT_THRESHOLD_L = 68.0   # Lab L* at which desaturation begins (0-100)
HIGHLIGHT_DESAT_ROLLOFF_L   = 8.0   # width of ramp in L* units
HIGHLIGHT_DESAT_SIGMA       = 10.0    # spatial Gaussian blur on the mask
DITHER_STRENGTH = 0.005

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
