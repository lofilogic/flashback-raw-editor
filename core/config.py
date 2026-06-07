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
import math as _math
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

# User-facing effect defaults. Units are documented per-field on VibeConfig.
# Conversion to the internal scalars the effect functions expect happens in
# the conversion helpers below; storage and UI both use these user-facing
# numbers.
CA_PIXELS = 5.0            # edge pixels of blue offset at the long edge of the rendered frame
# Legacy CA params — UNUSED by the current spectral CA (gpu.ca_frame /
# effects.apply_chromatic_aberration), which is driven solely by ca_pixels. Kept
# so existing presets/saved projects/UI don't break; candidates for repurposing
# or removal in a future cleanup.
CA_STEPS = 4
CA_BLUE_BLUR = 0.3         # px
CA_ZOOM_BLUR_PCT = 100.0   # percent
HALATION_THRESHOLD_STOPS = 4.0   # EV above middle grey
HALATION_BLUR_RADIUS = 4.0 # px
HALATION_STRENGTH_PCT = 50.0
SOFTNESS_SIGMA = 0.5       # px
# Edge (corner) softness — a radial defocus that grows toward the frame corners,
# emulating lens field curvature. Distinct from the global `softness` blur.
EDGE_SOFTNESS_STRENGTH_PCT = 60.0   # 0–100 → max sharp→blur blend at the corners
EDGE_SOFTNESS_SIGMA = 3.0           # px, blur radius of the soft copy
EDGE_SOFTNESS_START_PCT = 40.0      # 0–100 → radius (as % of corner) where softness begins
GRAIN_STRENGTH_PCT = 50.0
GRAIN_TILE_SCALE = 0.8     # <1.0 makes grain finer (tiles render denser); >1.0 makes it chunkier.
GRAIN_HIGHLIGHT_BIAS = 0.3 # 1.0 = grain biased to highlights, 0.0 = shadows, 0.5 = flat.
SHARPEN_STRENGTH_PCT = 50.0
SHARPEN_RADIUS = 1.0       # px
CNR_AMOUNT_PCT = 20.0      # sigma 8 at _CNR_SIGMA_MAX=20 (a touch under old "200%")
CNR_DESPIKE_PCT = 60.0        # chroma firefly/outlier clamp; 0 = off
CNR_DESPIKE_BIAS_PCT = 75.0  # 0 = symmetric, 100 = green (-a*) only
VIGNETTE_STRENGTH_PCT = 50.0
VIGNETTE_COLOR_PCT = 25.0
VIGNETTE_CURVE = 0.0       # -100…+100, higher = more feathered (softer)
BLOOM_STRENGTH_PCT = 30.0
BLOOM_THRESHOLD_STOPS = 3.0      # EV above middle grey

# Internal scalar maxima — the user-facing percent fields map 0–100 onto
# 0–MAX. Keeping these explicit makes the migration buckets trivial to
# write and makes the panel/pipeline agree on the same conversion.
_CNR_SIGMA_MAX = 20.0
# Despike clamp limits in Lab a*/b* units: gentle band at amount→0+, tight at
# amount→100. Smooth colour stays inside the band; only spikes get pulled back.
_CNR_DESPIKE_T_HI = 40.0
_CNR_DESPIKE_T_LO = 4.0
_VIGNETTE_COLOR_MAX = 0.2


# =============================================================================
# UNIT CONVERSIONS  (user-facing values  →  internal effect scalars)
# =============================================================================
# Each helper takes a value as stored on VibeConfig and returns what the
# effect function actually consumes. The pipeline calls these at the
# effect-function boundary in core/processor.py.

def ca_pixels_to_scale(pixels: float, long_edge: int) -> float:
    """Edge-pixel offset → CA radial scale factor, normalised by the LONG edge.

    CA samples are displaced radially by s * radius; ``s = pixels / (long_edge/2)``
    so the displacement at the long half-edge is exactly ``pixels``. Normalising
    by the long edge (max(W, H)) makes the fringe invariant to orientation and to
    post-shoot 90° rotation — rotation swaps W and H but not their max — so a
    portrait and a landscape framing of the same scene fringe identically, as a
    real lens does. For a landscape frame the long edge IS the width, so existing
    ca_pixels values are unchanged; only portrait/rotated frames are corrected.
    """
    if long_edge <= 0:
        return 0.0
    return float(pixels) / (float(long_edge) / 2.0)


def pct(value: float) -> float:
    """0–N percent → 0–N/100 (the generic [0,1] mapping)."""
    return float(value) / 100.0


def vignette_curve_to_power(curve: float) -> float:
    """Symmetric -100…+100 curve → cosine-falloff exponent.

    0 → 1.0 (neutral). Higher = softer / more feathered (exponent < 1
    keeps falloff high until near the corners). Lower = harder edge
    (exponent > 1 pulls darkening inward).
    """
    return float(2.0 ** (-float(curve) / 50.0))


def cnr_pct_to_sigma(amount_pct: float) -> float:
    return pct(amount_pct) * _CNR_SIGMA_MAX


def cnr_sigma_color(sigma: float) -> float:
    """Bilateral range sigma for chroma NR, scaled with the spatial sigma.

    A fixed range sigma capped the strength: the bilateral preserved chroma
    variation near edges, so cranking the spatial sigma plateaued (it never
    removed the last ~12% of chroma noise). Scaling the range tolerance with the
    spatial sigma lets high settings approach a near-Gaussian chroma blur (much
    stronger), while a floor of 15 keeps low settings edge-preserving (no colour
    bleed across saturated boundaries). Single source of truth for both the
    numpy/cv2 path and gpu.cnr_frame.
    """
    return max(15.0, float(sigma) * 3.0)


def cnr_despike_thresholds(amount_pct: float, bias_pct: float) -> tuple:
    """Per-direction Lab clamp limits for the chroma despike prepass.

    Returns ``(thr_green, thr_other)`` in a*/b* units. The prepass clamps each
    chroma channel into ``[median +/- thr]`` of its 3x3 neighbourhood, killing
    isolated colour spikes (fireflies) while leaving smooth colour untouched (a
    smooth region equals its own median, so the deviation is ~0). A bilateral
    filter is edge-preserving and treats a one-pixel spike as an edge to keep,
    which is why cranking cnr_amount washes out detail before it removes the
    spike — this is the tool that actually removes it.

    The green direction (a* below the median) uses ``thr_green``; magenta (a*
    above) and both b* directions use ``thr_other``. ``bias`` widens
    ``thr_other`` so 100% acts on green only while 0% is symmetric. ``amount<=0``
    returns ``(0, 0)`` = off. Single source of truth for the cv2 path and
    gpu.cnr_frame.
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
    """Stops above 18% middle grey → ACEScct-encoded threshold.

    The bloom/halation passes mask on ACEScct-encoded luminance, which is
    why the prior 0–100% slider was opaque (ACEScct is a log encoding, so
    65% sat ~1.7 stops above scene white, not at "65% brightness"). This
    helper takes a photographer-friendly EV value and produces the same
    ACEScct number the effect functions expect.

    Skips the toe branch of the ACEScct encoder: anything brighter than
    linear 0.0078 is in the log range, which covers all sensible stops
    values (the toe crosses linear at acescct ≈ 0.155, equivalent to
    roughly -4.5 stops below middle grey — well below any threshold the
    effects care about).
    """
    linear = _MID_GREY_LINEAR * (2.0 ** float(stops))
    return float((_math.log2(max(linear, 1e-10)) + 9.72) / 17.52)

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
    enable_edge_softness: bool = False
    enable_grain: bool = True
    enable_sharpen: bool = True
    enable_cnr: bool = True
    enable_lut: bool = True
    enable_vignette: bool = True
    enable_bloom: bool = True

    # ---- effect parameters (user-facing units; see conversion helpers) ----
    # Percent fields are stored as 0–N where N is each effect's natural max
    # (100 for clamped effects, 200/300/500 for ones that can over-drive).
    # Pixel fields are explicit pixel counts. Threshold fields are in EV
    # (stops) above 18% middle grey — 0 = middle grey, +N = N stops
    # brighter, default ≈ +4 (just into the specular highlight range).
    # vignette_curve is signed -100…+100 with 0 = neutral, positive = softer.
    halation_threshold_stops: float = HALATION_THRESHOLD_STOPS  # EV above mid grey
    halation_blur_radius: float = HALATION_BLUR_RADIUS         # px
    halation_strength_pct: float = HALATION_STRENGTH_PCT       # 0–300
    ca_pixels: float = CA_PIXELS                                # edge px @ long edge
    ca_steps: int = CA_STEPS
    ca_blue_blur: float = CA_BLUE_BLUR                          # px
    ca_zoom_blur_pct: float = CA_ZOOM_BLUR_PCT                  # 0–500
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
    # Tagged LUT reference. One of:
    #   ""                     — no LUT (tone-curve fallback path)
    #   "factory:<id>"         — looked up in FACTORY_LUTS against the
    #                            current build's asset dir
    #   "user:<absolute path>" — user-imported .cube file on disk
    # Factory ids decouple a saved vibe from the install-time on-disk
    # location of bundled LUTs, so a moved/upgraded install can't load the
    # wrong file. The migrator rewrites legacy `lut_path` strings into
    # this tagged form.
    lut_ref: str = ''
    dng_profile_name: str = 'Flashback Standard'

    # Pre-1.5 custom LUT path, preserved by the migrator. Purely
    # informational — never read by the pipeline. Lets users find and
    # re-import their original .cube once they've regenerated it against
    # the v2 color pipeline. Cleared once the user re-imports a LUT.
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

# Preset values are in user-facing units (see the conversion helpers above).
# vignette_curve = -50 * log2(power), so the previous feather=0.4 / "softer"
# maps to curve ≈ +66.
# =============================================================================
# LUT REGISTRY — factory id → bundled file (relative to the install root,
# resolved through resource_path at load time so PyInstaller bundles and
# dev runs both work). Saved vibes store these ids, never raw paths, so a
# moved install never silently picks up the wrong file.
# =============================================================================

FACTORY_LUTS = {
    'disposable':           'assets/luts/disposable.cube',
    'disposable_v1':        'assets/luts/disposable_V1.cube',
    'flashback_classic_v1': 'assets/luts/V1.cube',
    'point_shoot':          'assets/luts/pointandshoot.cube',
    'rangefinder':          'assets/luts/rangefinder.cube',
    'monochrome':           'assets/luts/monochrome.cube',
}

# Tag prefixes used on VibeConfig.lut_ref. Keep these as the single source
# of truth — sites that build or parse refs must use the constants below.
LUT_REF_FACTORY = 'factory:'
LUT_REF_USER = 'user:'

# Per-file-type LUT overrides. V1 negatives have a flatter, lower-DR capture
# than V2 DNGs, so the disposable look needs a LUT tuned for them. The override
# is transient (display/export only) — it never gets written back into the
# saved vibe, and it only swaps a *factory* ref, never a user-imported LUT.
_V1_LUT_OVERRIDES = {
    LUT_REF_FACTORY + 'disposable': LUT_REF_FACTORY + 'disposable_v1',
}


def effective_lut_ref(base_ref: str, is_v1: bool) -> str:
    """Resolve the LUT ref actually used to render a frame.

    For V1 negatives, swap in the V1-tuned variant of a factory look where one
    exists; everything else (V2 files, user LUTs, looks without a V1 variant)
    passes through unchanged.
    """
    if is_v1:
        return _V1_LUT_OVERRIDES.get(base_ref, base_ref)
    return base_ref


def resolve_lut_ref(ref: str):
    """Resolve a tagged LUT reference to an absolute filesystem path.

    Returns (absolute_path, origin) where origin ∈ {'factory', 'user', None}.
    Returns (None, None) for an empty ref. Returns (None, origin) when the
    referenced LUT cannot be found — the caller decides whether to fall
    back to the vibe's factory LUT or surface a notice.
    """
    # Imported here (not at module top) to avoid a circular import:
    # core/__init__.py loads this module during package init.
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
    # Unknown tag — treat as missing rather than guessing.
    return None, None


# `ca_pixels` is the blue fringe offset in pixels at the long half-edge of the
# rendered frame (see ca_pixels_to_scale — normalised by the long edge so it's
# orientation-invariant). The pipeline develops raws with half_size=True, so the
# rendered width is half the sensor width (2072 px for the ONE35 V2); 2–8 px is
# the visual baseline these presets are calibrated against.
#
# `ca_zoom_blur_pct` in the presets is legacy/inert — the spectral CA is driven
# only by ca_pixels (the radial spectral spread subsumes the old zoom-blur pass).
VIBE_PRESETS = {
    'disposable':           {'enable_ca': True,  'ca_pixels': 8.0, 'ca_zoom_blur_pct': 150.0, 'softness': 0.3, 'sharpness_pct': 200.0, 'sharpen_radius': 0.5, 'grain_pct': 120.0, 'vignette_pct': 10.0, 'vignette_curve':  66.0, 'bloom_pct': 15.0, 'lut': 'factory:disposable'},
    'flashback_classic_v1': {'enable_ca': True,  'ca_pixels':  5.0, 'ca_zoom_blur_pct': 200.0, 'softness': 0.3, 'sharpness_pct':  80.0, 'sharpen_radius': 0.5, 'grain_pct': 200.0, 'vignette_pct': 10.0, 'vignette_curve':  66.0, 'bloom_pct':  3.0, 'lut': 'factory:flashback_classic_v1', 'base_exposure_offset_v2': 0.0},
    'point_shoot':          {'enable_ca': True,  'ca_pixels':  2.0, 'ca_zoom_blur_pct': 100.0, 'softness': 0.3, 'sharpness_pct':  50.0, 'sharpen_radius': 1.0, 'grain_pct':  80.0, 'vignette_pct': 10.0, 'vignette_curve':   0.0, 'bloom_pct': 10.0, 'lut': 'factory:point_shoot'},
    'rangefinder':          {'enable_ca': False, 'ca_pixels':  0.0, 'ca_zoom_blur_pct': 100.0, 'softness': 0.1, 'sharpness_pct':  80.0, 'sharpen_radius': 1.0, 'grain_pct':  50.0, 'vignette_pct':  5.0, 'vignette_curve':   0.0, 'bloom_pct':  5.0, 'lut': 'factory:rangefinder'},
    'monochrome':           {'enable_ca': False, 'ca_pixels':  0.0, 'ca_zoom_blur_pct': 100.0, 'softness': 0.1, 'sharpness_pct':  80.0, 'sharpen_radius': 1.0, 'grain_pct': 150.0, 'vignette_pct': 20.0, 'vignette_curve':   0.0, 'bloom_pct':  5.0, 'lut': 'factory:monochrome'},
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
    dictionary uses short keys (enable_ca, ca_pixels, softness, …);
    we map those onto the dataclass field names. All numeric preset
    values are in user-facing units (px, percent, signed curve).
    """
    cfg = VibeConfig()  # all factory defaults
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
    cfg.ca_zoom_blur_pct            = preset.get('ca_zoom_blur_pct', CA_ZOOM_BLUR_PCT)
    return cfg


# Names of every VibeConfig field — used by the debug panel to detect
# "modified from factory" state.
VIBE_FIELD_NAMES = tuple(f.name for f in fields(VibeConfig))
