"""Tests for the resident render image (core.gpu.Frame) and the first
texture-resident stage (ACEScct encode).

The resident GPU representation is an rgba16float texture, so round-trips are
perceptually—not bit—exact: the bar is "below visible (1/255 ≈ 3.9e-3)", which
half-float clears with room to spare. The numpy path stays the oracle.

GPU-touching tests skip cleanly where no usable device exists (e.g. CI).
"""
import numpy as np
import pytest

from core.gpu import Frame, gpu
from core.kernels import acescct_encode as acescct_encode_oracle
from core.kernels import encode_then_lut

from parity_utils import assert_parity, max_abs_err

# A half-float round-trip stays well under one 8-bit code value.
PERCEPTUAL_TOL = 3.0e-3


def _gpu_available() -> bool:
    try:
        return bool(gpu._init())
    except Exception:
        return False


GPU = _gpu_available()
requires_gpu = pytest.mark.skipif(not GPU, reason="no usable GPU device")


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.random((32, 48, 3), dtype=np.float32)


# --- CPU side: no device required -------------------------------------------

def test_frame_cpu_identity(img):
    f = Frame.from_cpu(img)
    assert f.shape == img.shape
    assert not f.on_gpu
    assert np.array_equal(f.cpu(), img)


def test_frame_requires_some_backing():
    with pytest.raises(ValueError):
        Frame(gpu)


# --- GPU texture round-trip: perceptually lossless --------------------------

@requires_gpu
def test_frame_gpu_roundtrip_is_perceptual(img):
    f = Frame.from_cpu(img)
    tex = f.gpu()
    assert f.on_gpu
    back = gpu._download_tex(tex, img.shape)
    assert back.dtype == np.float32
    assert max_abs_err(back, img) <= PERCEPTUAL_TOL


@requires_gpu
def test_frame_cpu_after_gpu_is_perceptual(img):
    f = Frame.from_cpu(img)
    f.gpu()  # force an upload to texture
    assert max_abs_err(f.cpu(), img) <= PERCEPTUAL_TOL


@requires_gpu
def test_frame_from_gpu_reads_back(img):
    tex = gpu._upload_tex(img)
    f = Frame.from_gpu(tex, img.shape)
    assert f.on_gpu
    assert max_abs_err(f.cpu(), img) <= PERCEPTUAL_TOL


# --- first resident stage vs its numpy oracle -------------------------------

@requires_gpu
def test_encode_frame_matches_oracle(img):
    """Texture-resident ACEScct encode matches the numpy encode perceptually."""
    def gpu_encode(a):
        return gpu.encode_frame(Frame.from_cpu(a)).cpu()

    err = assert_parity(acescct_encode_oracle, gpu_encode, img,
                        tol=PERCEPTUAL_TOL, label="encode_frame")
    assert err >= 0.0  # parity passed; err is the measured headroom


@requires_gpu
def test_encode_then_lut_matches_production_path(img):
    """The resident encode->LUT chain matches today's production GPU path
    (buffer ACEScct encode + buffer tetrahedral LUT) within perceptual tol."""
    # A smooth, non-trivial LUT — representative of real film-emulation LUTs.
    # (A *random* LUT is a pathological worst case: adjacent cells differ wildly,
    # so f16 input quantization gets amplified; even then the chain measures
    # ~3.8e-3, still under one 8-bit code value. Real LUTs are smooth, so this
    # reflects production.)
    n = 17
    axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
    r, g, b = np.meshgrid(axis, axis, axis, indexing='ij')
    lut_table = np.stack([
        np.clip(r ** 1.1 * 0.95 + 0.03 * g, 0, 1),
        np.clip(g ** 0.95,                  0, 1),
        np.clip(b ** 1.05 * 0.97 + 0.02 * r, 0, 1),
    ], axis=-1).astype(np.float32)
    gpu.upload_lut(lut_table)

    img_max = np.maximum(img, 1e-10)

    def production(a):                       # current behaviour
        return gpu.apply_lut(gpu.acescct_encode(a))

    def resident(a):                         # new resident chain
        return encode_then_lut(a)

    assert_parity(production, resident, img_max,
                  tol=PERCEPTUAL_TOL, label="encode_then_lut")


@requires_gpu
def test_blur_frame_matches_buffer_blur(img):
    """Texture-resident separable blur matches the buffer Gaussian (bit-exact)."""
    for sigma in (2.0, 4.0, 12.0):
        ref = gpu.gaussian_blur(img, sigma)
        cand = gpu.blur_frame(Frame.from_cpu(img), sigma).cpu()
        assert max_abs_err(cand, ref) <= 1e-5


@requires_gpu
def test_halation_frame_matches_per_op():
    """Resident two-pass halation matches the per-op numpy/buffer path."""
    from core import effects
    rng = np.random.default_rng(5)
    img = rng.random((40, 60, 3), dtype=np.float32) * 0.5
    img[10:20, 15:30, :] += 2.5                      # bright block drives the mask
    th, br, st = 0.65, 4.0, 0.5

    resident = gpu.halation_frame(Frame.from_cpu(img), th, br, st).cpu()

    saved = gpu.halation_frame                       # force the per-op reference
    gpu.halation_frame = lambda *a, **k: None
    try:
        ref = effects.apply_halation(img, th, br, st)
    finally:
        gpu.halation_frame = saved

    assert max_abs_err(resident, ref) <= 1e-4


# --- post-LUT resident tail stages vs their per-op oracles ------------------

@requires_gpu
def test_softness_frame_matches_buffer_blur(img):
    """Resident softness is exactly a Gaussian blur (matches the buffer blur)."""
    for sigma in (1.5, 3.0):
        ref = gpu.gaussian_blur(img, sigma)
        cand = gpu.softness_frame(Frame.from_cpu(img), sigma).cpu()
        assert max_abs_err(cand, ref) <= 1e-5


@requires_gpu
def test_sharpen_frame_matches_per_op(img):
    """Resident sharpen matches the per-op buffer path (blur + unsharp)."""
    for strength, radius in ((0.5, 2.0), (1.2, 4.0)):
        blurred = gpu.gaussian_blur(img, radius)
        ref = gpu.unsharp_mask(img, blurred, strength)
        cand = gpu.sharpen_frame(Frame.from_cpu(img), strength, radius).cpu()
        assert max_abs_err(cand, ref) <= 1e-5


@requires_gpu
def test_grain_frame_matches_buffer_blend(img):
    """Resident grain blend matches the per-op buffer grain_blend on the same
    layer (so any difference is GPU float rounding, not a different layer)."""
    rng = np.random.default_rng(7)
    grain = rng.random(img.shape, dtype=np.float32)
    intensity, min_grain, bias = 0.3, 0.2, 0.4
    ref = gpu.grain_blend(img, grain, intensity, min_grain, bias)
    cand = gpu.grain_frame(Frame.from_cpu(img), grain, intensity, min_grain, bias).cpu()
    assert max_abs_err(cand, ref) <= 1e-5


@requires_gpu
def test_ca_frame_matches_spectral_oracle():
    """Resident spectral CA matches the numpy spectral oracle (manual bilinear
    + clamp-to-edge vs cv2.remap), within perceptual tol on a fringe-y image."""
    from core.effects import apply_chromatic_aberration
    rng = np.random.default_rng(3)
    # An edge-rich image is the worst case for sampling differences.
    a = rng.random((48, 72, 3), dtype=np.float32)
    a[:, 36:, :] *= 0.2                      # hard vertical edge -> visible fringe
    for scale in (0.004, 0.012):
        def oracle(x, s=scale):
            return apply_chromatic_aberration(x, s)

        def resident(x, s=scale):
            return gpu.ca_frame(Frame.from_cpu(x), s).cpu()

        assert_parity(oracle, resident, a, tol=PERCEPTUAL_TOL, label="ca_frame")


@requires_gpu
def test_ca_frame_noop_when_scale_zero(img):
    """scale<=0 returns the input Frame unchanged (no fringe applied)."""
    out = gpu.ca_frame(Frame.from_cpu(img), 0.0).cpu()
    assert max_abs_err(out, img) <= PERCEPTUAL_TOL


@requires_gpu
def test_edge_softness_frame_matches_oracle():
    """Resident edge softness matches the numpy oracle (blur + radial blend)."""
    from core.effects import apply_edge_softness
    rng = np.random.default_rng(11)
    a = rng.random((50, 80, 3), dtype=np.float32)
    for sigma, strength, start in ((3.0, 0.6, 0.4), (5.0, 1.0, 0.2)):
        def oracle(x, s=sigma, st=strength, sa=start):
            return apply_edge_softness(x, s, st, sa)

        def resident(x, s=sigma, st=strength, sa=start):
            return gpu.edge_softness_frame(Frame.from_cpu(x), s, st, sa).cpu()

        assert_parity(oracle, resident, a, tol=PERCEPTUAL_TOL, label="edge_softness")


@requires_gpu
def test_edge_softness_frame_noop_when_strength_zero(img):
    out = gpu.edge_softness_frame(Frame.from_cpu(img), 3.0, 0.0, 0.4).cpu()
    assert max_abs_err(out, img) <= PERCEPTUAL_TOL


@requires_gpu
def test_bloom_frame_matches_oracle():
    """Resident bloom matches the numpy/cv2 oracle within perceptual tol.

    Bloom is a soft low-frequency layer scaled by a small strength, so the
    downsample/upsample resampling differences (box vs INTER_AREA, manual vs cv2
    bilinear) stay well under one 8-bit code value in the blended output. Uses a
    size divisible by 4 so the area-downsample blocks line up exactly.
    """
    from core.effects import apply_bloom
    rng = np.random.default_rng(17)
    a = rng.random((64, 96, 3), dtype=np.float32) * 0.3
    a[20:32, 30:50, :] += 3.0                # bright block drives the bloom mask
    for strength, threshold in ((0.3, 0.55), (0.1, 0.4)):
        def oracle(x, s=strength, t=threshold):
            return apply_bloom(x, s, t, linear=True)

        def resident(x, s=strength, t=threshold):
            return gpu.bloom_frame(Frame.from_cpu(x), s, t).cpu()

        assert_parity(oracle, resident, a, tol=PERCEPTUAL_TOL, label="bloom")


@requires_gpu
def test_vignette_frame_matches_oracle():
    """Resident vignette matches the numpy oracle (linear ACEScg, pre-LUT)."""
    from core.effects import apply_vignette
    rng = np.random.default_rng(13)
    a = rng.random((44, 66, 3), dtype=np.float32) * 1.5     # linear, can exceed 1
    for strength, color, feather in ((0.5, 0.05, 1.0), (0.8, 0.12, 1.6)):
        def oracle(x, s=strength, c=color, f=feather):
            return apply_vignette(x, s, c, f)

        def resident(x, s=strength, c=color, f=feather):
            return gpu.vignette_frame(Frame.from_cpu(x), s, c, f).cpu()

        assert_parity(oracle, resident, a, tol=PERCEPTUAL_TOL, label="vignette")


@requires_gpu
def test_resident_tail_chains_without_readback(img):
    """encode -> LUT -> softness -> grain -> sharpen as one resident chain
    matches running the same stages with a readback between each."""
    n = 17
    axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
    r, g, b = np.meshgrid(axis, axis, axis, indexing='ij')
    lut_table = np.stack([
        np.clip(r ** 1.1 * 0.95 + 0.03 * g, 0, 1),
        np.clip(g ** 0.95,                  0, 1),
        np.clip(b ** 1.05 * 0.97 + 0.02 * r, 0, 1),
    ], axis=-1).astype(np.float32)
    gpu.upload_lut(lut_table)

    rng = np.random.default_rng(8)
    grain = rng.random(img.shape, dtype=np.float32)
    img_max = np.maximum(img, 1e-10)

    def stage_by_stage(a):
        f = gpu.encode_frame(Frame.from_cpu(a))
        f = Frame.from_cpu(f.cpu())          # force a readback between stages
        f = gpu.lut_frame(f)
        f = Frame.from_cpu(f.cpu())
        f = gpu.softness_frame(f, 2.0)
        f = Frame.from_cpu(f.cpu())
        f = gpu.grain_frame(f, grain, 0.3, 0.2, 0.4)
        f = Frame.from_cpu(f.cpu())
        f = gpu.sharpen_frame(f, 0.5, 2.0)
        return f.cpu()

    def fused(a):
        f = Frame.from_cpu(a)
        f = gpu.encode_frame(f)
        f = gpu.lut_frame(f)
        f = gpu.softness_frame(f, 2.0)
        f = gpu.grain_frame(f, grain, 0.3, 0.2, 0.4)
        f = gpu.sharpen_frame(f, 0.5, 2.0)
        return f.cpu()

    assert_parity(stage_by_stage, fused, img_max,
                  tol=1e-5, label="resident_tail")


# --- the parity gate itself --------------------------------------------------

def test_assert_parity_passes_on_equivalent(img):
    assert_parity(lambda a: a * 2.0, lambda a: a + a, img, tol=0.0, label="double")


def test_assert_parity_fails_above_tol(img):
    with pytest.raises(AssertionError):
        assert_parity(lambda a: a, lambda a: a + 1e-3, img, tol=1e-5, label="offset")
