"""Tests for Frame and the GPU stages against their numpy versions.

Tolerances are mostly "below one 8-bit code value". GPU tests skip without
a device (CI).
"""
import numpy as np
import pytest

from core.gpu import Frame, gpu
from core.kernels import acescct_encode as acescct_encode_oracle
from core.kernels import encode_then_lut

from parity_utils import assert_parity, max_abs_err

# Under one 8-bit code value (1/255).
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
    """Texture encode->LUT matches the buffer encode + buffer LUT."""
    # Smooth, like real LUTs.
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

    def production(a):
        return gpu.apply_lut(gpu.acescct_encode(a))

    def resident(a):
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
def test_halation_frame_approximates_per_op():
    """GPU halation blurs at half resolution, so it only approximates the CPU
    version. The disc's hard rim is where they differ most: ~0.04 in linear
    ACEScg on this synthetic image, under a code value on real ones."""
    from core import effects
    rng = np.random.default_rng(5)
    img = rng.random((40, 60, 3), dtype=np.float32) * 0.5
    img[10:20, 15:30, :] += 2.5                      # bright block drives the mask
    th, br, st = 0.65, 4.0, 0.5

    resident = gpu.halation_frame(Frame.from_cpu(img), th, br, st).cpu()

    saved = gpu.halation_frame                       # force the CPU reference
    gpu.halation_frame = lambda *a, **k: None
    try:
        ref = effects.apply_halation(img, th, br, st)
    finally:
        gpu.halation_frame = saved

    assert max_abs_err(resident, ref) <= 0.06


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
    """Texture grain blend matches the buffer grain_blend on the same layer."""
    rng = np.random.default_rng(7)
    grain = rng.random(img.shape, dtype=np.float32)
    intensity, min_grain, bias = 0.3, 0.2, 0.4
    ref = gpu.grain_blend(img, grain, intensity, min_grain, bias)
    cand = gpu.grain_frame(Frame.from_cpu(img), grain, intensity, min_grain, bias).cpu()
    assert max_abs_err(cand, ref) <= 1e-5


@requires_gpu
def test_ca_frame_matches_spectral_oracle():
    """GPU CA matches the numpy version on an image with hard edges."""
    from core.effects import apply_chromatic_aberration
    rng = np.random.default_rng(3)
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


_AP1_LUMA = np.array([0.2722, 0.6741, 0.0537], dtype=np.float32)


@requires_gpu
def test_color_transform_matches_numpy():
    """GPU 3x3 colour transform matches (img.reshape(-1,3) @ M.T)."""
    rng = np.random.default_rng(31)
    a = rng.random((37, 53, 3), dtype=np.float32) * 2.0 - 0.3   # incl. negatives
    M = np.array([[0.53, 0.22, 0.21],
                  [0.09, 0.99, -0.07],
                  [0.05, -0.37, 1.15]], dtype=np.float32)
    ref = (a.reshape(-1, 3) @ M.T).reshape(a.shape)
    cand = gpu.color_transform(a, M)
    assert max_abs_err(ref, cand) <= 1e-5


@requires_gpu
def test_cnr_frame_preserves_luma():
    """CNR preserves AP1 luminance."""
    rng = np.random.default_rng(21)
    a = rng.random((40, 60, 3), dtype=np.float32) * 0.8 + 0.05
    out = gpu.cnr_frame(Frame.from_cpu(a), sigma=3.0).cpu()
    luma_in = a @ _AP1_LUMA
    luma_out = out @ _AP1_LUMA
    assert max_abs_err(luma_in, luma_out) <= 1e-3


@requires_gpu
def test_cnr_frame_reduces_chroma_noise():
    """CNR reduces chroma noise on a flat patch."""
    rng = np.random.default_rng(22)
    base = np.full((48, 48, 3), 0.3, dtype=np.float32)
    noisy = base + rng.normal(0, 0.05, base.shape).astype(np.float32)
    out = gpu.cnr_frame(Frame.from_cpu(noisy), sigma=4.0).cpu()
    # chroma = colour minus its own luma (broadcast); compare spread
    def chroma_std(x):
        return float((x - (x @ _AP1_LUMA)[..., None]).std())
    assert chroma_std(out) < 0.6 * chroma_std(noisy)


@requires_gpu
def test_cnr_frame_vs_cv2_interior_sanity():
    """GPU CNR roughly matches cv2 away from the borders, which use different
    modes."""
    from core.effects import reduce_color_noise_chroma
    rng = np.random.default_rng(23)
    a = rng.random((48, 64, 3), dtype=np.float32)
    ref = reduce_color_noise_chroma(a, sigma=3.0)[4:-4, 4:-4]
    cand = gpu.cnr_frame(Frame.from_cpu(a), sigma=3.0).cpu()[4:-4, 4:-4]
    assert max_abs_err(ref, cand) <= 2e-2


def test_cnr_despike_removes_green_firefly():
    """Despike pulls in an isolated green spike and leaves clean pixels."""
    from core.effects import reduce_color_noise_chroma, _acescg_to_lab
    from core.config import cnr_despike_thresholds
    img = np.full((16, 16, 3), 0.2, dtype=np.float32)
    img[8, 8] = [0.04, 0.7, 0.04]               # bright green firefly (very -a*)
    a_before = _acescg_to_lab(img)[8, 8, 1]
    out = reduce_color_noise_chroma(img, sigma=0.0,
                                    despike=cnr_despike_thresholds(100.0, 60.0))
    a_after = _acescg_to_lab(out)[8, 8, 1]
    assert a_before < -50.0                       # genuinely a strong green spike
    assert abs(a_after) < 0.2 * abs(a_before)     # clamped most of the way back
    assert np.allclose(img[0, 0], out[0, 0], atol=1e-4)   # clean field untouched


def test_cnr_despike_green_bias_spares_magenta():
    """At 100% bias, green spikes are clamped and magenta ones are not."""
    from core.effects import reduce_color_noise_chroma, _acescg_to_lab
    from core.config import cnr_despike_thresholds
    despike = cnr_despike_thresholds(100.0, 100.0)
    green = np.full((16, 16, 3), 0.2, dtype=np.float32); green[8, 8] = [0.04, 0.7, 0.04]
    magenta = np.full((16, 16, 3), 0.2, dtype=np.float32); magenta[8, 8] = [0.7, 0.04, 0.7]
    g_out = reduce_color_noise_chroma(green, sigma=0.0, despike=despike)
    m_out = reduce_color_noise_chroma(magenta, sigma=0.0, despike=despike)
    g_red = abs(_acescg_to_lab(green)[8, 8, 1]) - abs(_acescg_to_lab(g_out)[8, 8, 1])
    m_red = abs(_acescg_to_lab(magenta)[8, 8, 1]) - abs(_acescg_to_lab(m_out)[8, 8, 1])
    assert g_red > 50.0          # green spike strongly clamped
    assert m_red < 1.0           # magenta spike essentially preserved


@requires_gpu
def test_cnr_despike_frame_vs_cv2_interior():
    """GPU despike matches the CPU version away from the borders."""
    from core.effects import reduce_color_noise_chroma
    from core.config import cnr_despike_thresholds
    rng = np.random.default_rng(24)
    a = (rng.random((48, 64, 3), dtype=np.float32) * 0.3 + 0.1)
    # sprinkle a few green fireflies for the clamp to bite on
    for y, x in [(10, 12), (20, 40), (33, 25), (41, 55)]:
        a[y, x] = [0.03, 0.8, 0.03]
    d = cnr_despike_thresholds(80.0, 60.0)
    ref = reduce_color_noise_chroma(a, sigma=0.0, despike=d)[2:-2, 2:-2]
    cand = gpu.cnr_frame(Frame.from_cpu(a), sigma=0.0, despike=d).cpu()[2:-2, 2:-2]
    assert max_abs_err(ref, cand) <= 2e-2


@requires_gpu
def test_bloom_frame_matches_oracle():
    """GPU bloom matches the CPU version within a code value. Size divisible
    by 4 so the downsample blocks line up."""
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
    """GPU vignette matches the numpy version."""
    from core.effects import apply_vignette
    rng = np.random.default_rng(13)
    # Bright corners and a fractional feather, to catch the pow-of-negative
    # NaN (black corners).
    a = rng.random((44, 66, 3), dtype=np.float32) * 0.4 + 0.5
    for strength, color, feather in ((0.5, 0.05, 1.0), (0.8, 0.12, 1.6), (0.1, 0.05, 0.4)):
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


# --- per-render arena (reuse + dirty-arena safety) --------------------------

def _smooth_lut(n=17):
    axis = np.linspace(0.0, 1.0, n, dtype=np.float32)
    r, g, b = np.meshgrid(axis, axis, axis, indexing='ij')
    return np.stack([
        np.clip(r ** 1.1 * 0.95 + 0.03 * g, 0, 1),
        np.clip(g ** 0.95,                  0, 1),
        np.clip(b ** 1.05 * 0.97 + 0.02 * r, 0, 1),
    ], axis=-1).astype(np.float32)


@requires_gpu
def test_arena_reuses_textures_across_renders():
    """Within a render, allocations get distinct textures; the next render
    reuses them."""
    shape = (32, 48, 3)

    gpu.begin_render()
    try:
        a0 = gpu._create_tex(shape)
        a1 = gpu._create_tex(shape)
        assert a0 is not a1            # two live Frames never share a texture
    finally:
        gpu.end_render()

    gpu.begin_render()
    try:
        b0 = gpu._create_tex(shape)
        b1 = gpu._create_tex(shape)
        assert b0 is a0 and b1 is a1   # next render reuses the same textures
    finally:
        gpu.end_render()

    # Outside a scope, allocation is fresh.
    assert gpu._create_tex(shape) is not a0


@requires_gpu
def test_dirty_arena_parity():
    """Stale data left in pooled textures by another render doesn't change
    the result."""
    from core.kernels import run_resident
    gpu.upload_lut(_smooth_lut())

    rng = np.random.default_rng(99)
    # divisible by 4 so bloom's area-downsample blocks line up
    img_a = (rng.random((64, 96, 3), dtype=np.float32) * 0.4).astype(np.float32)
    img_a[20:32, 30:50, :] += 3.0                      # highlights drive bloom
    grain = rng.random(img_a.shape, dtype=np.float32)

    chain_a = [
        lambda f: gpu.bloom_frame(f, 0.3, 0.55),       # pre-LUT, fills small pool
        gpu.encode_frame,
        gpu.lut_frame,
        lambda f: gpu.softness_frame(f, 2.0),
        lambda f: gpu.grain_frame(f, grain, 0.3, 0.2, 0.4),
        lambda f: gpu.sharpen_frame(f, 0.5, 2.0),
    ]

    clean = run_resident(img_a, chain_a)
    assert clean is not None

    # Dirty the pools with another chain on another image.
    img_b = (rng.random((64, 96, 3), dtype=np.float32) * 2.0).astype(np.float32)
    dirty_chain = [
        lambda f: gpu.cnr_frame(f, 4.0),
        lambda f: gpu.bloom_frame(f, 0.6, 0.2),
        gpu.encode_frame,
        gpu.lut_frame,
        lambda f: gpu.softness_frame(f, 5.0),
    ]
    assert run_resident(img_b, dirty_chain) is not None

    after_dirty = run_resident(img_a, chain_a)
    assert after_dirty is not None
    # Same input and math, so it must match exactly.
    assert max_abs_err(after_dirty, clean) <= 1e-6


# --- the parity gate itself --------------------------------------------------

def test_assert_parity_passes_on_equivalent(img):
    assert_parity(lambda a: a * 2.0, lambda a: a + a, img, tol=0.0, label="double")


def test_assert_parity_fails_above_tol(img):
    with pytest.raises(AssertionError):
        assert_parity(lambda a: a, lambda a: a + 1e-3, img, tol=1e-5, label="offset")
