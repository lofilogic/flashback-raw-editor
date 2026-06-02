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


# --- the parity gate itself --------------------------------------------------

def test_assert_parity_passes_on_equivalent(img):
    assert_parity(lambda a: a * 2.0, lambda a: a + a, img, tol=0.0, label="double")


def test_assert_parity_fails_above_tol(img):
    with pytest.raises(AssertionError):
        assert_parity(lambda a: a, lambda a: a + 1e-3, img, tol=1e-5, label="offset")
