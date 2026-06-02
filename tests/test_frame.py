"""Phase 0 tests for the resident-by-default render image (core.gpu.Frame).

These prove the foundation the migration rests on:
  * a CPU-backed Frame is a transparent numpy wrapper (works with no GPU);
  * a GPU round-trip is bit-exact, so making a stage resident can never shift
    pixels on its own;
  * the shared assert_parity gate both passes on equal output and fails on a
    difference above tolerance.

GPU-touching tests skip cleanly where no usable device exists (e.g. CI).
"""
import numpy as np
import pytest

from core.gpu import Frame, gpu

from parity_utils import assert_parity, max_abs_err


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


# --- GPU round-trip: must be lossless ---------------------------------------

@requires_gpu
def test_frame_gpu_roundtrip_is_bit_exact(img):
    f = Frame.from_cpu(img)
    buf = f.gpu()
    assert f.on_gpu
    back = gpu._download(buf, img.shape)
    assert back.dtype == np.float32
    assert np.array_equal(back, np.ascontiguousarray(img, dtype=np.float32))


@requires_gpu
def test_frame_cpu_after_gpu_is_unchanged(img):
    f = Frame.from_cpu(img)
    f.gpu()  # force an upload
    assert np.array_equal(f.cpu(), np.ascontiguousarray(img, dtype=np.float32))


@requires_gpu
def test_frame_from_gpu_reads_back(img):
    buf = gpu._upload(img)
    f = Frame.from_gpu(buf, img.shape)
    assert f.on_gpu
    assert max_abs_err(f.cpu(), img) == 0.0


# --- the parity gate itself --------------------------------------------------

def test_assert_parity_passes_on_equivalent(img):
    assert_parity(lambda a: a * 2.0, lambda a: a + a, img, tol=0.0, label="double")


def test_assert_parity_fails_above_tol(img):
    with pytest.raises(AssertionError):
        assert_parity(lambda a: a, lambda a: a + 1e-3, img, tol=1e-5, label="offset")
