"""Shared parity gate for the GPU-resident migration.

Every stage we move from CPU to GPU keeps its numpy/cv2 implementation as the
reference "oracle"; the GPU/WGSL implementation must reproduce it within a tight
tolerance. ``assert_parity`` is the single check used by every such stage test,
so visual parity is enforced uniformly instead of being re-invented per stage.
"""
import numpy as np


def max_abs_err(a, b) -> float:
    """Max absolute difference between two arrays (compared in float64)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} != {b.shape}"
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def assert_parity(reference, candidate, *inputs, tol=1e-5, label="stage") -> float:
    """Assert candidate(*inputs) matches reference(*inputs) within ``tol``.

    Each implementation gets its own copy of every input, so an in-place
    candidate can't corrupt the oracle's input (or vice versa). Returns the
    measured max abs error so callers can log how much headroom they have.
    """
    ref = reference(*[np.array(x, copy=True) for x in inputs])
    cand = candidate(*[np.array(x, copy=True) for x in inputs])
    err = max_abs_err(ref, cand)
    assert err <= tol, f"{label}: max abs err {err:.3e} exceeds tol {tol:.3e}"
    return err
