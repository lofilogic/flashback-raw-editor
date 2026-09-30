"""GPU stages are tested against their numpy/cv2 versions with assert_parity."""
import numpy as np


def max_abs_err(a, b) -> float:
    """Max absolute difference between two arrays (compared in float64)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} != {b.shape}"
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def assert_parity(reference, candidate, *inputs, tol=1e-5, label="stage") -> float:
    """Assert candidate(*inputs) matches reference(*inputs) within ``tol``.
    Each gets its own copy of the inputs. Returns the max abs error."""
    ref = reference(*[np.array(x, copy=True) for x in inputs])
    cand = candidate(*[np.array(x, copy=True) for x in inputs])
    err = max_abs_err(ref, cand)
    assert err <= tol, f"{label}: max abs err {err:.3e} exceeds tol {tol:.3e}"
    return err
