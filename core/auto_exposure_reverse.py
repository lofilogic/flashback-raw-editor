"""
Undo the ONE35 V2's autoexposure.

ISO and aperture are fixed, so ExposureTime is the whole AE decision.
Multiplying by T_ref / T makes bright scenes bright again, like a fixed
exposure camera would. Used for profiling.
"""
from typing import Optional


def compute_reverse_gain(exposure_s: Optional[float], t_ref_s: float) -> float:
    """T_ref / exposure, or 1 if unknown."""
    if not exposure_s or exposure_s <= 0 or t_ref_s <= 0:
        return 1.0
    return t_ref_s / exposure_s
