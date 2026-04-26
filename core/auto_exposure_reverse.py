"""
Reverse the in-camera autoexposure of Flashback DNGs.

The Flashback ONE35 V2 locks ISO and aperture, so its autoexposure decision is
fully captured by the EXIF ExposureTime tag. To make a bright scene look bright
in the output (matching a fixed-scan reference like a disposable camera), we
undo the camera's gain by multiplying linear pixel values by T_ref / T_actual.
"""
from fractions import Fraction
from typing import Optional
import exifread


def extract_exposure_seconds(path: str) -> Optional[float]:
    """Read EXIF ExposureTime in seconds. Returns None if unavailable."""
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False, stop_tag='Image ExposureTime')
        tag = tags.get('Image ExposureTime')
        if tag is None:
            return None
        # exifread returns a Ratio with numerator/denominator
        val = tag.values[0]
        return float(Fraction(val.num, val.den))
    except Exception:
        return None


def compute_reverse_gain(exposure_s: Optional[float], t_ref_s: float) -> float:
    """
    Linear multiplier that undoes the camera's autoexposure relative to T_ref.

    gain > 1 → scene was brighter than reference (short exposure), boost it.
    gain < 1 → scene was darker than reference (long exposure), cut it.
    gain == 1 when ExposureTime == T_ref.
    """
    if not exposure_s or exposure_s <= 0 or t_ref_s <= 0:
        return 1.0
    return t_ref_s / exposure_s
