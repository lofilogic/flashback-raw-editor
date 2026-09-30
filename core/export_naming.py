"""Export filenames.

Derived only from the source path, so the "already exported?" check can
recompute them later without any saved state.

  V2 DNGs (SN<serial>_<frame>)  -> FBV2_<frame5>          FBV2_00042
  V1 negatives                  -> FBV1_<roll4>_<frame5>  FBV1_3f9c_00007
  anything else                 -> the original stem

The V1 roll token is a hash of the negative's folder name (the roll).
"""

import hashlib
import re
from pathlib import Path

from .v1_negative import is_v1_negative

# SN<serial>_<frame>, see editor._CAMERA_DNG_PATTERN
_V2_FRAME_RE = re.compile(r'^SN\d+_(\d+)$', re.IGNORECASE)

# 16 bits: 50% collision chance at ~300 rolls in one folder.
_ROLL_HASH_LEN = 4


def _roll_token(roll_id: str) -> str:
    return hashlib.blake2s(roll_id.encode('utf-8'),
                           digest_size=8).hexdigest()[:_ROLL_HASH_LEN]


def _frame_token(stem: str) -> str:
    """Zero-pad numeric stems to 5 digits."""
    try:
        return f"{int(stem):05d}"
    except ValueError:
        return stem


def export_basename(file_path) -> str:
    """Export name without vibe suffix or extension."""
    p = Path(file_path)
    if is_v1_negative(str(p)):
        return f"FBV1_{_roll_token(p.parent.name)}_{_frame_token(p.stem)}"
    m = _V2_FRAME_RE.match(p.stem)
    if m:
        return f"FBV2_{int(m.group(1)):05d}"
    # RAW+JPEG pairs share a stem; keep their exports apart.
    if p.suffix.lower() in ('.jpg', '.jpeg'):
        return f"{p.stem}_jpg"
    return p.stem
