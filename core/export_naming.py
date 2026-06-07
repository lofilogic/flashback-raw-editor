"""Aesthetic, deterministic export filenames.

Both the export loop and the "already processed?" check derive the output name
the same way: a pure function of the source path (plus, for V1, its on-disk
sidecar). Keeping it stateless is what lets the processed-check recompute an
export's name in a later session without any saved mapping.

Naming:
  V2 DNGs   (``SN<serial>_<frame>``)  -> ``FBV2_<frame4>``  (e.g. FBV2_0042)
  other V2 files                      -> the original stem, unchanged
  V1 negatives                        -> ``FBV1_<roll4>_<frame4>``
                                         (e.g. FBV1_3f9c_0007)

For V1 the roll token is a short hash of the roll identifier (the negative's
parent folder, which is the import/zip name). The camera's roll id carries no
human-meaningful info — it exists only to be unique — so a hash of it is just
as good a key and needs no registry. ``frame`` is the negative's stem (the
camera writes them as sequential integers 0, 1, 2, …).
"""

import hashlib
import re
from pathlib import Path

from .v1_negative import is_v1_negative

# Camera-issued V2 filename shape: SN<serial>_<frame> (see editor's
# _CAMERA_DNG_PATTERN). We keep only the frame; the serial is noise.
_V2_FRAME_RE = re.compile(r'^SN\d+_(\d+)$', re.IGNORECASE)

# 4 hex chars = 16 bits of roll key. Birthday-bound: a 50% collision needs
# ~300 distinct rolls, so it's comfortable for personal libraries. Widen if a
# user ever shoots thousands of rolls into one output folder.
_ROLL_HASH_LEN = 4


def _roll_token(roll_id: str) -> str:
    return hashlib.blake2s(roll_id.encode('utf-8'),
                           digest_size=8).hexdigest()[:_ROLL_HASH_LEN]


def _frame_token(stem: str) -> str:
    """Zero-pad a numeric stem to 4 digits; pass non-numeric stems through."""
    try:
        return f"{int(stem):04d}"
    except ValueError:
        return stem


def export_basename(file_path) -> str:
    """Return the suffix-less, extension-less export name for a source file."""
    p = Path(file_path)
    if is_v1_negative(str(p)):
        return f"FBV1_{_roll_token(p.parent.name)}_{_frame_token(p.stem)}"
    m = _V2_FRAME_RE.match(p.stem)
    if m:
        return f"FBV2_{int(m.group(1)):04d}"
    return p.stem
