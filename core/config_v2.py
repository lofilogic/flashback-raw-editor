"""
processor_v2 effects + LUT configuration.

DebugConfigV2 is now an alias for v1's DebugConfig: both pipelines share
the same set of toggles and parameters, and the existing advanced
settings panel transparently drives v2 as well as v1. The flag names
(enable_halation, enable_lut, etc.) are identical between v1 and v2;
processor_v2 just consumes them in its own pipeline order.

If we ever need v2-only parameters (different default values, new fields
not in v1), add them as attributes to DebugConfig in core/config.py
rather than re-introducing a parallel class here.
"""
from .config import DebugConfig as DebugConfigV2  # noqa: F401
