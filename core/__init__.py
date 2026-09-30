"""
Image pipeline. Importing this package applies a NumPy 2 shim that
colour-science needs.
"""
import sys
import os
import numpy as np

# =============================================================================
# NumPy 2.0 Compatibility Shim
# =============================================================================
# colour-science still uses aliases NumPy 2 removed. Must run before it's
# imported.
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'bool_'):
    np.bool_ = bool
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128

import cv2 as _cv2
_cv2.setUseOptimized(True)
_cv2.setNumThreads(-1)

# =============================================================================
# Shared Utilities
# =============================================================================

def resource_path(relative_path):
    """Absolute path to a bundled file, from source or in a PyInstaller build."""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)
