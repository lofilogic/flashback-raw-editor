"""
Core package initialization.

Applies the NumPy 2.0 compatibility shim before colour-science is imported,
then enables OpenCV SIMD optimizations.
"""
import sys
import os
import numpy as np

# =============================================================================
# NumPy 2.0 Compatibility Shim
# =============================================================================
# colour-science was written for NumPy 1.x and uses deprecated type aliases.
# Must be applied before importing colour-science or any submodule that does.
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'bool_'):
    np.bool_ = bool
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128

# Enable OpenCV SIMD/multi-thread dispatch
import cv2 as _cv2
_cv2.setUseOptimized(True)
_cv2.setNumThreads(-1)

# =============================================================================
# Shared Utilities
# =============================================================================

def resource_path(relative_path):
    """Get absolute path to resource — works for dev and PyInstaller builds."""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)
