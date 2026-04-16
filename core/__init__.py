"""
Core package initialization.

Applies the NumPy 2.0 compatibility shim and configures the Numba cache
before any submodule in this package is imported. Both steps must happen
before colour-science and numba are imported anywhere.
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

# =============================================================================
# Numba Cache Configuration
# =============================================================================
# Must run before numba is imported anywhere in this package.
# Without a persistent cache, every startup takes 30-90s for recompilation.

def _get_cache_base_dir():
    """Return the platform-appropriate cache directory for the app."""
    import platform
    system = platform.system()
    if system == 'Darwin':
        return os.path.expanduser('~/Library/Caches/FlashbackOne35')
    elif system == 'Windows':
        local_appdata = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        return os.path.join(local_appdata, 'FlashbackOne35', 'Cache')
    else:  # Linux / other
        return os.path.expanduser('~/.cache/FlashbackOne35')

def _setup_numba_cache():
    """Setup persistent Numba cache for both dev and PyInstaller builds."""
    import llvmlite
    cache_dir = _get_cache_base_dir()

    if hasattr(sys, '_MEIPASS'):
        # PyInstaller: cache must be outside the temp _MEIxxxx folder
        version_key = f"numba_{llvmlite.__version__}_0"  # bump trailing int if kernels change
        cache_dir = os.path.join(cache_dir, version_key)
        print(f"[Numba] PyInstaller build detected, using cache: {cache_dir}")
    else:
        print(f"[Numba] Development mode, using cache: {cache_dir}")

    try:
        os.makedirs(cache_dir, exist_ok=True)
        test_file = os.path.join(cache_dir, ".write_test")
        with open(test_file, 'w') as f:
            f.write("ok")
        os.remove(test_file)
        os.environ['NUMBA_CACHE_DIR'] = cache_dir
        print(f"[Numba] Cache directory ready: {cache_dir}")

        # Seed the user's cache from the bundled pre-compiled cache on first launch.
        if hasattr(sys, '_MEIPASS'):
            bundled = os.path.join(sys._MEIPASS, '_numba_cache', version_key)
            cache_empty = not any(
                f.endswith('.nbc') for f in os.listdir(cache_dir)
            )
            if cache_empty and os.path.isdir(bundled):
                import shutil
                shutil.copytree(bundled, cache_dir, dirs_exist_ok=True)
                print(f"[Numba] Seeded cache from bundle — first launch will be fast.")

    except Exception as e:
        print(f"[Numba] Warning: Cannot use cache ({e}), disabling...")
        os.environ['NUMBA_DISABLE_JIT'] = '1'

_setup_numba_cache()

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
