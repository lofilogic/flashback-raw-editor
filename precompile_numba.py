"""
Pre-compile Numba functions before building.
Run this BEFORE pyinstaller.
"""
import sys
import os
import numpy as np

# Setup cache same as main app — must match _get_cache_base_dir() in core/__init__.py
import platform
system = platform.system()
if system == 'Darwin':
    cache_dir = os.path.expanduser('~/Library/Caches/FlashbackOne35')
elif system == 'Windows':
    local_appdata = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
    cache_dir = os.path.join(local_appdata, 'FlashbackOne35', 'Cache')
else:
    cache_dir = os.path.expanduser('~/.cache/FlashbackOne35')

os.makedirs(cache_dir, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = cache_dir

# Import and trigger compilation
from core.kernels import _trilinear_lut_numba

# Create dummy data to trigger compilation
print("Pre-compiling Numba functions...")
dummy_img = np.zeros((100, 100, 3), dtype=np.float32)
dummy_lut = np.zeros((33, 33, 33, 3), dtype=np.float32)

# This triggers compilation and saves to cache
_trilinear_lut_numba(dummy_img, dummy_lut, 33)

print(f"✓ Numba cache created at: {cache_dir}")
print("Ready to build with PyInstaller!")