# -*- mode: python ; coding: utf-8 -*-
import sys
import os
import llvmlite
from pathlib import Path

block_cipher = None

# Inject version from git tag (set by CI as FLASHBACK_VERSION=v0.1.0-beta7).
# Strip leading 'v' so CFBundleShortVersionString gets '0.1.0-beta7'.
_app_version = os.environ.get('FLASHBACK_VERSION', 'dev').lstrip('v')
print(f"=== Building version: {_app_version} ===")
with open('_version.py', 'w') as _vf:
    _vf.write(f'__version__ = "{_app_version}"\n')

# Pre-compile Numba cache automatically before bundling.
import subprocess
print("=== Pre-compiling Numba cache ===")
try:
    subprocess.run([sys.executable, 'precompile_numba.py'], check=True)
    print("=== Numba cache ready ===\n")
except subprocess.CalledProcessError as e:
    print(f"WARNING: Numba precompile failed ({e}). App will JIT-compile on first launch.\n")

# Bundle the pre-compiled Numba cache so new users don't hit the JIT freeze.
_version_key = f"numba_{llvmlite.__version__}_0"
import platform as _platform
_system = _platform.system()
if _system == 'Darwin':
    _cache_base = os.path.expanduser('~/Library/Caches/FlashbackOne35')
elif _system == 'Windows':
    _cache_base = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'FlashbackOne35', 'Cache')
else:
    _cache_base = os.path.expanduser('~/.cache/FlashbackOne35')
_cache_src = os.path.join(_cache_base, _version_key)
_have_cache = os.path.isdir(_cache_src)
if not _have_cache:
    print(f"WARNING: Numba cache not found at {_cache_src}. App will JIT-compile on first launch.")

# Collect data files
added_files = [
    # (source_path, destination_folder_in_app)
    ('assets', 'assets'),
]
if _have_cache:
    added_files.append((_cache_src, f'_numba_cache/{_version_key}'))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        # App packages
        'core',
        'core.config',
        'core.kernels',
        'core.effects',
        'core.processor',
        'ui',
        'ui.widgets',
        'ui.debug_panel',
        'ui.editor',

        # Core processing libraries
        'exifread',
        'rawpy',
        'colour',
        'colour.models',
        'colour.models.rgb_to_rgb',
        'colour.RGB_COLOURSPACES',
        
        # Numba JIT compilation
        'numba',
        'numba.core.types',
        'llvmlite',
        'llvmlite.binding',
        
        # Image processing
        'scipy.ndimage',
        'scipy.interpolate',
        'cv2',
        'cv2.cv2',
        
        # Export libraries (CRITICAL - for JPEG export in builds)
        'PIL',
        'PIL.Image',
        'PIL.JpegImagePlugin',
        'PIL.ImageFile',
        'PIL._imaging',
        'imageio',
        'imageio.plugins',
        'imageio.plugins.pillow',
        'imageio.plugins.pillowmulti',
        
        # NumPy support
        'numpy.core._dtype_ctypes',
        
        # File path handling
        'pathlib',
    ],
    hookspath=[],
    hooksconfig={
        # Include PIL plugins
        'pil': {
            'include_files': ['PIL'],
        },
    },
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'sklearn',
        'pandas',
        'pytest',
        # Qt modules we don't use — saves ~500MB on PySide6 6.x
        'PySide6.QtWebEngine',
        'PySide6.QtWebEngineCore',
        'PySide6.QtWebEngineWidgets',
        'PySide6.QtWebChannel',
        'PySide6.QtWebSockets',
        'PySide6.QtMultimedia',
        'PySide6.QtMultimediaWidgets',
        'PySide6.Qt3DCore',
        'PySide6.Qt3DRender',
        'PySide6.Qt3DInput',
        'PySide6.Qt3DLogic',
        'PySide6.Qt3DAnimation',
        'PySide6.Qt3DExtras',
        'PySide6.QtCharts',
        'PySide6.QtDataVisualization',
        'PySide6.QtLocation',
        'PySide6.QtPositioning',
        'PySide6.QtBluetooth',
        'PySide6.QtNfc',
        'PySide6.QtRemoteObjects',
        'PySide6.QtScxml',
        'PySide6.QtSensors',
        'PySide6.QtSerialPort',
        'PySide6.QtSql',
        'PySide6.QtTest',
        # 'unittest',  # REMOVED - needed by numpy.testing
        # 'pydoc',     # REMOVED - might be needed by stdlib
        # 'email',     # REMOVED - needed by pkg_resources
        # 'http',      # REMOVED - needed by pkg_resources
        # 'xml',       # REMOVED - needed by pkg_resources/plistlib
        # 'xmlrpc',    # REMOVED - needed by pkg_resources
    ],  # Exclude heavy/unused deps to reduce build size
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False, 
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

_icon = None
if _system == 'Darwin':
    _icon = 'assets/icons/icon.icns'
elif _system == 'Windows':
    if os.path.exists('assets/icons/icon.ico'):
        _icon = 'assets/icons/icon.ico'

exe = EXE(
    pyz,
    a.scripts,
    [],  # Remove binaries/zipfiles/datas from EXE (moved to COLLECT)
    exclude_binaries=True,  # Important: this makes it onedir mode
    name='Flashback One35',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No terminal window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_icon,
)

# COLLECT creates directory with unpacked libraries (faster loading!)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Flashback One35',
)

if _system == 'Darwin':
    app = BUNDLE(
        coll,  # Bundle the COLLECT, not exe directly
        name='Flashback One35.app',
        icon='assets/icons/icon.icns',
        bundle_identifier='com.julian.flashback',
        info_plist={
            'CFBundleName': 'Flashback One35 v2',
            'CFBundleDisplayName': 'Flashback One35 v2',
            'CFBundleShortVersionString': _app_version,
            'CFBundleVersion': _app_version,
            'NSHighResolutionCapable': 'True',
            'LSBackgroundOnly': 'False',
            'LSUIElement': 'False',
        },
    )