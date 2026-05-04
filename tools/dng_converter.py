"""Cross-platform wrapper around Adobe DNG Converter (macOS + Windows).

Locates the converter executable, converts any raw file (or compressed DNG)
into a linear + uncompressed DNG that the HALD-inject pipeline can ingest
cleanly across all sensor types (Bayer, X-Trans, multi-channel, …).

If the converter isn't installed (or the user is on Linux), DNGs that are
*already* LinearRaw + uncompressed can be passed through unchanged. Anything
else requires Adobe DNG Converter — there's no FOSS substitute for the
camera-profile work it bakes in.

A user-supplied path can be persisted to skip auto-detection on future runs.
"""
from __future__ import annotations
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import tifffile

CONFIG_FILE = Path.home() / '.flashback_colormatch_dng_path'

# Canonical raw extensions Adobe DNG Converter handles. Listed for the GUI's
# file-picker filter; the converter itself sniffs by content too.
RAW_EXTENSIONS = [
    '.dng', '.arw', '.raf', '.nef', '.nrw', '.cr2', '.cr3', '.crw',
    '.orf', '.pef', '.rw2', '.rwl', '.iiq', '.mos', '.x3f',
    '.3fr', '.fff', '.dcr', '.k25', '.kdc', '.srw', '.srf', '.sr2',
    '.erf', '.mef', '.mrw', '.raw',
]


def _builtin_candidate_paths():
    """Common Adobe DNG Converter install locations per OS."""
    sysname = platform.system()
    if sysname == 'Darwin':
        return [
            '/Applications/Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter',
            str(Path.home() / 'Applications' / 'Adobe DNG Converter.app'
                / 'Contents' / 'MacOS' / 'Adobe DNG Converter'),
        ]
    if sysname == 'Windows':
        return [
            r'C:\Program Files\Adobe\Adobe DNG Converter\Adobe DNG Converter.exe',
            r'C:\Program Files (x86)\Adobe\Adobe DNG Converter\Adobe DNG Converter.exe',
        ]
    return []


def get_saved_path() -> str | None:
    """Return the user-saved converter path if present and still valid."""
    try:
        if CONFIG_FILE.exists():
            p = CONFIG_FILE.read_text().strip()
            if p and os.path.isfile(p):
                return p
    except Exception:
        pass
    return None


def set_saved_path(path: str | None) -> None:
    """Persist a user-specified converter path (or clear it if path is None)."""
    if path:
        CONFIG_FILE.write_text(str(path))
    elif CONFIG_FILE.exists():
        CONFIG_FILE.unlink()


def find_dng_converter() -> str | None:
    """Locate the Adobe DNG Converter executable. Tries: saved path → built-in
    OS-specific install paths → macOS Spotlight. Returns path or None."""
    p = get_saved_path()
    if p:
        return p
    for c in _builtin_candidate_paths():
        if os.path.isfile(c):
            return c
    # macOS Spotlight fallback (Adobe sometimes installs to non-standard dirs)
    if platform.system() == 'Darwin':
        try:
            r = subprocess.run(
                ['mdfind', 'kMDItemCFBundleIdentifier == "com.adobe.AdobeDNGConverter"'],
                capture_output=True, text=True, timeout=5,
            )
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.endswith('.app'):
                    exe = Path(line) / 'Contents' / 'MacOS' / 'Adobe DNG Converter'
                    if exe.is_file():
                        return str(exe)
        except Exception:
            pass
    return None


def is_linear_uncompressed_dng(path: str) -> bool:
    """True if the DNG is already LinearRaw (3-ch demosaiced) and uncompressed
    — i.e. ready for HALD inject without re-running through DNG Converter.
    """
    try:
        with tifffile.TiffFile(path) as tif:
            def _check(page):
                if 262 not in page.tags:
                    return None
                photometric = int(page.tags[262].value)
                if photometric not in (32803, 34892):
                    return None
                compression = int(page.tags[259].value) if 259 in page.tags else 1
                # LinearRaw (34892) + uncompressed (1) = ready
                return photometric == 34892 and compression == 1
            for page in tif.pages:
                r = _check(page)
                if r is not None:
                    return r
                # SubIFDs (LinearRaw is often nested)
                if hasattr(page, 'pages') and page.pages:
                    for sub in page.pages:
                        r = _check(sub)
                        if r is not None:
                            return r
    except Exception:
        return False
    return False


def convert_to_linear_dng(input_path: str, output_path: str,
                          compatibility: str = '-p2', log=print) -> str:
    """Convert a raw or DNG file to a linear + uncompressed DNG at output_path.

    Always uses linear (-l, demosaiced) so X-Trans/multi-channel sensors land
    as LinearRaw 3-ch — what our HALD-inject pipeline expects.
    """
    converter = find_dng_converter()
    if converter is None:
        raise RuntimeError(
            'Adobe DNG Converter not found.\n\n'
            'Free download: https://helpx.adobe.com/camera-raw/digital-negative.html\n\n'
            'Default install locations:\n'
            '  macOS:   /Applications/Adobe DNG Converter.app\n'
            '  Windows: C:\\Program Files\\Adobe\\Adobe DNG Converter\\\n'
            '\n'
            'If installed elsewhere, point the tool at the executable manually.'
        )

    input_path = str(input_path)
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    log(f'Adobe DNG Converter → linear, uncompressed DNG')

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            converter,
            '-l',              # linear (demosaiced)
            '-u',              # uncompressed
            compatibility,     # output compatibility
            '-mp',             # multi-process
            '-d', tmp,         # output directory
            input_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.stdout.strip():
            log('  ' + proc.stdout.strip())
        if proc.returncode != 0:
            if proc.stderr.strip():
                log('  ' + proc.stderr.strip())
            raise RuntimeError(f'Adobe DNG Converter exited with code {proc.returncode}')

        dngs = sorted(Path(tmp).glob('*.dng')) + sorted(Path(tmp).glob('*.DNG'))
        if not dngs:
            raise RuntimeError('Adobe DNG Converter produced no .dng output')
        if len(dngs) > 1:
            log(f'  warning: multiple DNGs in output, using {dngs[0].name}')

        shutil.copy2(dngs[0], output_path)
        log(f'  → {output_path}')
    return output_path


def normalise_input_to_linear_dng(input_path: str, output_path: str, log=print) -> str:
    """Smart entry point: skip conversion if input is already a clean linear
    DNG; otherwise call Adobe DNG Converter.

    Returns the path to a linear + uncompressed DNG (== output_path on convert,
    == input_path on skip). The HALD inject can run on either.
    """
    if input_path.lower().endswith('.dng') and is_linear_uncompressed_dng(input_path):
        log(f'Input is already LinearRaw + uncompressed → skipping conversion')
        return input_path
    return convert_to_linear_dng(input_path, output_path, log=log)
