"""DNG export must find the raw strip in camera files (IFD0) and in our own
exports (SubIFD, with the thumbnail in IFD0). Re-exporting an export used to
package the thumbnail as raw data.
"""
import struct

import numpy as np
import pytest

from core.dng_export import (
    _BYTE, _SHORT, _LONG, _RATIONAL,
    _pack_ifd, _read_ifd, _find_raw_strip, _pack_rational_array, export_dng,
)

TAG_PROFILE_NAME = 50936

# Small stand-in for a camera file.
FAKE_W, FAKE_H = 8, 4
FAKE_RAW = bytes((i * 7 + 3) % 256 for i in range(FAKE_W * FAKE_H * 2))


def _write_camera_style_dng(path):
    """A single-IFD DNG with the CFA raw in IFD0 — how the camera writes them."""
    def build(raw_off):
        return [
            (254, _LONG, 1, struct.pack('<I', 0)),
            (256, _LONG, 1, struct.pack('<I', FAKE_W)),
            (257, _LONG, 1, struct.pack('<I', FAKE_H)),
            (258, _SHORT, 1, struct.pack('<H', 16)),
            (259, _SHORT, 1, struct.pack('<H', 1)),
            (262, _SHORT, 1, struct.pack('<H', 32803)),   # CFA
            (273, _LONG, 1, struct.pack('<I', raw_off)),
            (277, _SHORT, 1, struct.pack('<H', 1)),
            (278, _LONG, 1, struct.pack('<I', FAKE_H)),
            (279, _LONG, 1, struct.pack('<I', len(FAKE_RAW))),
            (282, _RATIONAL, 1, _pack_rational_array([300.0])),
            (283, _RATIONAL, 1, _pack_rational_array([300.0])),
            (296, _SHORT, 1, struct.pack('<H', 2)),
            (50706, _BYTE, 4, b'\x01\x04\x00\x00'),
        ]

    raw_off = 8 + len(_pack_ifd(build(0), 8))
    with open(path, 'wb') as f:
        f.write(b'II\x2a\x00\x08\x00\x00\x00')
        f.write(_pack_ifd(build(raw_off), 8))
        f.write(FAKE_RAW)
    return path


def _read_strip(path):
    with open(path, 'rb') as f:
        found = _find_raw_strip(f)
        assert found is not None, f"no CFA strip located in {path}"
        off, length = found
        f.seek(off)
        return f.read(length)


def read_profile_name(path):
    """ProfileName (tag 50936) from IFD0. exifread doesn't decode this DNG tag,
    so walk to it directly."""
    with open(path, 'rb') as f:
        header = f.read(8)
        bo = '<' if header[:2] == b'II' else '>'
        (ifd0_off,) = struct.unpack(bo + 'I', header[4:8])
        ifd0 = _read_ifd(f, ifd0_off, bo)
        _type, count, payload = ifd0[TAG_PROFILE_NAME]
        if count > 4:
            (off,) = struct.unpack(bo + 'I', payload)
            f.seek(off)
            payload = f.read(count)
        return payload[:count].rstrip(b'\x00').decode()


@pytest.fixture
def source_dng(tmp_path):
    return _write_camera_style_dng(tmp_path / 'camera.dng')


def test_find_raw_strip_reads_ifd0_on_camera_files(source_dng):
    """Camera originals keep the raw in IFD0."""
    assert _read_strip(source_dng) == FAKE_RAW


def test_export_preserves_raw_strip(source_dng, tmp_path):
    out = tmp_path / 'pass1.dng'
    assert export_dng(str(source_dng), str(out), np.zeros((4, 6, 3), np.uint8))
    assert _read_strip(out) == FAKE_RAW


def test_reexport_preserves_raw_strip(source_dng, tmp_path):
    """Re-exporting an export keeps the raw, not the thumbnail."""
    p1, p2 = tmp_path / 'pass1.dng', tmp_path / 'pass2.dng'
    thumb = np.zeros((4, 6, 3), np.uint8)
    assert export_dng(str(source_dng), str(p1), thumb)
    assert export_dng(str(p1), str(p2), thumb)

    assert _read_strip(p2) == FAKE_RAW
    # A second pass should produce identical bytes.
    assert p2.read_bytes() == p1.read_bytes()


def test_find_raw_strip_skips_rgb_ifd0(source_dng, tmp_path):
    """The IFD0 thumbnail of our exports is never taken as the raw."""
    out = tmp_path / 'pass1.dng'
    thumb = np.zeros((4, 6, 3), np.uint8)
    assert export_dng(str(source_dng), str(out), thumb)

    with open(out, 'rb') as f:
        off, length = _find_raw_strip(f)
    assert length == len(FAKE_RAW)
    assert length != thumb.nbytes


def test_find_raw_strip_rejects_non_tiff(tmp_path):
    junk = tmp_path / 'junk.dng'
    junk.write_bytes(b'not a tiff at all')
    with open(junk, 'rb') as f:
        assert _find_raw_strip(f) is None
