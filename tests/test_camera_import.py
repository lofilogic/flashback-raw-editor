"""Tests for the camera import path's DNG metadata.

ProfileName is an app-wide preference: it decides which profile Camera Raw /
Lightroom bind the file to. Import used to hardcode the 'Flashback Standard'
default while Export > DNG passed the user's setting, so a name set in the app
only ever reached exported files — imported captures silently disagreed.
"""
import numpy as np
import pytest

from core.camera_import import export_camera_dng
from core.config import VibeConfig
from test_dng_export import _write_camera_style_dng, read_profile_name


class StubProcessor:
    """Minimal stand-in: export_camera_dng only needs load_image() and .vibe."""

    def __init__(self, profile_name=None):
        self.vibe = VibeConfig()
        if profile_name is not None:
            self.vibe.dng_profile_name = profile_name
        self.loaded = []

    def load_image(self, path):
        self.loaded.append(path)
        return np.zeros((40, 60, 3), dtype=np.uint8)


@pytest.fixture
def source_dng(tmp_path):
    return _write_camera_style_dng(tmp_path / 'camera.dng')


def test_import_defaults_to_flashback_standard(source_dng, tmp_path):
    """The shipped default is unchanged — we can't name another vendor's
    profile ourselves."""
    out = tmp_path / 'sub' / 'imported.dng'
    export_camera_dng(source_dng, out, StubProcessor())
    assert read_profile_name(out) == 'Flashback Standard'


def test_import_honours_user_profile_name(source_dng, tmp_path):
    """The actual regression: a user-set profile name must reach imported
    files, not just exported ones."""
    out = tmp_path / 'sub' / 'imported.dng'
    export_camera_dng(source_dng, out, StubProcessor('Adobe Standard'))
    assert read_profile_name(out) == 'Adobe Standard'


def test_import_and_export_agree_on_profile_name(source_dng, tmp_path):
    """Import and Export > DNG must write the same ProfileName for the same
    setting; that they diverged is what made this a bug."""
    from core.dng_export import export_dng

    name = 'Some Custom Profile'
    imported = tmp_path / 'sub' / 'imported.dng'
    export_camera_dng(source_dng, imported, StubProcessor(name))

    exported = tmp_path / 'exported.dng'
    export_dng(str(imported), str(exported), np.zeros((10, 12, 3), np.uint8), name)

    assert read_profile_name(imported) == read_profile_name(exported) == name


def test_import_returns_display_image(source_dng, tmp_path):
    """Callers reuse the returned display image instead of re-reading."""
    proc = StubProcessor()
    out = tmp_path / 'sub' / 'imported.dng'
    result = export_camera_dng(source_dng, out, proc)
    assert result is not None
    assert proc.loaded == [str(source_dng)]
