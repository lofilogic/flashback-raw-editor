"""V1 negative detection and its cache, which extract_negatives_from_zip
must clear."""
import json
import zipfile

import pytest

from core.v1_negative import extract_negatives_from_zip, is_v1_negative

W, H = 4, 3


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the cache between tests."""
    is_v1_negative.cache_clear()
    yield
    is_v1_negative.cache_clear()


def _write_negative(folder, stem='frame001', width=W, height=H):
    """Write a valid V1 pair: extensionless raw + same-named .json sidecar."""
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / stem
    raw.write_bytes(bytes(width * height))
    (folder / f'{stem}.json').write_text(json.dumps({'width': width, 'height': height}))
    return raw


def _write_roll_zip(zip_path, stem='frame002', width=W, height=H):
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr(f'{stem}.json', json.dumps({'width': width, 'height': height}))
        zf.writestr(stem, bytes(width * height))
    return zip_path


def test_detects_a_valid_negative(tmp_path):
    assert is_v1_negative(str(_write_negative(tmp_path))) is True


def test_rejects_a_raw_whose_size_contradicts_its_sidecar(tmp_path):
    """The raw must be exactly width*height bytes."""
    raw = _write_negative(tmp_path)
    raw.write_bytes(bytes(W * H + 1))
    is_v1_negative.cache_clear()
    assert is_v1_negative(str(raw)) is False


def test_rejects_a_path_with_no_sidecar(tmp_path):
    lonely = tmp_path / 'frame003'
    lonely.write_bytes(bytes(W * H))
    assert is_v1_negative(str(lonely)) is False


def test_repeated_calls_are_served_from_cache(tmp_path):
    raw = str(_write_negative(tmp_path))
    is_v1_negative(raw)
    is_v1_negative(raw)
    assert is_v1_negative.cache_info().hits == 1


def test_extracting_a_roll_invalidates_a_stale_negative_result(tmp_path):
    """A path cached as 'not a negative' before the roll is extracted there
    must be detected afterwards."""
    dest = tmp_path / 'out'
    probe = str(dest / 'frame002')
    assert is_v1_negative(probe) is False

    extract_negatives_from_zip(str(_write_roll_zip(tmp_path / 'roll.zip')), dest)

    assert is_v1_negative(probe) is True
