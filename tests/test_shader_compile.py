"""Smoke test: every WGSL compute pipeline must compile on the active backend.

This guards against shader-translation regressions that only surface at pipeline
creation — most notably the Naga SPIR-V backend panic
("internal error: ... Expression is not cached!") that crashed the whole app on
the Vulkan backend (Windows default, and Linux/RADV on the Steam Deck) while the
D3D12 backend stayed fine. Such a panic is a Rust abort, not a Python exception,
so it takes down the entire test process — which is exactly the loud failure we
want in CI.

Runs wherever a GPU adapter exists. On CI that means a software Vulkan device
(Mesa lavapipe); Naga generates SPIR-V regardless of the underlying driver, so
lavapipe reproduces the same codegen path a real Vulkan GPU would take.
"""
import pytest

from core.gpu import GPUPipeline, gpu


def _gpu_available() -> bool:
    try:
        return bool(gpu._init())
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _gpu_available(), reason="no usable GPU device")

# Every (attribute name, entry point) the pipeline table promises to build.
_EXPECTED_PIPELINES = [
    (pipe_attr, entry)
    for _shader, _spec, _bgl, pipes in GPUPipeline._PIPELINE_TABLE
    for pipe_attr, entry in pipes
]


@requires_gpu
@pytest.mark.parametrize("pipe_attr,entry", _EXPECTED_PIPELINES,
                         ids=[f"{a}:{e}" for a, e in _EXPECTED_PIPELINES])
def test_pipeline_compiled(pipe_attr, entry):
    """Each declared compute pipeline exists on the initialised singleton.

    _init() already ran _build_pipelines() (the create_compute_pipeline calls
    that invoke Naga); if any shader failed to translate, the process would have
    aborted before we got here. This asserts the table and the shaders stayed in
    sync — every entry point produced a live pipeline object.
    """
    pipeline = getattr(gpu, pipe_attr, None)
    assert pipeline is not None, f"pipeline {pipe_attr} ({entry}) was not built"
