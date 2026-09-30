"""Every compute pipeline builds on the current backend.

Naga has panicked on Vulkan ("Expression is not cached!") with shaders that
worked on D3D12 and Metal. A panic aborts the whole process, so a failure here
is loud. CI runs this on lavapipe, which goes through the same SPIR-V codegen.
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
    """Every pipeline in the table was created."""
    pipeline = getattr(gpu, pipe_attr, None)
    assert pipeline is not None, f"pipeline {pipe_attr} ({entry}) was not built"
