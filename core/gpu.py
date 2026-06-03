"""
GPU compute pipeline via wgpu (WebGPU native).

Provides a singleton GPUPipeline with methods for each accelerated operation.
Falls back gracefully if no GPU is available.

Usage:
    from .gpu import gpu, HAS_GPU
    if HAS_GPU:
        result = gpu.apply_lut(img, lut_table)
    else:
        result = cpu_fallback(img, lut_table)

All methods accept and return float32 numpy arrays with shape (H, W, 3).
The LUT buffer is persistent on the GPU — upload once per vibe change.
"""
from __future__ import annotations
import logging
import os
import struct
import numpy as np

try:
    import wgpu
    _WGPU_AVAILABLE = True
except ImportError:
    _WGPU_AVAILABLE = False

log = logging.getLogger(__name__)


def _read_shader(name: str) -> str:
    shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')
    with open(os.path.join(shader_dir, name), 'r') as f:
        return f.read()


class GPUPipeline:
    """Singleton GPU compute pipeline. Lazy-initialized on first use."""

    def __init__(self):
        self._device = None
        self._lut_pipeline = None
        self._lut_bg_layout = None
        self._acescct_pipeline_decode = None
        self._acescct_pipeline_encode = None
        self._acescct_bg_layout = None
        self._grain_pipeline = None
        self._grain_bg_layout = None
        self._screen_pipeline = None
        self._unsharp_pipeline = None
        self._blend_bg_layout = None
        self._gauss_pipeline_h = None
        self._gauss_pipeline_v = None
        self._gauss_bg_layout = None
        self._encode_tex_pipeline = None   # texture-resident ACEScct encode
        self._encode_tex_bg_layout = None
        self._lut_tex_pipeline = None      # texture-resident tetrahedral LUT
        self._lut_tex_bg_layout = None
        self._gauss_tex_pipeline_h = None  # texture-resident separable blur
        self._gauss_tex_pipeline_v = None
        self._gauss_tex_bg_layout = None
        self._hal_mask_pipeline = None     # texture-resident halation passes
        self._hal_mask_bg_layout = None
        self._hal_hi_pipeline = None
        self._hal_hi_bg_layout = None
        self._hal_combine_pipeline = None
        self._hal_combine_bg_layout = None
        self._unsharp_tex_pipeline = None  # texture-resident unsharp mask
        self._unsharp_tex_bg_layout = None
        self._grain_tex_pipeline = None    # texture-resident grain blend
        self._grain_tex_bg_layout = None
        self._ca_tex_pipeline = None       # texture-resident spectral CA
        self._ca_tex_bg_layout = None
        self._edge_soft_pipeline = None    # texture-resident edge (corner) softness
        self._edge_soft_bg_layout = None
        self._vignette_pipeline = None     # texture-resident vignette (pre-LUT)
        self._vignette_bg_layout = None
        self._bloom_dm_pipeline = None     # texture-resident bloom: downsample+mask
        self._bloom_dm_bg_layout = None
        self._bloom_ua_pipeline = None     # texture-resident bloom: upsample+add
        self._bloom_ua_bg_layout = None
        self._cnr_to_lab_pipeline = None   # texture-resident CNR (Lab + bilateral)
        self._cnr_to_acescg_pipeline = None
        self._cnr_bil_pipeline = None
        self._cnr_io_bg_layout = None
        self._cnr_bil_bg_layout = None
        self._colormat_pipeline = None     # buffer 3x3 colour transform (load-time)
        self._colormat_bg_layout = None
        self._lut_buf = None   # persistent GPU LUT buffer
        self._lut_size = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init(self):
        if self._device is not None:
            return True
        if not _WGPU_AVAILABLE:
            return False
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
            self._device = adapter.request_device_sync()
            self._build_pipelines()
            log.info("✓ GPU pipeline ready: %s", adapter.summary)
            return True
        except Exception as e:
            log.warning("⚠ GPU init failed (%s), using CPU fallbacks", e)
            self._device = None
            return False

    def _build_pipelines(self):
        dev = self._device

        # --- LUT pipeline ---
        lut_src = _read_shader('lut.wgsl')
        lut_mod = dev.create_shader_module(code=lut_src)
        self._lut_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
            {'binding': 1, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
            {'binding': 2, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.storage}},
            {'binding': 3, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        pl = dev.create_pipeline_layout(bind_group_layouts=[self._lut_bg_layout])
        self._lut_pipeline = dev.create_compute_pipeline(
            layout=pl, compute={'module': lut_mod, 'entry_point': 'main'})

        # --- ACEScct pipeline ---
        acescct_src = _read_shader('acescct.wgsl')
        acescct_mod = dev.create_shader_module(code=acescct_src)
        self._acescct_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
            {'binding': 1, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.storage}},
        ])
        pl2 = dev.create_pipeline_layout(bind_group_layouts=[self._acescct_bg_layout])
        self._acescct_pipeline_decode = dev.create_compute_pipeline(
            layout=pl2, compute={'module': acescct_mod, 'entry_point': 'main_decode'})
        self._acescct_pipeline_encode = dev.create_compute_pipeline(
            layout=pl2, compute={'module': acescct_mod, 'entry_point': 'main_encode'})

        # --- Grain pipeline ---
        grain_src = _read_shader('grain.wgsl')
        grain_mod = dev.create_shader_module(code=grain_src)
        self._grain_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
            {'binding': 1, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
            {'binding': 2, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.storage}},
            {'binding': 3, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        pl3 = dev.create_pipeline_layout(bind_group_layouts=[self._grain_bg_layout])
        self._grain_pipeline = dev.create_compute_pipeline(
            layout=pl3, compute={'module': grain_mod, 'entry_point': 'main'})

        # --- Screen blend + unsharp mask pipeline ---
        blend_src = _read_shader('blend.wgsl')
        blend_mod = dev.create_shader_module(code=blend_src)
        self._blend_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
            {'binding': 1, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
            {'binding': 2, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.storage}},
            {'binding': 3, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        pl4 = dev.create_pipeline_layout(bind_group_layouts=[self._blend_bg_layout])
        self._screen_pipeline = dev.create_compute_pipeline(
            layout=pl4, compute={'module': blend_mod, 'entry_point': 'main_screen'})
        self._unsharp_pipeline = dev.create_compute_pipeline(
            layout=pl4, compute={'module': blend_mod, 'entry_point': 'main_unsharp'})

        # --- Gaussian blur pipeline ---
        gauss_src = _read_shader('gaussian_blur.wgsl')
        gauss_mod = dev.create_shader_module(code=gauss_src)
        # Same layout as blend (img_in, kernel, img_out, uniforms) but bindings 0+1
        # are both read-only storage.
        self._gauss_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
            {'binding': 1, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
            {'binding': 2, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.storage}},
            {'binding': 3, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        pl5 = dev.create_pipeline_layout(bind_group_layouts=[self._gauss_bg_layout])
        self._gauss_pipeline_h = dev.create_compute_pipeline(
            layout=pl5, compute={'module': gauss_mod, 'entry_point': 'main_h'})
        self._gauss_pipeline_v = dev.create_compute_pipeline(
            layout=pl5, compute={'module': gauss_mod, 'entry_point': 'main_v'})

        # --- ACEScct encode (texture-resident) pipeline ---
        enc_src = _read_shader('encode_tex.wgsl')
        enc_mod = dev.create_shader_module(code=enc_src)
        self._encode_tex_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': wgpu.ShaderStage.COMPUTE,
             'texture': {'sample_type': wgpu.TextureSampleType.unfilterable_float,
                         'view_dimension': wgpu.TextureViewDimension.d2}},
            {'binding': 1, 'visibility': wgpu.ShaderStage.COMPUTE,
             'storage_texture': {'access': wgpu.StorageTextureAccess.write_only,
                                 'format': self._TEX_FORMAT,
                                 'view_dimension': wgpu.TextureViewDimension.d2}},
        ])
        pl6 = dev.create_pipeline_layout(bind_group_layouts=[self._encode_tex_bg_layout])
        self._encode_tex_pipeline = dev.create_compute_pipeline(
            layout=pl6, compute={'module': enc_mod, 'entry_point': 'main'})

        # --- LUT (texture-resident tetrahedral) pipeline ---
        lut_tex_src = _read_shader('lut_tex.wgsl')
        lut_tex_mod = dev.create_shader_module(code=lut_tex_src)
        self._lut_tex_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': wgpu.ShaderStage.COMPUTE,
             'texture': {'sample_type': wgpu.TextureSampleType.unfilterable_float,
                         'view_dimension': wgpu.TextureViewDimension.d2}},
            {'binding': 1, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
            {'binding': 2, 'visibility': wgpu.ShaderStage.COMPUTE,
             'storage_texture': {'access': wgpu.StorageTextureAccess.write_only,
                                 'format': self._TEX_FORMAT,
                                 'view_dimension': wgpu.TextureViewDimension.d2}},
            {'binding': 3, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        pl7 = dev.create_pipeline_layout(bind_group_layouts=[self._lut_tex_bg_layout])
        self._lut_tex_pipeline = dev.create_compute_pipeline(
            layout=pl7, compute={'module': lut_tex_mod, 'entry_point': 'main'})

        # --- Gaussian blur (texture-resident, separable) pipeline ---
        gauss_tex_src = _read_shader('gaussian_blur_tex.wgsl')
        gauss_tex_mod = dev.create_shader_module(code=gauss_tex_src)
        self._gauss_tex_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': wgpu.ShaderStage.COMPUTE,
             'texture': {'sample_type': wgpu.TextureSampleType.unfilterable_float,
                         'view_dimension': wgpu.TextureViewDimension.d2}},
            {'binding': 1, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
            {'binding': 2, 'visibility': wgpu.ShaderStage.COMPUTE,
             'storage_texture': {'access': wgpu.StorageTextureAccess.write_only,
                                 'format': self._TEX_FORMAT,
                                 'view_dimension': wgpu.TextureViewDimension.d2}},
        ])
        pl8 = dev.create_pipeline_layout(bind_group_layouts=[self._gauss_tex_bg_layout])
        self._gauss_tex_pipeline_h = dev.create_compute_pipeline(
            layout=pl8, compute={'module': gauss_tex_mod, 'entry_point': 'main_h'})
        self._gauss_tex_pipeline_v = dev.create_compute_pipeline(
            layout=pl8, compute={'module': gauss_tex_mod, 'entry_point': 'main_v'})

        # --- Halation (texture-resident) passes: mask, highlights, combine ---
        _tex = {'sample_type': wgpu.TextureSampleType.unfilterable_float,
                'view_dimension': wgpu.TextureViewDimension.d2}
        _store = {'access': wgpu.StorageTextureAccess.write_only,
                  'format': self._TEX_FORMAT, 'view_dimension': wgpu.TextureViewDimension.d2}
        _C = wgpu.ShaderStage.COMPUTE

        hal_mask_mod = dev.create_shader_module(code=_read_shader('halation_mask.wgsl'))
        self._hal_mask_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': _C, 'texture': _tex},
            {'binding': 1, 'visibility': _C, 'storage_texture': _store},
            {'binding': 2, 'visibility': _C, 'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        self._hal_mask_pipeline = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._hal_mask_bg_layout]),
            compute={'module': hal_mask_mod, 'entry_point': 'main'})

        hal_hi_mod = dev.create_shader_module(code=_read_shader('halation_highlights.wgsl'))
        self._hal_hi_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': _C, 'texture': _tex},
            {'binding': 1, 'visibility': _C, 'texture': _tex},
            {'binding': 2, 'visibility': _C, 'storage_texture': _store},
        ])
        self._hal_hi_pipeline = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._hal_hi_bg_layout]),
            compute={'module': hal_hi_mod, 'entry_point': 'main'})

        hal_comb_mod = dev.create_shader_module(code=_read_shader('halation_combine.wgsl'))
        self._hal_combine_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': _C, 'texture': _tex},
            {'binding': 1, 'visibility': _C, 'texture': _tex},
            {'binding': 2, 'visibility': _C, 'texture': _tex},
            {'binding': 3, 'visibility': _C, 'storage_texture': _store},
            {'binding': 4, 'visibility': _C, 'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        self._hal_combine_pipeline = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._hal_combine_bg_layout]),
            compute={'module': hal_comb_mod, 'entry_point': 'main'})

        # --- Unsharp mask (texture-resident) pipeline: img + (img-blur)*k ---
        unsharp_tex_mod = dev.create_shader_module(code=_read_shader('unsharp_tex.wgsl'))
        self._unsharp_tex_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': _C, 'texture': _tex},
            {'binding': 1, 'visibility': _C, 'texture': _tex},
            {'binding': 2, 'visibility': _C, 'storage_texture': _store},
            {'binding': 3, 'visibility': _C, 'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        self._unsharp_tex_pipeline = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._unsharp_tex_bg_layout]),
            compute={'module': unsharp_tex_mod, 'entry_point': 'main'})

        # --- Spectral chromatic aberration (texture-resident) pipeline ---
        ca_tex_mod = dev.create_shader_module(code=_read_shader('ca_tex.wgsl'))
        self._ca_tex_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': _C, 'texture': _tex},
            {'binding': 1, 'visibility': _C, 'storage_texture': _store},
            {'binding': 2, 'visibility': _C, 'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        self._ca_tex_pipeline = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._ca_tex_bg_layout]),
            compute={'module': ca_tex_mod, 'entry_point': 'main'})

        # --- Grain blend (texture-resident) pipeline ---
        grain_tex_mod = dev.create_shader_module(code=_read_shader('grain_tex.wgsl'))
        self._grain_tex_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': _C, 'texture': _tex},
            {'binding': 1, 'visibility': _C, 'texture': _tex},
            {'binding': 2, 'visibility': _C, 'storage_texture': _store},
            {'binding': 3, 'visibility': _C, 'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        self._grain_tex_pipeline = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._grain_tex_bg_layout]),
            compute={'module': grain_tex_mod, 'entry_point': 'main'})

        # --- Edge (corner) softness (texture-resident) pipeline ---
        edge_soft_mod = dev.create_shader_module(code=_read_shader('edge_softness_tex.wgsl'))
        self._edge_soft_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': _C, 'texture': _tex},
            {'binding': 1, 'visibility': _C, 'texture': _tex},
            {'binding': 2, 'visibility': _C, 'storage_texture': _store},
            {'binding': 3, 'visibility': _C, 'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        self._edge_soft_pipeline = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._edge_soft_bg_layout]),
            compute={'module': edge_soft_mod, 'entry_point': 'main'})

        # --- Vignette (texture-resident, pre-LUT linear) pipeline ---
        vig_mod = dev.create_shader_module(code=_read_shader('vignette_tex.wgsl'))
        self._vignette_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': _C, 'texture': _tex},
            {'binding': 1, 'visibility': _C, 'storage_texture': _store},
            {'binding': 2, 'visibility': _C, 'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        self._vignette_pipeline = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._vignette_bg_layout]),
            compute={'module': vig_mod, 'entry_point': 'main'})

        # --- Bloom (texture-resident, pre-LUT linear): downmask + upadd ---
        bloom_dm_mod = dev.create_shader_module(code=_read_shader('bloom_downmask.wgsl'))
        self._bloom_dm_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': _C, 'texture': _tex},
            {'binding': 1, 'visibility': _C, 'storage_texture': _store},
            {'binding': 2, 'visibility': _C, 'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        self._bloom_dm_pipeline = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._bloom_dm_bg_layout]),
            compute={'module': bloom_dm_mod, 'entry_point': 'main'})

        bloom_ua_mod = dev.create_shader_module(code=_read_shader('bloom_upadd.wgsl'))
        self._bloom_ua_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': _C, 'texture': _tex},
            {'binding': 1, 'visibility': _C, 'texture': _tex},
            {'binding': 2, 'visibility': _C, 'storage_texture': _store},
            {'binding': 3, 'visibility': _C, 'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        self._bloom_ua_pipeline = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._bloom_ua_bg_layout]),
            compute={'module': bloom_ua_mod, 'entry_point': 'main'})

        # --- CNR (texture-resident, pre-LUT linear): Lab transforms + bilateral ---
        cnr_mod = dev.create_shader_module(code=_read_shader('cnr.wgsl'))
        self._cnr_io_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': _C, 'texture': _tex},
            {'binding': 1, 'visibility': _C, 'storage_texture': _store},
        ])
        cnr_io_pl = dev.create_pipeline_layout(bind_group_layouts=[self._cnr_io_bg_layout])
        self._cnr_to_lab_pipeline = dev.create_compute_pipeline(
            layout=cnr_io_pl, compute={'module': cnr_mod, 'entry_point': 'main_to_lab'})
        self._cnr_to_acescg_pipeline = dev.create_compute_pipeline(
            layout=cnr_io_pl, compute={'module': cnr_mod, 'entry_point': 'main_to_acescg'})
        self._cnr_bil_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': _C, 'texture': _tex},
            {'binding': 1, 'visibility': _C, 'storage_texture': _store},
            {'binding': 2, 'visibility': _C, 'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        self._cnr_bil_pipeline = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._cnr_bil_bg_layout]),
            compute={'module': cnr_mod, 'entry_point': 'main_bilateral'})

        # --- Color-space matrix transform (buffer, load-time raw->ACEScg) ---
        colormat_mod = dev.create_shader_module(code=_read_shader('color_matmul.wgsl'))
        self._colormat_bg_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': _C, 'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
            {'binding': 1, 'visibility': _C, 'buffer': {'type': wgpu.BufferBindingType.storage}},
            {'binding': 2, 'visibility': _C, 'buffer': {'type': wgpu.BufferBindingType.uniform}},
        ])
        self._colormat_pipeline = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._colormat_bg_layout]),
            compute={'module': colormat_mod, 'entry_point': 'main'})

    # ------------------------------------------------------------------
    # LUT management
    # ------------------------------------------------------------------

    def upload_lut(self, lut_table: np.ndarray):
        """Upload a LUT table to the GPU. Call once per vibe change.
        lut_table: float32 array of shape (N, N, N, 3), N=lut_size."""
        if not self._init():
            return
        flat = np.ascontiguousarray(lut_table.astype(np.float32)).ravel()
        self._lut_buf = self._device.create_buffer_with_data(
            data=flat.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._lut_size = lut_table.shape[0]

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _upload(self, arr: np.ndarray):
        data = np.ascontiguousarray(arr.astype(np.float32)).ravel()
        return self._device.create_buffer_with_data(
            data=data.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

    def _make_output(self, n_floats: int):
        return self._device.create_buffer(
            size=n_floats * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

    def _make_staging(self, n_floats: int):
        return self._device.create_buffer(
            size=n_floats * 4,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )

    def _readback(self, buf_out, buf_staging, n_floats: int, shape):
        enc = self._device.create_command_encoder()
        enc.copy_buffer_to_buffer(buf_out, 0, buf_staging, 0, n_floats * 4)
        self._device.queue.submit([enc.finish()])
        buf_staging.map_sync(mode=wgpu.MapMode.READ)
        result = np.frombuffer(buf_staging.read_mapped(), dtype=np.float32).copy()
        buf_staging.unmap()
        return result.reshape(shape)

    def _download(self, buf, shape) -> np.ndarray:
        """Read an arbitrary resident storage buffer back into a float32 array.

        Same readback as the per-op methods, but against a buffer the caller
        already owns (used by Frame.cpu()). Allocates its own staging buffer.
        """
        if not self._init():
            raise RuntimeError("GPU device unavailable")
        n = int(np.prod(shape))
        stg = self._make_staging(n)
        enc = self._device.create_command_encoder()
        enc.copy_buffer_to_buffer(buf, 0, stg, 0, n * 4)
        self._device.queue.submit([enc.finish()])
        stg.map_sync(mode=wgpu.MapMode.READ)
        result = np.frombuffer(stg.read_mapped(), dtype=np.float32).copy()
        stg.unmap()
        return result.reshape(shape)

    # ------------------------------------------------------------------
    # Texture-resident image transfer (rgba16float working space)
    # ------------------------------------------------------------------
    #
    # The resident render image is an rgba32float 2D texture. f32 (not f16) is
    # the right call *here*: this pipeline is CPU-bound, and half-float would
    # trade ~29 ms of CPU conversion per render (≈130 ms on the slow Windows
    # CPU) to save ~25 MB of GPU memory — sub-ms of bandwidth at 3 MP. f32 keeps
    # full precision (so the resident path matches the f32 CPU oracle to float
    # rounding), needs no pack/unpack passes, and costs only 2D texture
    # bandwidth we have to spare. RGB carries the image; alpha is 1.0. Most
    # stages use textureLoad (no filtering); the rare stage that wants bilinear
    # does it manually, so f32's non-filterability costs nothing.

    _TEX_FORMAT = 'rgba32float'

    def _create_tex(self, shape):
        h, w = shape[:2]
        return self._device.create_texture(
            size=(w, h, 1),
            format=self._TEX_FORMAT,
            usage=(wgpu.TextureUsage.TEXTURE_BINDING
                   | wgpu.TextureUsage.STORAGE_BINDING
                   | wgpu.TextureUsage.COPY_SRC
                   | wgpu.TextureUsage.COPY_DST),
        )

    def _upload_tex(self, arr: np.ndarray):
        """Upload an (H, W, 3) float32 array into a fresh rgba32float texture."""
        if not self._init():
            raise RuntimeError("GPU device unavailable")
        h, w = arr.shape[:2]
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = np.ascontiguousarray(arr[:, :, :3], dtype=np.float32)
        tex = self._create_tex(arr.shape)
        self._device.queue.write_texture(
            {'texture': tex},
            rgba.tobytes(),
            {'bytes_per_row': w * 4 * 4, 'rows_per_image': h},
            (w, h, 1),
        )
        return tex

    def _download_tex(self, tex, shape) -> np.ndarray:
        """Read an rgba32float texture back into an (H, W, 3) float32 array.

        copy_texture_to_buffer requires bytes_per_row to be a multiple of 256,
        so we copy into a row-padded buffer and strip the padding on the host.
        """
        if not self._init():
            raise RuntimeError("GPU device unavailable")
        h, w = shape[:2]
        unpadded = w * 16                     # rgba32float = 16 bytes / texel
        padded = ((unpadded + 255) // 256) * 256
        buf = self._device.create_buffer(
            size=padded * h,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        enc = self._device.create_command_encoder()
        enc.copy_texture_to_buffer(
            {'texture': tex},
            {'buffer': buf, 'bytes_per_row': padded, 'rows_per_image': h},
            (w, h, 1),
        )
        self._device.queue.submit([enc.finish()])
        buf.map_sync(mode=wgpu.MapMode.READ)
        raw = np.frombuffer(buf.read_mapped(), dtype=np.float32).copy()
        buf.unmap()
        rgba = raw.reshape(h, padded // 4)[:, : w * 4].reshape(h, w, 4)
        return np.ascontiguousarray(rgba[:, :, :3], dtype=np.float32)

    # ------------------------------------------------------------------
    # Resident stages (Frame -> Frame). These never upload or read back;
    # transfers happen only when a caller asks a Frame for the other side.
    # ------------------------------------------------------------------

    def encode_frame(self, frame: "Frame"):
        """ACEScct encode, texture-resident: Frame in -> Frame out, no readback.

        Resident twin of kernels.acescct_encode — same math (clamped to 1e-10),
        but consumes and produces a GPU texture so it chains with neighbouring
        GPU stages. Returns None if the GPU is unavailable (caller falls back).
        """
        if not self._init():
            return None
        h, w = frame.shape[:2]
        dst = self._create_tex(frame.shape)
        bg = self._device.create_bind_group(layout=self._encode_tex_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': dst.create_view()},
        ])
        enc = self._device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._encode_tex_pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8)
        cp.end()
        self._device.queue.submit([enc.finish()])
        return Frame.from_gpu(dst, frame.shape, self)

    def lut_frame(self, frame: "Frame"):
        """Tetrahedral 3D LUT, texture-resident: Frame in -> Frame out.

        Resident twin of apply_lut — same Sakamoto tetrahedral math against the
        persistently-uploaded LUT (see upload_lut). Returns None if the GPU is
        unavailable or no LUT is loaded (caller falls back).
        """
        if not self._init() or self._lut_buf is None:
            return None
        h, w = frame.shape[:2]
        dst = self._create_tex(frame.shape)
        uni = self._uniform(struct.pack('4I', self._lut_size, 0, 0, 0))
        bg = self._device.create_bind_group(layout=self._lut_tex_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': {'buffer': self._lut_buf, 'offset': 0, 'size': self._lut_buf.size}},
            {'binding': 2, 'resource': dst.create_view()},
            {'binding': 3, 'resource': {'buffer': uni, 'offset': 0, 'size': uni.size}},
        ])
        enc = self._device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._lut_tex_pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8)
        cp.end()
        self._device.queue.submit([enc.finish()])
        return Frame.from_gpu(dst, frame.shape, self)

    def blur_frame(self, frame: "Frame", sigma: float):
        """Separable Gaussian blur, texture-resident: Frame in -> Frame out.

        Matches gpu.gaussian_blur (clamp-to-edge, same normalised kernel) but
        keeps the image on the GPU. Both passes share one command encoder.
        """
        if not self._init():
            return None
        if sigma <= 0:
            return frame
        h, w = frame.shape[:2]
        kernel = self._gauss_kernel(sigma)
        kbuf = self._device.create_buffer_with_data(
            data=kernel.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        mid = self._create_tex(frame.shape)
        dst = self._create_tex(frame.shape)
        nx, ny = (w + 7) // 8, (h + 7) // 8
        enc = self._device.create_command_encoder()
        bg_h = self._device.create_bind_group(layout=self._gauss_tex_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': {'buffer': kbuf, 'offset': 0, 'size': kbuf.size}},
            {'binding': 2, 'resource': mid.create_view()},
        ])
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._gauss_tex_pipeline_h)
        cp.set_bind_group(0, bg_h)
        cp.dispatch_workgroups(nx, ny)
        cp.end()
        bg_v = self._device.create_bind_group(layout=self._gauss_tex_bg_layout, entries=[
            {'binding': 0, 'resource': mid.create_view()},
            {'binding': 1, 'resource': {'buffer': kbuf, 'offset': 0, 'size': kbuf.size}},
            {'binding': 2, 'resource': dst.create_view()},
        ])
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._gauss_tex_pipeline_v)
        cp.set_bind_group(0, bg_v)
        cp.dispatch_workgroups(nx, ny)
        cp.end()
        self._device.queue.submit([enc.finish()])
        return Frame.from_gpu(dst, frame.shape, self)

    def _run2d(self, pipeline, bind_group, w: int, h: int):
        """Submit a single 2D compute pass over a w*h image (8x8 workgroups)."""
        enc = self._device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bind_group)
        cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8)
        cp.end()
        self._device.queue.submit([enc.finish()])

    def _halation_mask(self, frame: "Frame", threshold: float, k: float):
        h, w = frame.shape[:2]
        dst = self._create_tex(frame.shape)
        uni = self._uniform(struct.pack('4f', threshold, k, 0.0, 0.0))
        bg = self._device.create_bind_group(layout=self._hal_mask_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': dst.create_view()},
            {'binding': 2, 'resource': {'buffer': uni, 'offset': 0, 'size': uni.size}},
        ])
        self._run2d(self._hal_mask_pipeline, bg, w, h)
        return Frame.from_gpu(dst, frame.shape, self)

    def _halation_highlights(self, img: "Frame", mask: "Frame"):
        h, w = img.shape[:2]
        dst = self._create_tex(img.shape)
        bg = self._device.create_bind_group(layout=self._hal_hi_bg_layout, entries=[
            {'binding': 0, 'resource': img.gpu().create_view()},
            {'binding': 1, 'resource': mask.gpu().create_view()},
            {'binding': 2, 'resource': dst.create_view()},
        ])
        self._run2d(self._hal_hi_pipeline, bg, w, h)
        return Frame.from_gpu(dst, img.shape, self)

    def _halation_combine(self, img: "Frame", glow1: "Frame", glow2: "Frame", strength: float):
        h, w = img.shape[:2]
        dst = self._create_tex(img.shape)
        uni = self._uniform(struct.pack('4f', strength, 0.0, 0.0, 0.0))
        bg = self._device.create_bind_group(layout=self._hal_combine_bg_layout, entries=[
            {'binding': 0, 'resource': img.gpu().create_view()},
            {'binding': 1, 'resource': glow1.gpu().create_view()},
            {'binding': 2, 'resource': glow2.gpu().create_view()},
            {'binding': 3, 'resource': dst.create_view()},
            {'binding': 4, 'resource': {'buffer': uni, 'offset': 0, 'size': uni.size}},
        ])
        self._run2d(self._hal_combine_pipeline, bg, w, h)
        return Frame.from_gpu(dst, img.shape, self)

    def halation_frame(self, frame: "Frame", threshold: float, blur_radius: float,
                       strength: float, k: float = 20.0):
        """Two-pass halation, fully texture-resident: Frame in -> Frame out.

        Mirrors effects.apply_halation (same thresholds, radii, channel weights
        and screen blend) but uploads once and reads back once instead of the
        ~9 CPU<->GPU round-trips the per-op path makes. Returns None if the GPU
        is unavailable (caller falls back to the numpy/buffer path).
        """
        if not self._init():
            return None

        def glow(thresh, br):
            mask = self._halation_mask(frame, thresh, k)
            mask = self.blur_frame(mask, 2.0)
            hi = self._halation_highlights(frame, mask)
            return self.blur_frame(hi, br)

        g1 = glow(threshold, blur_radius)
        g2 = glow(min(threshold + 0.15, 0.98), blur_radius * 3.0)
        return self._halation_combine(frame, g1, g2, strength)

    # ------------------------------------------------------------------
    # Post-LUT resident tail (display sRGB): softness, grain, sharpen
    # ------------------------------------------------------------------

    def softness_frame(self, frame: "Frame", sigma: float):
        """Film-softness Gaussian blur, texture-resident: Frame in -> Frame out.

        Resident twin of effects.apply_softness — it is exactly a separable
        Gaussian blur, so this just forwards to blur_frame (kept as a named
        stage so the post-LUT chain reads like the per-op pipeline).
        """
        return self.blur_frame(frame, sigma)

    def sharpen_frame(self, frame: "Frame", strength: float, radius: float):
        """Unsharp-mask sharpen, texture-resident: Frame in -> Frame out.

        Resident twin of effects.apply_sharpen: blur the image, then combine
        ``img + (img - blurred) * strength`` (same math as gpu.unsharp_mask),
        all on the GPU. The result is left unclamped, matching the per-op path;
        the host clips once after the final readback.
        """
        if not self._init():
            return None
        blurred = self.blur_frame(frame, radius)
        if blurred is None:
            return None
        h, w = frame.shape[:2]
        dst = self._create_tex(frame.shape)
        uni = self._uniform(struct.pack('4f', strength, 0.0, 0.0, 0.0))
        bg = self._device.create_bind_group(layout=self._unsharp_tex_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': blurred.gpu().create_view()},
            {'binding': 2, 'resource': dst.create_view()},
            {'binding': 3, 'resource': {'buffer': uni, 'offset': 0, 'size': uni.size}},
        ])
        self._run2d(self._unsharp_tex_pipeline, bg, w, h)
        return Frame.from_gpu(dst, frame.shape, self)

    def grain_frame(self, frame: "Frame", grain_layer: np.ndarray,
                    intensity: float, min_grain: float = 0.2,
                    highlight_bias: float = 0.0):
        """Film-grain blend, texture-resident: Frame in -> Frame out.

        Resident twin of gpu.grain_blend — same per-channel falloff math. The
        grain layer is generated on the CPU (random tiles, see processor) and
        uploaded as a texture here; the image itself stays GPU-resident, so this
        saves the image upload+readback of the per-op path. ``grain_layer`` must
        be an (H, W, 3) float32 array matching ``frame``'s spatial size.
        """
        if not self._init():
            return None
        h, w = frame.shape[:2]
        grain_tex = self._upload_tex(grain_layer)
        dst = self._create_tex(frame.shape)
        uni = self._uniform(struct.pack('4f', intensity, min_grain, highlight_bias, 0.0))
        bg = self._device.create_bind_group(layout=self._grain_tex_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': grain_tex.create_view()},
            {'binding': 2, 'resource': dst.create_view()},
            {'binding': 3, 'resource': {'buffer': uni, 'offset': 0, 'size': uni.size}},
        ])
        self._run2d(self._grain_tex_pipeline, bg, w, h)
        return Frame.from_gpu(dst, frame.shape, self)

    def ca_frame(self, frame: "Frame", scale: float, samples: int = 16):
        """Spectral chromatic aberration, texture-resident: Frame in -> Frame out.

        Resident twin of effects.apply_chromatic_aberration (the spectral model):
        integrates ``samples`` points across the spectrum, each radially
        displaced by its own magnification (red at 1.0, blue at 1.0 + ``scale``)
        and weighted by that band's RGB sensitivity. ``scale`` is the same value
        the per-op path takes (ca_pixels_to_scale). Returns None if the GPU is
        unavailable, or the input Frame unchanged when there's nothing to do.
        """
        if not self._init():
            return None
        if scale <= 0:
            return frame
        h, w = frame.shape[:2]
        dst = self._create_tex(frame.shape)
        uni = self._uniform(struct.pack('4f', float(scale), float(samples), 0.0, 0.0))
        bg = self._device.create_bind_group(layout=self._ca_tex_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': dst.create_view()},
            {'binding': 2, 'resource': {'buffer': uni, 'offset': 0, 'size': uni.size}},
        ])
        self._run2d(self._ca_tex_pipeline, bg, w, h)
        return Frame.from_gpu(dst, frame.shape, self)

    def edge_softness_frame(self, frame: "Frame", sigma: float, strength: float,
                            start: float):
        """Radial edge (corner) softness, texture-resident: Frame in -> Frame out.

        Blurs the frame once (blur_frame) and blends sharp->blurred with a weight
        that grows from ``start`` (as a fraction of the corner radius) out to the
        corners, scaled by ``strength`` (0..1). Resident twin of
        effects.apply_edge_softness. Returns the input unchanged when there is
        nothing to do, or None if the GPU is unavailable.
        """
        if not self._init():
            return None
        if strength <= 0 or sigma <= 0:
            return frame
        blurred = self.blur_frame(frame, sigma)
        if blurred is None:
            return None
        h, w = frame.shape[:2]
        dst = self._create_tex(frame.shape)
        uni = self._uniform(struct.pack('4f', float(strength), float(start), 0.0, 0.0))
        bg = self._device.create_bind_group(layout=self._edge_soft_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': blurred.gpu().create_view()},
            {'binding': 2, 'resource': dst.create_view()},
            {'binding': 3, 'resource': {'buffer': uni, 'offset': 0, 'size': uni.size}},
        ])
        self._run2d(self._edge_soft_pipeline, bg, w, h)
        return Frame.from_gpu(dst, frame.shape, self)

    # ------------------------------------------------------------------
    # Pre-LUT resident stages (linear ACEScg): vignette
    # ------------------------------------------------------------------

    def vignette_frame(self, frame: "Frame", strength: float, color_shift: float,
                       feather: float):
        """Cosine vignette with cool-edge tint, texture-resident: Frame in ->
        Frame out. Resident twin of effects.apply_vignette (linear ACEScg).
        Returns the input unchanged when strength<=0, or None if no GPU.
        """
        if not self._init():
            return None
        if strength <= 0:
            return frame
        h, w = frame.shape[:2]
        dst = self._create_tex(frame.shape)
        uni = self._uniform(struct.pack('4f', float(strength), float(color_shift),
                                        float(feather), 0.0))
        bg = self._device.create_bind_group(layout=self._vignette_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': dst.create_view()},
            {'binding': 2, 'resource': {'buffer': uni, 'offset': 0, 'size': uni.size}},
        ])
        self._run2d(self._vignette_pipeline, bg, w, h)
        return Frame.from_gpu(dst, frame.shape, self)

    def bloom_frame(self, frame: "Frame", strength: float, threshold: float):
        """Large-radius bloom, texture-resident: Frame in -> Frame out.

        Resident twin of effects.apply_bloom (the linear/additive render path):
        area-downsample 4x, mask highlights above ``threshold`` (ACEScct), blur
        the small layer, bilinear-upsample and add ``strength`` * layer back.
        Everything stays on the GPU. Returns the input unchanged when there's
        nothing to do, or None if the GPU is unavailable.
        """
        if not self._init():
            return None
        if strength <= 0:
            return frame
        h, w = frame.shape[:2]
        scale = 4
        bh, bw = max(4, h // scale), max(4, w // scale)
        small_shape = (bh, bw, 3)

        # Stage 1: area-downsample + highlight mask -> small bloom source.
        small = self._create_tex(small_shape)
        uni_dm = self._uniform(struct.pack('4f', float(threshold), 0.0, 0.0, 0.0))
        bg_dm = self._device.create_bind_group(layout=self._bloom_dm_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': small.create_view()},
            {'binding': 2, 'resource': {'buffer': uni_dm, 'offset': 0, 'size': uni_dm.size}},
        ])
        self._run2d(self._bloom_dm_pipeline, bg_dm, bw, bh)

        # Blur the small layer (same kernel as the per-op gaussian_blur).
        # Derive sigma from the long downsampled edge so the glow size is
        # orientation-invariant (rotation swaps bw/bh but not their max).
        sigma = max(2, max(bw, bh) // 5)
        blurred = self.blur_frame(Frame.from_gpu(small, small_shape, self), float(sigma))
        if blurred is None:
            return None

        # Stage 2: bilinear upsample + additive blend onto the full image.
        dst = self._create_tex(frame.shape)
        uni_ua = self._uniform(struct.pack('4f', float(strength), 0.0, 0.0, 0.0))
        bg_ua = self._device.create_bind_group(layout=self._bloom_ua_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': blurred.gpu().create_view()},
            {'binding': 2, 'resource': dst.create_view()},
            {'binding': 3, 'resource': {'buffer': uni_ua, 'offset': 0, 'size': uni_ua.size}},
        ])
        self._run2d(self._bloom_ua_pipeline, bg_ua, w, h)
        return Frame.from_gpu(dst, frame.shape, self)

    def _cnr_io(self, pipeline, src_tex, shape):
        """Run a CNR tex->tex pass (Lab transform) and return the dst texture."""
        h, w = shape[:2]
        dst = self._create_tex(shape)
        bg = self._device.create_bind_group(layout=self._cnr_io_bg_layout, entries=[
            {'binding': 0, 'resource': src_tex.create_view()},
            {'binding': 1, 'resource': dst.create_view()},
        ])
        self._run2d(pipeline, bg, w, h)
        return dst

    def cnr_frame(self, frame: "Frame", sigma: float):
        """Chroma noise reduction in Lab, texture-resident: Frame in -> Frame out.

        Resident twin of effects.reduce_color_noise_chroma: ACEScg -> Lab, an
        edge-preserving bilateral on a*/b* only (L* untouched, so luma is
        preserved), then Lab -> ACEScg. Window/sigmas mirror the cv2 call
        (d = max(5, int(sigma)*2+3) odd, range sigma 15). Returns the input
        unchanged when sigma<=0, or None if the GPU is unavailable.
        """
        if not self._init():
            return None
        if sigma <= 0:
            return frame
        d = max(5, int(sigma) * 2 + 3)
        if d % 2 == 0:
            d += 1
        radius = d // 2

        lab = self._cnr_io(self._cnr_to_lab_pipeline, frame.gpu(), frame.shape)
        filt = self._create_tex(frame.shape)
        uni = self._uniform(struct.pack('4f', float(sigma), 15.0, float(radius), 0.0))
        bg = self._device.create_bind_group(layout=self._cnr_bil_bg_layout, entries=[
            {'binding': 0, 'resource': lab.create_view()},
            {'binding': 1, 'resource': filt.create_view()},
            {'binding': 2, 'resource': {'buffer': uni, 'offset': 0, 'size': uni.size}},
        ])
        h, w = frame.shape[:2]
        self._run2d(self._cnr_bil_pipeline, bg, w, h)
        out = self._cnr_io(self._cnr_to_acescg_pipeline, filt, frame.shape)
        return Frame.from_gpu(out, frame.shape, self)

    def _uniform(self, data: bytes):
        # Uniform buffers must be multiples of 16 bytes
        padded = data + b'\x00' * (16 - len(data) % 16) if len(data) % 16 else data
        return self._device.create_buffer_with_data(
            data=padded,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

    def _dispatch(self, pipeline, bind_group, n_elements: int, workgroup_size: int = 256):
        n_wg = (n_elements + workgroup_size - 1) // workgroup_size
        # WebGPU limits each dispatch dimension to 65535; use 2D for large images.
        # Shaders reconstruct the linear index as: id.y * (65535 * workgroup_size) + id.x
        if n_wg <= 65535:
            nx, ny = n_wg, 1
        else:
            nx = 65535
            ny = (n_wg + nx - 1) // nx
        enc = self._device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bind_group)
        cp.dispatch_workgroups(nx, ny)
        cp.end()
        return enc

    # ------------------------------------------------------------------
    # Public GPU operations
    # ------------------------------------------------------------------

    def apply_lut(self, img: np.ndarray) -> np.ndarray:
        """Apply the currently-uploaded LUT via tetrahedral interpolation."""
        if not self._init() or self._lut_buf is None:
            return None
        h, w = img.shape[:2]
        n = h * w * 3
        flat = np.ascontiguousarray(img.astype(np.float32)).ravel()

        buf_in  = self._upload(flat)
        buf_out = self._make_output(n)
        buf_stg = self._make_staging(n)
        uni     = self._uniform(struct.pack('4I', w, h, self._lut_size, 0))

        bg = self._device.create_bind_group(layout=self._lut_bg_layout, entries=[
            {'binding': 0, 'resource': {'buffer': buf_in,       'offset': 0, 'size': buf_in.size}},
            {'binding': 1, 'resource': {'buffer': self._lut_buf,'offset': 0, 'size': self._lut_buf.size}},
            {'binding': 2, 'resource': {'buffer': buf_out,      'offset': 0, 'size': buf_out.size}},
            {'binding': 3, 'resource': {'buffer': uni,          'offset': 0, 'size': uni.size}},
        ])
        enc = self._dispatch(self._lut_pipeline, bg, h * w, workgroup_size=64)
        enc.copy_buffer_to_buffer(buf_out, 0, buf_stg, 0, n * 4)
        self._device.queue.submit([enc.finish()])

        buf_stg.map_sync(mode=wgpu.MapMode.READ)
        result = np.frombuffer(buf_stg.read_mapped(), dtype=np.float32).copy()
        buf_stg.unmap()
        return result.reshape(h, w, 3)

    def color_transform(self, img: np.ndarray, M: np.ndarray) -> np.ndarray | None:
        """Per-pixel 3x3 colour-space transform: out = (img.reshape(-1,3) @ M.T).

        Load-time helper for raw -> ACEScg. Returns None if the GPU is
        unavailable (caller falls back to numpy). M is a (3, 3) float array.
        """
        if not self._init():
            return None
        shape = img.shape
        flat = np.ascontiguousarray(img, dtype=np.float32).ravel()
        n = flat.size

        buf_in  = self._upload(flat)
        buf_out = self._make_output(n)
        buf_stg = self._make_staging(n)
        rows = np.zeros((3, 4), dtype=np.float32)
        rows[:, :3] = np.asarray(M, dtype=np.float32)
        uni = self._uniform(rows.tobytes())

        bg = self._device.create_bind_group(layout=self._colormat_bg_layout, entries=[
            {'binding': 0, 'resource': {'buffer': buf_in,  'offset': 0, 'size': buf_in.size}},
            {'binding': 1, 'resource': {'buffer': buf_out, 'offset': 0, 'size': buf_out.size}},
            {'binding': 2, 'resource': {'buffer': uni,     'offset': 0, 'size': uni.size}},
        ])
        enc = self._dispatch(self._colormat_pipeline, bg, n // 3, workgroup_size=256)
        enc.copy_buffer_to_buffer(buf_out, 0, buf_stg, 0, n * 4)
        self._device.queue.submit([enc.finish()])

        buf_stg.map_sync(mode=wgpu.MapMode.READ)
        result = np.frombuffer(buf_stg.read_mapped(), dtype=np.float32).copy()
        buf_stg.unmap()
        return result.reshape(shape)

    def acescct_decode(self, img: np.ndarray) -> np.ndarray:
        """ACEScct → linear. Operates in-place semantics (returns new array)."""
        if not self._init():
            return None
        orig_shape = img.shape
        flat = np.ascontiguousarray(img.astype(np.float32)).ravel()
        n = flat.size

        buf_in  = self._upload(flat)
        buf_out = self._make_output(n)
        buf_stg = self._make_staging(n)

        bg = self._device.create_bind_group(layout=self._acescct_bg_layout, entries=[
            {'binding': 0, 'resource': {'buffer': buf_in,  'offset': 0, 'size': buf_in.size}},
            {'binding': 1, 'resource': {'buffer': buf_out, 'offset': 0, 'size': buf_out.size}},
        ])
        enc = self._dispatch(self._acescct_pipeline_decode, bg, n)
        enc.copy_buffer_to_buffer(buf_out, 0, buf_stg, 0, n * 4)
        self._device.queue.submit([enc.finish()])

        buf_stg.map_sync(mode=wgpu.MapMode.READ)
        result = np.frombuffer(buf_stg.read_mapped(), dtype=np.float32).copy()
        buf_stg.unmap()
        return result.reshape(orig_shape)

    def acescct_encode(self, img: np.ndarray) -> np.ndarray:
        """Linear → ACEScct."""
        if not self._init():
            return None
        orig_shape = img.shape
        flat = np.ascontiguousarray(img.astype(np.float32)).ravel()
        n = flat.size

        buf_in  = self._upload(flat)
        buf_out = self._make_output(n)
        buf_stg = self._make_staging(n)

        bg = self._device.create_bind_group(layout=self._acescct_bg_layout, entries=[
            {'binding': 0, 'resource': {'buffer': buf_in,  'offset': 0, 'size': buf_in.size}},
            {'binding': 1, 'resource': {'buffer': buf_out, 'offset': 0, 'size': buf_out.size}},
        ])
        enc = self._dispatch(self._acescct_pipeline_encode, bg, n)
        enc.copy_buffer_to_buffer(buf_out, 0, buf_stg, 0, n * 4)
        self._device.queue.submit([enc.finish()])

        buf_stg.map_sync(mode=wgpu.MapMode.READ)
        result = np.frombuffer(buf_stg.read_mapped(), dtype=np.float32).copy()
        buf_stg.unmap()
        return result.reshape(orig_shape)

    def grain_blend(self, image: np.ndarray, grain: np.ndarray,
                    intensity: float, min_grain: float, highlight_bias: float) -> np.ndarray:
        """Grain blend with highlight bias."""
        if not self._init():
            return None
        orig_shape = image.shape
        flat_img   = np.ascontiguousarray(image.astype(np.float32)).ravel()
        flat_grain = np.ascontiguousarray(grain.astype(np.float32)).ravel()
        n = flat_img.size

        buf_img  = self._upload(flat_img)
        buf_grn  = self._upload(flat_grain)
        buf_out  = self._make_output(n)
        buf_stg  = self._make_staging(n)
        uni      = self._uniform(struct.pack('4f', intensity, min_grain, highlight_bias, 0.0))

        bg = self._device.create_bind_group(layout=self._grain_bg_layout, entries=[
            {'binding': 0, 'resource': {'buffer': buf_img, 'offset': 0, 'size': buf_img.size}},
            {'binding': 1, 'resource': {'buffer': buf_grn, 'offset': 0, 'size': buf_grn.size}},
            {'binding': 2, 'resource': {'buffer': buf_out, 'offset': 0, 'size': buf_out.size}},
            {'binding': 3, 'resource': {'buffer': uni,     'offset': 0, 'size': uni.size}},
        ])
        enc = self._dispatch(self._grain_pipeline, bg, n)
        enc.copy_buffer_to_buffer(buf_out, 0, buf_stg, 0, n * 4)
        self._device.queue.submit([enc.finish()])

        buf_stg.map_sync(mode=wgpu.MapMode.READ)
        result = np.frombuffer(buf_stg.read_mapped(), dtype=np.float32).copy()
        buf_stg.unmap()
        return result.reshape(orig_shape)

    def screen_blend(self, base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Screen blend: 1 - (1-base)*(1-blend)."""
        if not self._init():
            return None
        orig_shape = base.shape
        n = base.size

        buf_base  = self._upload(base.ravel())
        buf_blend = self._upload(blend.ravel())
        buf_out   = self._make_output(n)
        buf_stg   = self._make_staging(n)
        uni       = self._uniform(struct.pack('4f', 0.0, 0.0, 0.0, 0.0))

        bg = self._device.create_bind_group(layout=self._blend_bg_layout, entries=[
            {'binding': 0, 'resource': {'buffer': buf_base,  'offset': 0, 'size': buf_base.size}},
            {'binding': 1, 'resource': {'buffer': buf_blend, 'offset': 0, 'size': buf_blend.size}},
            {'binding': 2, 'resource': {'buffer': buf_out,   'offset': 0, 'size': buf_out.size}},
            {'binding': 3, 'resource': {'buffer': uni,       'offset': 0, 'size': uni.size}},
        ])
        enc = self._dispatch(self._screen_pipeline, bg, n)
        enc.copy_buffer_to_buffer(buf_out, 0, buf_stg, 0, n * 4)
        self._device.queue.submit([enc.finish()])

        buf_stg.map_sync(mode=wgpu.MapMode.READ)
        result = np.frombuffer(buf_stg.read_mapped(), dtype=np.float32).copy()
        buf_stg.unmap()
        return result.reshape(orig_shape)

    def unsharp_mask(self, image: np.ndarray, blurred: np.ndarray, strength: float) -> np.ndarray:
        """Unsharp mask: image + (image - blurred) * strength."""
        if not self._init():
            return None
        orig_shape = image.shape
        n = image.size

        buf_img  = self._upload(image.ravel())
        buf_blur = self._upload(blurred.ravel())
        buf_out  = self._make_output(n)
        buf_stg  = self._make_staging(n)
        uni      = self._uniform(struct.pack('4f', strength, 0.0, 0.0, 0.0))

        bg = self._device.create_bind_group(layout=self._blend_bg_layout, entries=[
            {'binding': 0, 'resource': {'buffer': buf_img,  'offset': 0, 'size': buf_img.size}},
            {'binding': 1, 'resource': {'buffer': buf_blur, 'offset': 0, 'size': buf_blur.size}},
            {'binding': 2, 'resource': {'buffer': buf_out,  'offset': 0, 'size': buf_out.size}},
            {'binding': 3, 'resource': {'buffer': uni,      'offset': 0, 'size': uni.size}},
        ])
        enc = self._dispatch(self._unsharp_pipeline, bg, n)
        enc.copy_buffer_to_buffer(buf_out, 0, buf_stg, 0, n * 4)
        self._device.queue.submit([enc.finish()])

        buf_stg.map_sync(mode=wgpu.MapMode.READ)
        result = np.frombuffer(buf_stg.read_mapped(), dtype=np.float32).copy()
        buf_stg.unmap()
        return result.reshape(orig_shape)

    # ------------------------------------------------------------------
    # Gaussian blur
    # ------------------------------------------------------------------

    @staticmethod
    def _gauss_kernel(sigma: float) -> np.ndarray:
        """Compute a normalised 1-D Gaussian kernel matching cv2's auto-size rule."""
        radius = max(1, int(round(sigma * 3)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
        return k / k.sum()

    def gaussian_blur(self, img: np.ndarray, sigma: float) -> np.ndarray | None:
        """Separable Gaussian blur on CPU or 3-channel image.

        Accepts (H, W) single-channel or (H, W, 3) three-channel float32 arrays.
        Returns the same shape. Returns None if GPU unavailable.
        """
        if not self._init():
            return None
        if sigma <= 0:
            return img.copy()

        single_ch = img.ndim == 2
        if single_ch:
            img3 = img[:, :, np.newaxis]   # treat as 1-channel
            num_ch = 1
        else:
            img3 = img
            num_ch = img.shape[2]

        h, w = img3.shape[:2]
        kernel = self._gauss_kernel(sigma)
        k_size = len(kernel)
        n = h * w * num_ch

        flat = np.ascontiguousarray(img3.astype(np.float32)).ravel()
        buf_in  = self._upload(flat)
        buf_mid = self._make_output(n)   # intermediate between H and V
        buf_out = self._make_output(n)
        buf_stg = self._make_staging(n)
        buf_k   = self._device.create_buffer_with_data(
            data=kernel.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        uni = self._uniform(struct.pack('4I', w, h, k_size, num_ch))

        # Horizontal pass: buf_in → buf_mid
        bg_h = self._device.create_bind_group(layout=self._gauss_bg_layout, entries=[
            {'binding': 0, 'resource': {'buffer': buf_in,  'offset': 0, 'size': buf_in.size}},
            {'binding': 1, 'resource': {'buffer': buf_k,   'offset': 0, 'size': buf_k.size}},
            {'binding': 2, 'resource': {'buffer': buf_mid, 'offset': 0, 'size': buf_mid.size}},
            {'binding': 3, 'resource': {'buffer': uni,     'offset': 0, 'size': uni.size}},
        ])
        enc1 = self._dispatch(self._gauss_pipeline_h, bg_h, h * w, workgroup_size=64)
        self._device.queue.submit([enc1.finish()])

        # Vertical pass: buf_mid → buf_out (separate submit ensures H finishes first)
        bg_v = self._device.create_bind_group(layout=self._gauss_bg_layout, entries=[
            {'binding': 0, 'resource': {'buffer': buf_mid, 'offset': 0, 'size': buf_mid.size}},
            {'binding': 1, 'resource': {'buffer': buf_k,   'offset': 0, 'size': buf_k.size}},
            {'binding': 2, 'resource': {'buffer': buf_out, 'offset': 0, 'size': buf_out.size}},
            {'binding': 3, 'resource': {'buffer': uni,     'offset': 0, 'size': uni.size}},
        ])
        enc2 = self._dispatch(self._gauss_pipeline_v, bg_v, h * w, workgroup_size=64)
        enc2.copy_buffer_to_buffer(buf_out, 0, buf_stg, 0, n * 4)
        self._device.queue.submit([enc2.finish()])

        buf_stg.map_sync(mode=wgpu.MapMode.READ)
        result = np.frombuffer(buf_stg.read_mapped(), dtype=np.float32).copy()
        buf_stg.unmap()

        result = result.reshape(h, w, num_ch)
        if single_ch:
            return result[:, :, 0]
        return result


class Frame:
    """Render-scoped image handle that lazily lives on the CPU or the GPU.

    Holds one image in (H, W, 3) layout. The CPU side is float32; the GPU side
    is an rgba16float 2D texture (the resident render representation). Making a
    stage GPU-resident changes only *where* the pixels live and trades exact
    f32 for perceptually-transparent half-float — never the visible result.

    The "truth" is on whichever side last wrote it. ``cpu()`` and ``gpu()``
    materialise the other side on demand and cache it, so a CPU<->GPU transfer
    happens only at a real backend boundary. When two GPU-resident stages run
    back to back the intermediate never round-trips through numpy — that is the
    entire point of the resident-by-default guideline: as stages are converted
    to take and return GPU-backed Frames, the transfers between converted
    neighbours drop out on their own, with no stage knowing about its
    neighbours.
    """

    __slots__ = ("_p", "_cpu", "_tex", "_shape")

    def __init__(self, pipeline: "GPUPipeline", *, cpu=None, tex=None, shape=None):
        if cpu is None and tex is None:
            raise ValueError("Frame needs either cpu data or a gpu texture")
        if tex is not None and cpu is None and shape is None:
            raise ValueError("Frame from a gpu texture needs an explicit shape")
        self._p = pipeline
        self._cpu = None if cpu is None else np.ascontiguousarray(cpu, dtype=np.float32)
        self._tex = tex
        self._shape = tuple(shape) if shape is not None else self._cpu.shape

    @classmethod
    def from_cpu(cls, arr, pipeline: "GPUPipeline" = None) -> "Frame":
        """Wrap a numpy array. No upload happens until .gpu() is first called."""
        return cls(pipeline or gpu, cpu=arr)

    @classmethod
    def from_gpu(cls, tex, shape, pipeline: "GPUPipeline" = None) -> "Frame":
        """Wrap a GPU texture. No readback happens until .cpu() is called."""
        return cls(pipeline or gpu, tex=tex, shape=shape)

    @property
    def shape(self):
        return self._shape

    @property
    def on_gpu(self) -> bool:
        """True if the image currently has a GPU-resident (texture) copy."""
        return self._tex is not None

    def cpu(self) -> np.ndarray:
        """Return the image as float32, reading back from the GPU only if needed."""
        if self._cpu is None:
            self._cpu = self._p._download_tex(self._tex, self._shape)
        return self._cpu

    def gpu(self):
        """Return the rgba16float texture, uploading from the CPU only if needed."""
        if self._tex is None:
            self._tex = self._p._upload_tex(self._cpu)
        return self._tex


# Singleton — one GPU device shared across the app
gpu = GPUPipeline()
HAS_GPU = _WGPU_AVAILABLE
