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


# Singleton — one GPU device shared across the app
gpu = GPUPipeline()
HAS_GPU = _WGPU_AVAILABLE
