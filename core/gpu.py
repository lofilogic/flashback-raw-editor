"""
wgpu compute pipeline.

``gpu`` is a lazily initialised GPUPipeline singleton. The *_frame methods
take and return Frames and keep pixels on the GPU between stages; the older
buffer methods take numpy arrays. Callers check HAS_GPU and fall back to the
numpy versions in kernels.py / effects.py.
"""
from __future__ import annotations
import logging
import os
import struct
import threading
import numpy as np

log = logging.getLogger(__name__)

try:
    import wgpu
    _WGPU_AVAILABLE = True
except ImportError:
    _WGPU_AVAILABLE = False
    # Usually a source checkout without requirements installed. Warn, since
    # slow renders otherwise look like a driver problem.
    log.warning("⚠ 'wgpu' is not installed — GPU acceleration is OFF and "
                "rendering will be slow. Install dependencies with: "
                "pip install -r requirements.txt")

# Backends can only be set once, before the wgpu instance exists (see _init).
_INSTANCE_EXTRAS_SET = False


def _read_shader(name: str) -> str:
    shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')
    with open(os.path.join(shader_dir, name), 'r') as f:
        return f.read()


def _destroy_gpu_resource(resource):
    """Free a texture/buffer now instead of at GC. Test doubles may not have
    destroy()."""
    try:
        resource.destroy()
    except Exception:
        pass


class _RenderArena:
    """Thread-local bump allocator for per-render textures and uniforms.

    Allocating a fresh texture per stage per frame was the largest
    interactive cost (driver work on the CPU). Within a render, each acquire
    returns a new slot; begin() rewinds the indices so the next render reuses
    the same resources.

    Every allocation in a render gets its own slot, so live Frames never share
    a texture. This relies on no Frame outliving its render. Reused textures
    contain stale data, which is fine because every stage overwrites its whole
    output (covered by the dirty-arena parity test).

    Thread-local because the preview and thumbnail workers render
    concurrently on the shared device.
    """

    def __init__(self):
        self._local = threading.local()

    def _state(self):
        s = self._local
        if not hasattr(s, "depth"):
            s.depth = 0
            s.active = False
            s.tex_pools = {}   # (h, w) -> list[texture]
            s.tex_idx = {}     # (h, w) -> next slot
            s.uni_pools = {}   # nbytes -> list[buffer]
            s.uni_idx = {}     # nbytes -> next slot
        return s

    @property
    def active(self) -> bool:
        return self._state().active

    def begin(self):
        """Open a render scope. Reentrant; the outermost call rewinds."""
        s = self._state()
        s.depth += 1
        if s.depth == 1:
            s.active = True
            s.tex_idx = {}
            s.uni_idx = {}

    def end(self):
        """Close a render scope. Call after the result has been read back."""
        s = self._state()
        s.depth = max(0, s.depth - 1)
        if s.depth == 0:
            s.active = False
            self._evict_unused(s)

    def _evict_unused(self, s):
        """Free pools the last render didn't use.

        Pools are keyed by size, so without this every image resolution seen
        keeps a full render's worth of textures alive. Scrubbing one image
        reuses the same keys and frees nothing."""
        for key in [k for k in s.tex_pools if k not in s.tex_idx]:
            for tex in s.tex_pools.pop(key):
                _destroy_gpu_resource(tex)
        for key in [k for k in s.uni_pools if k not in s.uni_idx]:
            for buf in s.uni_pools.pop(key):
                _destroy_gpu_resource(buf)

    def acquire_tex(self, shape, create_fn):
        s = self._state()
        key = tuple(shape[:2])
        pool = s.tex_pools.setdefault(key, [])
        i = s.tex_idx.get(key, 0)
        if i >= len(pool):
            pool.append(create_fn())
        s.tex_idx[key] = i + 1
        return pool[i]

    def acquire_uni(self, nbytes, create_fn):
        s = self._state()
        pool = s.uni_pools.setdefault(nbytes, [])
        i = s.uni_idx.get(nbytes, 0)
        if i >= len(pool):
            pool.append(create_fn())
        s.uni_idx[nbytes] = i + 1
        return pool[i]


class GPUPipeline:
    """Compute pipelines and resources. Initialised on first use."""

    def __init__(self):
        self._device = None
        self._lut_pipeline = None
        self._lut_bg_layout = None
        self._acescct_pipeline_decode = None
        self._acescct_pipeline_encode = None
        self._acescct_bg_layout = None
        self._grain_pipeline = None
        self._grain_bg_layout = None
        self._unsharp_pipeline = None
        self._blend_bg_layout = None
        self._gauss_pipeline_h = None
        self._gauss_pipeline_v = None
        self._gauss_bg_layout = None
        # texture pipelines
        self._encode_tex_pipeline = None
        self._encode_tex_bg_layout = None
        self._lut_tex_pipeline = None
        self._lut_tex_bg_layout = None
        self._gauss_tex_pipeline_h = None
        self._gauss_tex_pipeline_v = None
        self._gauss_tex_bg_layout = None
        self._hal_mask_pipeline = None
        self._hal_mask_bg_layout = None
        self._hal_hi_pipeline = None
        self._hal_hi_bg_layout = None
        self._hal_combine_pipeline = None
        self._hal_combine_bg_layout = None
        self._unsharp_tex_pipeline = None
        self._unsharp_tex_bg_layout = None
        self._grain_tex_pipeline = None
        self._grain_tex_bg_layout = None
        self._ca_tex_pipeline = None
        self._ca_tex_bg_layout = None
        self._edge_soft_pipeline = None
        self._edge_soft_bg_layout = None
        self._vignette_pipeline = None
        self._vignette_bg_layout = None
        self._bloom_dm_pipeline = None     # bloom downsample + mask
        self._bloom_dm_bg_layout = None
        self._bloom_ua_pipeline = None     # bloom upsample + add
        self._bloom_ua_bg_layout = None
        self._cnr_to_lab_pipeline = None
        self._cnr_to_acescg_pipeline = None
        self._cnr_bil_pipeline = None
        self._cnr_despike_pipeline = None
        self._cnr_io_bg_layout = None
        self._cnr_bil_bg_layout = None
        self._colormat_pipeline = None     # 3x3 colour transform on a buffer, at load
        self._colormat_bg_layout = None
        # Per-thread LUT: the render workers can each need a different one
        # (V1 negatives use a V1 variant).
        self._lut_local = threading.local()
        self._arena = _RenderArena()
        # Filled by _init, reported by status(). Missing drivers or a VM can
        # land on a software adapter, which otherwise looks like it works.
        self.adapter_info = {}
        self.adapter_summary = None
        self.is_software_adapter = False
        self.init_failed = False

    @property
    def _lut_buf(self):
        return getattr(self._lut_local, 'buf', None)

    @_lut_buf.setter
    def _lut_buf(self, value):
        self._lut_local.buf = value

    @property
    def _lut_size(self):
        return getattr(self._lut_local, 'size', 0)

    @_lut_size.setter
    def _lut_size(self, value):
        self._lut_local.size = value

    def status(self) -> dict:
        """How the device resolved, for the UI. Initialises if needed.

        mode is 'gpu', 'software' (CPU adapter) or 'cpu' (numpy fallback).
        forced is True when LOFILOGIC_FORCE_CPU chose the CPU path."""
        # Don't probe the adapter when the CPU path is in use anyway, or the
        # banner would report a GPU that isn't being used.
        if not HAS_GPU:
            return {
                'mode': 'cpu',
                'available': _WGPU_AVAILABLE,
                'forced': _FORCE_CPU,
                'summary': self.adapter_summary,
                'info': dict(self.adapter_info),
            }
        ok = self._init()
        if not ok or self._device is None:
            mode = 'cpu'
        elif self.is_software_adapter:
            mode = 'software'
        else:
            mode = 'gpu'
        return {
            'mode': mode,
            'available': _WGPU_AVAILABLE,
            'forced': False,
            'summary': self.adapter_summary,
            'info': dict(self.adapter_info),
        }

    # ------------------------------------------------------------------
    # Per-render arena scope
    # ------------------------------------------------------------------

    def begin_render(self):
        """Allocate from the arena until end_render (see run_resident)."""
        self._arena.begin()

    def end_render(self):
        """Call after the result has been read back."""
        self._arena.end()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init(self):
        if self._device is not None:
            return True
        if not _WGPU_AVAILABLE:
            return False
        try:
            # Backends are probed when the instance is created, so they have to
            # be chosen before that.
            #
            # No GL on Linux: its EGL init aborts the process on some setups
            # (Steam Deck, panic in wgpu-hal gles/egl.rs). Elsewhere GL is kept
            # as a last resort for when DX12/Vulkan can't create a device, e.g.
            # a new GPU on an old graphics runtime.
            import sys as _sys
            backends = (["Primary"] if _sys.platform.startswith("linux")
                        else ["Primary", "GL"])
            # On a retry the instance already exists and set_instance_extras
            # would raise, hiding the original error.
            global _INSTANCE_EXTRAS_SET
            if not _INSTANCE_EXTRAS_SET:
                from wgpu.backends.wgpu_native.extras import set_instance_extras
                set_instance_extras(backends=backends)
                _INSTANCE_EXTRAS_SET = True
            adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
            info = dict(getattr(adapter, 'info', {}) or {})
            self.adapter_info = info
            self.adapter_summary = getattr(adapter, 'summary', None) or info.get('description', '?')
            # power_preference is only a hint; we may still get a software
            # adapter.
            adapter_type = str(info.get('adapter_type', '')).lower()
            sl = self.adapter_summary.lower()
            self.is_software_adapter = (
                adapter_type in ('cpu', 'software')
                or any(s in sl for s in ('warp', 'lavapipe', 'llvmpipe',
                                         'swiftshader', 'basic render', 'microsoft basic'))
            )
            self._device = adapter.request_device_sync()
            self._build_pipelines()
            self.init_failed = False
            if self.is_software_adapter:
                log.warning(
                    "⚠ GPU pipeline bound to a SOFTWARE adapter (%s, backend=%s) — "
                    "renders will be very slow. Check that GPU drivers are installed "
                    "and current; on a brand-new GPU the graphics runtime may be too "
                    "old to drive it.", self.adapter_summary, info.get('backend_type', '?'))
            else:
                log.info("✓ GPU pipeline ready: %s (type=%s, backend=%s)",
                         self.adapter_summary, info.get('adapter_type', '?'),
                         info.get('backend_type', '?'))
            return True
        except Exception as e:
            log.warning("⚠ GPU init failed (%s), using CPU fallbacks", e)
            self._device = None
            self.init_failed = True
            return False

    # (shader, layout spec, bgl attribute, ((pipeline attr, entry point), ...))
    # One spec char per binding, see _bgl. wgpu checks the spec against the
    # WGSL at pipeline creation, so a mismatch fails at init.
    _PIPELINE_TABLE = (
        # buffer pipelines
        ('lut.wgsl',               'RRWU',  '_lut_bg_layout',         (('_lut_pipeline', 'main'),)),
        ('acescct.wgsl',           'RW',    '_acescct_bg_layout',     (('_acescct_pipeline_decode', 'main_decode'),
                                                                       ('_acescct_pipeline_encode', 'main_encode'))),
        ('grain.wgsl',             'RRWU',  '_grain_bg_layout',       (('_grain_pipeline', 'main'),)),
        ('blend.wgsl',             'RRWU',  '_blend_bg_layout',       (('_unsharp_pipeline', 'main_unsharp'),)),
        ('gaussian_blur.wgsl',     'RRWU',  '_gauss_bg_layout',       (('_gauss_pipeline_h', 'main_h'),
                                                                       ('_gauss_pipeline_v', 'main_v'))),
        # texture pipelines
        ('encode_tex.wgsl',        'TS',    '_encode_tex_bg_layout',  (('_encode_tex_pipeline', 'main'),)),
        ('lut_tex.wgsl',           'TRSU',  '_lut_tex_bg_layout',     (('_lut_tex_pipeline', 'main'),)),
        ('gaussian_blur_tex.wgsl', 'TRS',   '_gauss_tex_bg_layout',   (('_gauss_tex_pipeline_h', 'main_h'),
                                                                       ('_gauss_tex_pipeline_v', 'main_v'))),
        ('downsample_tex.wgsl',    'TS',    '_downsample_bg_layout',  (('_downsample_pipeline', 'main'),)),
        ('disc_blur_tex.wgsl',     'TSU',   '_disc_bg_layout',        (('_disc_pipeline', 'main'),)),
        ('upsample_tex.wgsl',      'TS',    '_upsample_bg_layout',    (('_upsample_pipeline', 'main'),)),
        ('halation_mask.wgsl',     'TSU',   '_hal_mask_bg_layout',    (('_hal_mask_pipeline', 'main'),)),
        ('halation_highlights.wgsl', 'TTSU', '_hal_hi_bg_layout',     (('_hal_hi_pipeline', 'main'),)),
        ('halation_combine.wgsl', 'TTTTSU', '_hal_combine_bg_layout', (('_hal_combine_pipeline', 'main'),)),
        ('unsharp_tex.wgsl',       'TTSU',  '_unsharp_tex_bg_layout', (('_unsharp_tex_pipeline', 'main'),)),
        ('ca_tex.wgsl',            'TSU',   '_ca_tex_bg_layout',      (('_ca_tex_pipeline', 'main'),)),
        ('grain_tex.wgsl',         'TTSU',  '_grain_tex_bg_layout',   (('_grain_tex_pipeline', 'main'),)),
        ('edge_softness_tex.wgsl', 'TTSU',  '_edge_soft_bg_layout',   (('_edge_soft_pipeline', 'main'),)),
        ('vignette_tex.wgsl',      'TSU',   '_vignette_bg_layout',    (('_vignette_pipeline', 'main'),)),
        ('bloom_downmask.wgsl',    'TSU',   '_bloom_dm_bg_layout',    (('_bloom_dm_pipeline', 'main'),)),
        ('bloom_upadd.wgsl',       'TTSU',  '_bloom_ua_bg_layout',    (('_bloom_ua_pipeline', 'main'),)),
        ('cnr.wgsl',               'TS',    '_cnr_io_bg_layout',      (('_cnr_to_lab_pipeline', 'main_to_lab'),
                                                                       ('_cnr_to_acescg_pipeline', 'main_to_acescg'))),
        ('cnr.wgsl',               'TSU',   '_cnr_bil_bg_layout',     (('_cnr_bil_pipeline', 'main_bilateral'),
                                                                       ('_cnr_despike_pipeline', 'main_despike'))),
        ('color_matmul.wgsl',      'RWU',   '_colormat_bg_layout',    (('_colormat_pipeline', 'main'),)),
    )

    def _bgl(self, spec: str):
        """Compute bind-group layout, one char per binding:
            R read-only storage buffer   W storage buffer   U uniform buffer
            T sampled texture (unfilterable f32)
            S write-only storage texture (_TEX_FORMAT)
        """
        kind = {
            'R': {'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
            'W': {'buffer': {'type': wgpu.BufferBindingType.storage}},
            'U': {'buffer': {'type': wgpu.BufferBindingType.uniform}},
            'T': {'texture': {'sample_type': wgpu.TextureSampleType.unfilterable_float,
                              'view_dimension': wgpu.TextureViewDimension.d2}},
            'S': {'storage_texture': {'access': wgpu.StorageTextureAccess.write_only,
                                      'format': self._TEX_FORMAT,
                                      'view_dimension': wgpu.TextureViewDimension.d2}},
        }
        return self._device.create_bind_group_layout(entries=[
            {'binding': i, 'visibility': wgpu.ShaderStage.COMPUTE, **kind[ch]}
            for i, ch in enumerate(spec)
        ])

    def _build_pipelines(self):
        """Build everything in _PIPELINE_TABLE. A shader used by two rows is
        compiled once."""
        dev = self._device
        modules = {}
        for shader, spec, bgl_attr, pipes in self._PIPELINE_TABLE:
            layout = self._bgl(spec)
            setattr(self, bgl_attr, layout)
            pl = dev.create_pipeline_layout(bind_group_layouts=[layout])
            if shader not in modules:
                modules[shader] = dev.create_shader_module(code=_read_shader(shader))
            mod = modules[shader]
            for pipe_attr, entry in pipes:
                setattr(self, pipe_attr, dev.create_compute_pipeline(
                    layout=pl, compute={'module': mod, 'entry_point': entry}))

    # ------------------------------------------------------------------
    # LUT management
    # ------------------------------------------------------------------

    def upload_lut(self, lut_table: np.ndarray):
        """Upload an (N, N, N, 3) LUT for the current thread."""
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
        """Read a storage buffer back into a float32 array."""
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
    # Texture transfer
    # ------------------------------------------------------------------
    #
    # Images live in rgba32float textures, alpha unused. Not f16: the pipeline
    # is CPU-bound, and packing to half floats costs ~29 ms of CPU per render
    # (~130 ms on a slow Windows machine) to save bandwidth we don't need. f32
    # also keeps results matching the CPU versions to rounding. Stages use
    # textureLoad, so f32 not being filterable doesn't matter.

    _TEX_FORMAT = 'rgba32float'

    def _create_tex(self, shape):
        # Pooled inside a render scope, fresh otherwise.
        if self._arena.active:
            return self._arena.acquire_tex(shape, lambda: self._alloc_tex(shape))
        return self._alloc_tex(shape)

    def _alloc_tex(self, shape):
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
        """Read a texture back into an (H, W, 3) float32 array."""
        if not self._init():
            raise RuntimeError("GPU device unavailable")
        h, w = shape[:2]
        # bytes_per_row must be a multiple of 256.
        unpadded = w * 16
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
    # Frame -> Frame stages. They don't transfer anything themselves and
    # return None if there's no GPU, so the caller can fall back.
    # ------------------------------------------------------------------

    def encode_frame(self, frame: "Frame"):
        """ACEScct encode. CPU version: kernels.acescct_encode."""
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
        """Tetrahedral LUT with the uploaded table. None if no LUT is loaded."""
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
        """Separable Gaussian blur, clamp-to-edge."""
        if not self._init():
            return None
        if sigma <= 0:
            return frame
        return self._separable_blur(frame, self._gauss_kernel(sigma))

    def blur_frame_exp(self, frame: "Frame", lam: float):
        """Separable exp(-|x|/lam) blur: a sharp peak with a long tail, used
        for the halation scatter."""
        if not self._init():
            return None
        if lam <= 0:
            return frame
        return self._separable_blur(frame, self._exp_kernel(lam))

    def disc_blur(self, frame: "Frame", radius: float):
        """Average within `radius` texels (a defocus disc). O(r^2), so it's
        run at half resolution."""
        if not self._init():
            return None
        if radius <= 0:
            return frame
        h, w = frame.shape[:2]
        r = int(round(radius))
        dst = self._create_tex(frame.shape)
        uni = self._uniform(struct.pack('fiff', float(r * r), r, 0.0, 0.0))
        bg = self._device.create_bind_group(layout=self._disc_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': dst.create_view()},
            {'binding': 2, 'resource': {'buffer': uni, 'offset': 0, 'size': uni.size}},
        ])
        self._run2d(self._disc_pipeline, bg, w, h)
        return Frame.from_gpu(dst, frame.shape, self)

    def _separable_blur(self, frame: "Frame", kernel: np.ndarray):
        h, w = frame.shape[:2]
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
        """One compute pass over w*h in 8x8 workgroups."""
        enc = self._device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bind_group)
        cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8)
        cp.end()
        self._device.queue.submit([enc.finish()])

    def _downsample(self, frame: "Frame", factor: int):
        """Box downsample by ``factor``, at least 4 px."""
        h, w = frame.shape[:2]
        small_shape = (max(4, h // factor), max(4, w // factor), 3)
        sh, sw = small_shape[:2]
        dst = self._create_tex(small_shape)
        bg = self._device.create_bind_group(layout=self._downsample_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': dst.create_view()},
        ])
        self._run2d(self._downsample_pipeline, bg, sw, sh)
        return Frame.from_gpu(dst, small_shape, self)

    def _upsample(self, frame: "Frame", target_shape):
        """Bilinear upsample to ``target_shape``."""
        th, tw = target_shape[:2]
        dst = self._create_tex(target_shape)
        bg = self._device.create_bind_group(layout=self._upsample_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': dst.create_view()},
        ])
        self._run2d(self._upsample_pipeline, bg, tw, th)
        return Frame.from_gpu(dst, target_shape, self)

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

    def _halation_highlights(self, img: "Frame", mask: "Frame", tint):
        h, w = img.shape[:2]
        dst = self._create_tex(img.shape)
        # vec3 tint padded to 16 bytes.
        uni = self._uniform(struct.pack('4f', tint[0], tint[1], tint[2], 0.0))
        bg = self._device.create_bind_group(layout=self._hal_hi_bg_layout, entries=[
            {'binding': 0, 'resource': img.gpu().create_view()},
            {'binding': 1, 'resource': mask.gpu().create_view()},
            {'binding': 2, 'resource': dst.create_view()},
            {'binding': 3, 'resource': {'buffer': uni, 'offset': 0, 'size': uni.size}},
        ])
        self._run2d(self._hal_hi_pipeline, bg, w, h)
        return Frame.from_gpu(dst, img.shape, self)

    def _halation_combine(self, img: "Frame", glows, strength: float):
        h, w = img.shape[:2]
        dst = self._create_tex(img.shape)
        uni = self._uniform(struct.pack('4f', strength, 0.0, 0.0, 0.0))
        g0, g1, g2 = glows
        bg = self._device.create_bind_group(layout=self._hal_combine_bg_layout, entries=[
            {'binding': 0, 'resource': img.gpu().create_view()},
            {'binding': 1, 'resource': g0.gpu().create_view()},
            {'binding': 2, 'resource': g1.gpu().create_view()},
            {'binding': 3, 'resource': g2.gpu().create_view()},
            {'binding': 4, 'resource': dst.create_view()},
            {'binding': 5, 'resource': {'buffer': uni, 'offset': 0, 'size': uni.size}},
        ])
        self._run2d(self._hal_combine_pipeline, bg, w, h)
        return Frame.from_gpu(dst, img.shape, self)

    def halation_frame(self, frame: "Frame", threshold: float, blur_radius: float,
                       strength: float, warmth_pct: float = 100.0, k: float = 20.0):
        """Three-scale halation. CPU version: effects.apply_halation."""
        if not self._init():
            return None

        from .config import HALATION_SCALES, halation_scale_tint

        def glow(thresh, size, tint, kind):
            mask = self._halation_mask(frame, thresh, k)
            mask = self.blur_frame(mask, 2.0)
            hi = self._halation_highlights(frame, mask, tint)
            # Blur at half resolution: ~8x cheaper, and the bilinear upsample
            # softens the disc rim, which looks better anyway.
            small = self._downsample(hi, 2)
            if small is None:
                return None
            if kind == 'disc':
                small = self.disc_blur(small, size * 0.5)
            else:
                small = self.blur_frame_exp(small, size * 0.5)
            return self._upsample(small, frame.shape)

        glows = []
        for radius_mult, thresh_off, weight, gf, bf, kind in HALATION_SCALES:
            tint = halation_scale_tint(gf, bf, weight, warmth_pct)
            glows.append(glow(min(threshold + thresh_off, 0.98),
                              blur_radius * radius_mult, tint, kind))
        return self._halation_combine(frame, glows, strength)

    # ------------------------------------------------------------------
    # Post-LUT stages (display sRGB)
    # ------------------------------------------------------------------

    def softness_frame(self, frame: "Frame", sigma: float):
        """CPU version: effects.apply_softness."""
        return self.blur_frame(frame, sigma)

    def sharpen_frame(self, frame: "Frame", strength: float, radius: float):
        """Unsharp mask, unclamped. CPU version: effects.apply_sharpen."""
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
        """Blend a CPU-generated (H, W, 3) grain layer. Same math as
        grain_blend."""
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
        """Spectral CA: ``samples`` wavelengths, magnified from 1.0 (red) to
        1.0 + scale (blue). CPU version: effects.apply_chromatic_aberration."""
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
        """Blend toward a blurred copy from ``start`` (fraction of the corner
        radius) outward. CPU version: effects.apply_edge_softness."""
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
    # Pre-LUT stages (linear ACEScg)
    # ------------------------------------------------------------------

    def vignette_frame(self, frame: "Frame", strength: float, color_shift: float,
                       feather: float):
        """Cosine vignette with a cool edge tint. CPU version:
        effects.apply_vignette."""
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
        """Downsample 4x with a highlight mask, blur, upsample and add.
        CPU version: effects.apply_bloom."""
        if not self._init():
            return None
        if strength <= 0:
            return frame
        h, w = frame.shape[:2]
        scale = 4
        bh, bw = max(4, h // scale), max(4, w // scale)
        small_shape = (bh, bw, 3)

        # Downsample + highlight mask.
        small = self._create_tex(small_shape)
        uni_dm = self._uniform(struct.pack('4f', float(threshold), 0.0, 0.0, 0.0))
        bg_dm = self._device.create_bind_group(layout=self._bloom_dm_bg_layout, entries=[
            {'binding': 0, 'resource': frame.gpu().create_view()},
            {'binding': 1, 'resource': small.create_view()},
            {'binding': 2, 'resource': {'buffer': uni_dm, 'offset': 0, 'size': uni_dm.size}},
        ])
        self._run2d(self._bloom_dm_pipeline, bg_dm, bw, bh)

        # Sigma from the long edge so rotation doesn't change the glow.
        sigma = max(2, max(bw, bh) // 5)
        blurred = self.blur_frame(Frame.from_gpu(small, small_shape, self), float(sigma))
        if blurred is None:
            return None

        # Upsample + add.
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
        """ACEScg <-> Lab pass."""
        h, w = shape[:2]
        dst = self._create_tex(shape)
        bg = self._device.create_bind_group(layout=self._cnr_io_bg_layout, entries=[
            {'binding': 0, 'resource': src_tex.create_view()},
            {'binding': 1, 'resource': dst.create_view()},
        ])
        self._run2d(pipeline, bg, w, h)
        return dst

    def _cnr_lab_pass(self, pipeline, src_tex, shape, uni):
        """Despike or bilateral pass in Lab."""
        h, w = shape[:2]
        dst = self._create_tex(shape)
        bg = self._device.create_bind_group(layout=self._cnr_bil_bg_layout, entries=[
            {'binding': 0, 'resource': src_tex.create_view()},
            {'binding': 1, 'resource': dst.create_view()},
            {'binding': 2, 'resource': {'buffer': uni, 'offset': 0, 'size': uni.size}},
        ])
        self._run2d(pipeline, bg, w, h)
        return dst

    def cnr_frame(self, frame: "Frame", sigma: float, despike=(0.0, 0.0)):
        """Chroma NR: optional despike, then a bilateral on a*/b* in Lab.

        ``despike`` comes from config.cnr_despike_thresholds. The window
        matches the cv2 version, effects.reduce_color_noise_chroma.
        """
        if not self._init():
            return None
        thr_green, thr_other = despike
        if sigma <= 0 and thr_green <= 0:
            return frame
        from .config import cnr_sigma_color

        lab = self._cnr_io(self._cnr_to_lab_pipeline, frame.gpu(), frame.shape)
        if thr_green > 0:
            uni_d = self._uniform(struct.pack(
                '8f', 0.0, 0.0, 0.0, float(thr_green), float(thr_other), 0.0, 0.0, 0.0))
            lab = self._cnr_lab_pass(self._cnr_despike_pipeline, lab, frame.shape, uni_d)
        if sigma > 0:
            d = max(5, int(sigma) * 2 + 3)
            if d % 2 == 0:
                d += 1
            radius = d // 2
            sigma_color = cnr_sigma_color(sigma)
            uni = self._uniform(struct.pack(
                '8f', float(sigma), float(sigma_color), float(radius), 0.0, 0.0, 0.0, 0.0, 0.0))
            lab = self._cnr_lab_pass(self._cnr_bil_pipeline, lab, frame.shape, uni)
        out = self._cnr_io(self._cnr_to_acescg_pipeline, lab, frame.shape)
        return Frame.from_gpu(out, frame.shape, self)

    def _uniform(self, data: bytes):
        # Uniform buffers must be multiples of 16 bytes
        padded = data + b'\x00' * (16 - len(data) % 16) if len(data) % 16 else data
        if self._arena.active:
            buf = self._arena.acquire_uni(len(padded), lambda n=len(padded): self._device.create_buffer(
                size=n,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            ))
            self._device.queue.write_buffer(buf, 0, padded)
            return buf
        return self._device.create_buffer_with_data(
            data=padded,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

    def _dispatch(self, pipeline, bind_group, n_elements: int, workgroup_size: int = 256):
        n_wg = (n_elements + workgroup_size - 1) // workgroup_size
        # Dispatch dimensions are limited to 65535, so large images go 2D.
        # Shaders rebuild the index as id.y * (65535 * workgroup_size) + id.x
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
    # Buffer operations (numpy in, numpy out)
    # ------------------------------------------------------------------

    def apply_lut(self, img: np.ndarray) -> np.ndarray:
        """Tetrahedral LUT with the uploaded table."""
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
        """img @ M.T per pixel. Used at load for raw -> ACEScg."""
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
        """ACEScct → linear."""
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
        """Normalised 1-D Gaussian, radius 3 sigma."""
        radius = max(1, int(round(sigma * 3)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
        return k / k.sum()

    @staticmethod
    def _exp_kernel(lam: float) -> np.ndarray:
        """Normalised 1-D exp(-|x|/lam), radius 4 lam so the tail isn't cut
        off. Must match kernels.exp_blur."""
        radius = max(1, int(round(lam * 4)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-np.abs(x) / lam).astype(np.float32)
        return k / k.sum()

    def gaussian_blur(self, img: np.ndarray, sigma: float) -> np.ndarray | None:
        """Separable Gaussian blur on an (H, W) or (H, W, 3) array."""
        if not self._init():
            return None
        if sigma <= 0:
            return img.copy()

        single_ch = img.ndim == 2
        if single_ch:
            img3 = img[:, :, np.newaxis]
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
        buf_mid = self._make_output(n)
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

        # Vertical pass: buf_mid → buf_out
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
    """An (H, W, 3) image held as a float32 array, a GPU texture, or both.

    cpu() and gpu() create the missing side on demand and cache it, so
    consecutive GPU stages never go through numpy. Frames are written once;
    stages return new ones.
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
        """Wrap a numpy array. Uploads on the first .gpu()."""
        return cls(pipeline or gpu, cpu=arr)

    @classmethod
    def from_gpu(cls, tex, shape, pipeline: "GPUPipeline" = None) -> "Frame":
        """Wrap a texture. Reads back on the first .cpu()."""
        return cls(pipeline or gpu, tex=tex, shape=shape)

    @property
    def shape(self):
        return self._shape

    @property
    def on_gpu(self) -> bool:
        return self._tex is not None

    def cpu(self) -> np.ndarray:
        if self._cpu is None:
            self._cpu = self._p._download_tex(self._tex, self._shape)
        return self._cpu

    def gpu(self):
        if self._tex is None:
            self._tex = self._p._upload_tex(self._cpu)
        return self._tex


gpu = GPUPipeline()
# LOFILOGIC_FORCE_CPU=1 uses the numpy paths even when wgpu is available.
_FORCE_CPU = os.environ.get('LOFILOGIC_FORCE_CPU', '').lower() in ('1', 'true', 'yes')
HAS_GPU = _WGPU_AVAILABLE and not _FORCE_CPU
