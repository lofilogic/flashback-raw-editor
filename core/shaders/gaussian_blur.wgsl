// Separable Gaussian blur — two entry points share the same bind group layout.
//
// Run main_h first (horizontal pass), then main_v (vertical pass) on the result.
// Both passes support 1-channel (H×W) and 3-channel (H×W×3) images via
// the num_channels uniform. Images are stored as flat f32 arrays with
// interleaved channels: index = (y * width + x) * num_channels + c.
//
// Kernel weights are a pre-computed normalised 1-D Gaussian passed as a
// storage buffer. Boundary pixels are handled by clamping to edge (repeat
// the edge value), which matches cv2.BORDER_REFLECT_101... actually
// cv2.GaussianBlur default is BORDER_REFLECT_101 but BORDER_REPLICATE is
// close enough for the large sigmas we use here (halation, CNR, desat).

struct Uniforms {
    width:        u32,
    height:       u32,
    kernel_size:  u32,
    num_channels: u32,
}

@group(0) @binding(0) var<storage, read>       img_in:  array<f32>;
@group(0) @binding(1) var<storage, read>       kernel:  array<f32>;
@group(0) @binding(2) var<storage, read_write> img_out: array<f32>;
@group(0) @binding(3) var<uniform>             u:       Uniforms;

@compute @workgroup_size(64)
fn main_h(@builtin(global_invocation_id) id: vec3u) {
    let pixel = id.x;
    if pixel >= u.width * u.height { return; }

    let x    = i32(pixel % u.width);
    let y    = i32(pixel / u.width);
    let half = i32(u.kernel_size / 2u);
    let base = pixel * u.num_channels;

    for (var c: u32 = 0u; c < u.num_channels; c++) {
        var acc = 0.0f;
        for (var k: u32 = 0u; k < u.kernel_size; k++) {
            let sx  = clamp(x + i32(k) - half, 0, i32(u.width) - 1);
            let idx = (u32(y) * u.width + u32(sx)) * u.num_channels + c;
            acc    += img_in[idx] * kernel[k];
        }
        img_out[base + c] = acc;
    }
}

@compute @workgroup_size(64)
fn main_v(@builtin(global_invocation_id) id: vec3u) {
    let pixel = id.x;
    if pixel >= u.width * u.height { return; }

    let x    = i32(pixel % u.width);
    let y    = i32(pixel / u.width);
    let half = i32(u.kernel_size / 2u);
    let base = pixel * u.num_channels;

    for (var c: u32 = 0u; c < u.num_channels; c++) {
        var acc = 0.0f;
        for (var k: u32 = 0u; k < u.kernel_size; k++) {
            let sy  = clamp(y + i32(k) - half, 0, i32(u.height) - 1);
            let idx = (u32(sy) * u.width + u32(x)) * u.num_channels + c;
            acc    += img_in[idx] * kernel[k];
        }
        img_out[base + c] = acc;
    }
}
