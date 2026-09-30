// Separable Gaussian blur on a flat buffer: main_h, then main_v.
// 1 or 3 interleaved channels: index = (y * width + x) * num_channels + c.
//
// Clamp-to-edge. cv2 defaults to BORDER_REFLECT_101, which is close enough at
// these sigmas.

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
    let pixel = id.y * 4194240u + id.x; // 65535 * 64, for 2D dispatch
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
    let pixel = id.y * 4194240u + id.x;
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
