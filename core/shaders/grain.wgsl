// Film grain blend with luma-based highlight bias.
// Formula matches the Numba kernel exactly:
//   grain_delta = (2*grain - 1) * intensity
//   weight = (1-bias)*(1-pixel) + bias*pixel
//   falloff = min_grain + weight*(1-min_grain)
//   result = clamp(pixel + grain_delta*falloff, 0, 1)

struct Uniforms {
    intensity:      f32,
    min_grain:      f32,
    highlight_bias: f32,
    _pad:           f32,
}

@group(0) @binding(0) var<storage, read>       image:  array<f32>;
@group(0) @binding(1) var<storage, read>       grain:  array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform>             u:      Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.y * 16776960u + id.x; // 65535 * 256 — supports 2D dispatch for large images
    if i >= arrayLength(&image) { return; }
    let img_val    = image[i];
    let grain_val  = grain[i];
    let grain_delta = (2.0 * grain_val - 1.0) * u.intensity;
    let weight     = (1.0 - u.highlight_bias) * (1.0 - img_val) + u.highlight_bias * img_val;
    let falloff    = u.min_grain + weight * (1.0 - u.min_grain);
    output[i]      = clamp(img_val + grain_delta * falloff, 0.0, 1.0);
}
