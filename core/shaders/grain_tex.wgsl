// Film grain blend, texture-resident twin of grain.wgsl.
//
// Identical math to the buffer kernel, per channel:
//   grain_delta = (2*grain - 1) * intensity
//   weight      = (1-bias)*(1-pixel) + bias*pixel
//   falloff     = min_grain + weight*(1-min_grain)
//   result      = clamp(pixel + grain_delta*falloff, 0, 1)
//
// Reads the image and a same-size grain layer as rgba32float textures and
// writes the blended result, so grain stays GPU-resident between stages.

struct Uniforms {
    intensity:      f32,
    min_grain:      f32,
    highlight_bias: f32,
    _pad:           f32,
}

@group(0) @binding(0) var               image: texture_2d<f32>;
@group(0) @binding(1) var               grain: texture_2d<f32>;
@group(0) @binding(2) var               dst:   texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform>      u:     Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(image);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    let img_val   = textureLoad(image, p, 0).rgb;
    let grain_val = textureLoad(grain, p, 0).rgb;
    let grain_delta = (2.0 * grain_val - 1.0) * u.intensity;
    let weight      = (1.0 - u.highlight_bias) * (1.0 - img_val) + u.highlight_bias * img_val;
    let falloff     = u.min_grain + weight * (1.0 - u.min_grain);
    let o = clamp(img_val + grain_delta * falloff, vec3f(0.0), vec3f(1.0));
    textureStore(dst, p, vec4f(o, 1.0));
}
