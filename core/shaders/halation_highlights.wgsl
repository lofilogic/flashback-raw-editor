// Highlights for one halation scale: img * mask * tint.
// tint comes from config.halation_scale_tint.

struct U { tint: vec3f, _p: f32, }

@group(0) @binding(0) var          img:  texture_2d<f32>;
@group(0) @binding(1) var          mask: texture_2d<f32>;
@group(0) @binding(2) var          dst:  texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> u:    U;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(img);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    let c = textureLoad(img, p, 0).rgb;
    let m = textureLoad(mask, p, 0).r;
    textureStore(dst, p, vec4f(c * m * u.tint, 1.0));
}
