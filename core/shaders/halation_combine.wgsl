// Halation combine: glow_combined = (glow1 + glow2*0.6) * strength, then
// screen-blend onto the base image and clamp to >= 0. Matches the tail of
// effects.apply_halation: screen_blend(img, glow_combined), max(result, 0).

struct U { strength: f32, _p0: f32, _p1: f32, _p2: f32, }

@group(0) @binding(0) var          img:   texture_2d<f32>;
@group(0) @binding(1) var          glow1: texture_2d<f32>;
@group(0) @binding(2) var          glow2: texture_2d<f32>;
@group(0) @binding(3) var          dst:   texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<uniform> u:     U;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(img);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    let base = textureLoad(img,   p, 0).rgb;
    let g1   = textureLoad(glow1, p, 0).rgb;
    let g2   = textureLoad(glow2, p, 0).rgb;
    let glow = (g1 + g2 * 0.6) * u.strength;
    let res  = 1.0 - (1.0 - base) * (1.0 - glow);   // screen blend
    textureStore(dst, p, vec4f(max(res, vec3f(0.0)), 1.0));
}
