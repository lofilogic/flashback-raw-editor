// out = max(0, img + (g_core + g_mid + g_wide) * strength)
//
// Additive, not screen: screen assumes [0, 1], and with linear values above 1
// it wiped out the halo exactly around the brightest highlights.

struct U { strength: f32, _p0: f32, _p1: f32, _p2: f32, }

@group(0) @binding(0) var          img: texture_2d<f32>;
@group(0) @binding(1) var          g0:  texture_2d<f32>;
@group(0) @binding(2) var          g1:  texture_2d<f32>;
@group(0) @binding(3) var          g2:  texture_2d<f32>;
@group(0) @binding(4) var          dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var<uniform> u:   U;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(img);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    let base = textureLoad(img, p, 0).rgb;
    let glow = (textureLoad(g0, p, 0).rgb
              + textureLoad(g1, p, 0).rgb
              + textureLoad(g2, p, 0).rgb) * u.strength;
    textureStore(dst, p, vec4f(max(base + glow, vec3f(0.0)), 1.0));
}
