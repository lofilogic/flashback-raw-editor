// Halation combine: glow = (g_core + g_mid + g_wide) * strength, ADDED to the
// base image in linear light, clamped to >= 0. The per-scale weights and chroma
// are already baked into each glow (see halation_highlights + the tint from
// config.HALATION_SCALES), so this just sums them.
//
// Additive, not screen: halation is re-exposure from scattered/back-reflected
// light, which physically ADDS. Screen blend (1-(1-a)(1-b)) also assumes both
// operands are in [0,1] — but this runs on linear ACEScg where highlights
// exceed 1, so screen produced (1-base)(1-glow) > 0 from two negatives and
// CRUSHED the halo to zero exactly where the source was brightest. Additive
// keeps the defined, punchy halo around blown highlights.

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
