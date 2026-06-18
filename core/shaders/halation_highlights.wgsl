// Halation highlight extraction for one scale: highlights = img * mask, then
// tinted per-channel by `tint` (weight already folded in). The tint carries the
// scale's chroma — red is the carrier, green/blue fall off with warmth — so the
// summed glow reddens outward. Replaces the old fixed r / 0.2g / 0 weighting,
// which was radius-independent and couldn't make the halo redden with distance.
// See config.HALATION_SCALES / halation_scale_tint for where tint comes from.

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
