// Disc (circle-of-confusion) blur: averages every texel within `radius` of the
// centre. This is the halation core kernel — back-reflection off the film base
// is a defocused copy of the highlights (a geometric circle of confusion set by
// base thickness), so it has a DEFINED edge, unlike a Gaussian/exponential that
// only decays. Not separable, so this is a single O(r^2) 2D pass; halation runs
// it at half resolution (the bilinear upsample afterwards is the rim-soften).
//
// `radius` is in texels of THIS texture; r2 = radius*radius (the circle test).

struct U { r2: f32, radius: i32, _p0: f32, _p1: f32, }

@group(0) @binding(0) var          src: texture_2d<f32>;
@group(0) @binding(1) var          dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> u:   U;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(src));
    if gid.x >= u32(dims.x) || gid.y >= u32(dims.y) { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    var acc = vec4f(0.0);
    var n   = 0.0;
    for (var dy: i32 = -u.radius; dy <= u.radius; dy++) {
        for (var dx: i32 = -u.radius; dx <= u.radius; dx++) {
            if f32(dx * dx + dy * dy) <= u.r2 {
                let s = clamp(p + vec2i(dx, dy), vec2i(0), dims - vec2i(1));
                acc += textureLoad(src, s, 0);
                n   += 1.0;
            }
        }
    }
    textureStore(dst, p, acc / max(n, 1.0));
}
