// Average of all texels within `radius`. The halation core: back-reflection
// off the film base is a defocused copy of the highlights, with a defined
// edge. Not separable, O(r^2), so it runs at half resolution.

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
