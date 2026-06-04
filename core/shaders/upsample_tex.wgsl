// Bilinear upsample, texture-resident (rgba32float).
//
// Reads a small src and writes a larger dst, manual bilinear (f32 textures are
// not filterable, so no hardware sampler). Same cv2 coordinate convention +
// clamp-to-edge as bloom_upadd.wgsl, but a plain resize with no additive blend —
// the upsample half of halation's downsample -> blur -> upsample glow.

@group(0) @binding(0) var small: texture_2d<f32>;
@group(0) @binding(1) var dst:   texture_storage_2d<rgba32float, write>;

fn sample_small(pos: vec2f, sdim: vec2f) -> vec3f {
    let p  = clamp(pos, vec2f(0.0), sdim - vec2f(1.0));
    let p0 = floor(p);
    let fr = p - p0;
    let i0 = vec2i(p0);
    let i1 = vec2i(min(p0 + vec2f(1.0), sdim - vec2f(1.0)));
    let c00 = textureLoad(small, vec2i(i0.x, i0.y), 0).rgb;
    let c10 = textureLoad(small, vec2i(i1.x, i0.y), 0).rgb;
    let c01 = textureLoad(small, vec2i(i0.x, i1.y), 0).rgb;
    let c11 = textureLoad(small, vec2i(i1.x, i1.y), 0).rgb;
    return mix(mix(c00, c10, fr.x), mix(c01, c11, fr.x), fr.y);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let fdim = textureDimensions(dst);
    if gid.x >= fdim.x || gid.y >= fdim.y { return; }
    let sdim = vec2f(textureDimensions(small));
    let ratio = sdim / vec2f(fdim);
    let pos = (vec2f(f32(gid.x), f32(gid.y)) + vec2f(0.5)) * ratio - vec2f(0.5);
    textureStore(dst, vec2i(i32(gid.x), i32(gid.y)), vec4f(sample_small(pos, sdim), 1.0));
}
