// Bloom stage 2 (texture-resident): bilinear upsample of the blurred bloom layer
// + additive blend onto the full-res image.
//
// Twin of the second half of effects.apply_bloom (linear path):
//   bloom_layer = bilinear_upsample(blurred_small)        # cv2 INTER_LINEAR
//   out         = max(0, full + bloom_layer * strength)
//
// The upsample uses cv2's coordinate convention: src = (dst+0.5)*ssize/dsize-0.5,
// bilinear with clamp-to-edge. rgba32float is not filterable, so it's manual.

struct U {
    strength: f32,
    _p0: f32, _p1: f32, _p2: f32,
}

@group(0) @binding(0) var          full:  texture_2d<f32>;
@group(0) @binding(1) var          small: texture_2d<f32>;   // blurred bloom layer
@group(0) @binding(2) var          dst:   texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> u:     U;

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
    let fdim = textureDimensions(full);
    if gid.x >= fdim.x || gid.y >= fdim.y { return; }
    let sdim = vec2f(textureDimensions(small));
    let ratio = sdim / vec2f(fdim);
    let pos = (vec2f(f32(gid.x), f32(gid.y)) + vec2f(0.5)) * ratio - vec2f(0.5);

    let pi = vec2i(i32(gid.x), i32(gid.y));
    let base = textureLoad(full, pi, 0).rgb;
    let bloom = sample_small(pos, sdim);
    textureStore(dst, pi, vec4f(max(vec3f(0.0), base + bloom * u.strength), 1.0));
}
