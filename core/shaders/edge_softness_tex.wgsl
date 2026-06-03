// Radial edge (corner) softness, texture-resident.
//
// Emulates lens field curvature: the centre stays sharp and the image softens
// smoothly toward the corners. Implemented as a sharp->blurred blend whose
// weight grows with radius — cheap, GPU-friendly, and reads like gentle
// defocus. The blurred input is a full Gaussian blur of the sharp image
// (produced by blur_frame); this pass only chooses how much of it to mix in.
//
//   r_norm = dist(p, centre) / dist(corner, centre)        in [0, 1]
//   w      = smoothstep(start, 1.0, r_norm) * strength
//   out    = mix(sharp, blurred, w)

struct U {
    strength: f32,   // max blend weight at the corners (0..1)
    start:    f32,   // r_norm where softening begins (0..1)
    _p0:      f32,
    _p1:      f32,
}

@group(0) @binding(0) var          sharp:   texture_2d<f32>;
@group(0) @binding(1) var          blurred: texture_2d<f32>;
@group(0) @binding(2) var          dst:     texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> u:       U;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dimsu = textureDimensions(sharp);
    if gid.x >= dimsu.x || gid.y >= dimsu.y { return; }
    let dims = vec2f(dimsu);
    let p = vec2f(f32(gid.x), f32(gid.y));
    let c = dims * 0.5;
    let r_norm = length(p - c) / max(length(c), 1e-6);
    let w = smoothstep(u.start, 1.0, r_norm) * u.strength;

    let pi = vec2i(i32(gid.x), i32(gid.y));
    let a = textureLoad(sharp, pi, 0).rgb;
    let b = textureLoad(blurred, pi, 0).rgb;
    textureStore(dst, pi, vec4f(mix(a, b, w), 1.0));
}
