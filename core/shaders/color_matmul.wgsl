// out = M @ rgb per pixel, on a flat buffer. Used at load for large images.

struct U {
    r0: vec4f,   // M[0], M[1], M[2] in .xyz (.w padding)
    r1: vec4f,
    r2: vec4f,
}

@group(0) @binding(0) var<storage, read>       inp:  array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(2) var<uniform>             u:    U;

// NaN -> 0, Inf -> finite. See acescct.wgsl.
fn sanitize(v: f32) -> f32 {
    let n = select(v, 0.0, v != v);
    return clamp(n, -3.4e38, 3.4e38);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let px = id.y * 16776960u + id.x;   // 65535 * 256, for 2D dispatch
    let count = arrayLength(&inp) / 3u;
    if px >= count { return; }
    let base = px * 3u;
    let rgb = vec3f(sanitize(inp[base]), sanitize(inp[base + 1u]), sanitize(inp[base + 2u]));
    outp[base]      = sanitize(dot(u.r0.xyz, rgb));
    outp[base + 1u] = sanitize(dot(u.r1.xyz, rgb));
    outp[base + 2u] = sanitize(dot(u.r2.xyz, rgb));
}
