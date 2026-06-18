// Per-pixel 3x3 colour-space matrix transform (out = M @ rgb).
//
// Load-time helper for raw -> ACEScg (and the generic linsRGB -> ACEScg): a flat
// (N*3) float buffer in, the same out, with M passed as three rows. The CPU
// equivalent is (img.reshape(-1,3) @ M.T). Worth it mainly for large frames,
// where the numpy matmul scales with megapixels; tiny frames are dominated by
// transfer either way and fall back fine.

struct U {
    r0: vec4f,   // M[0], M[1], M[2] in .xyz (.w padding)
    r1: vec4f,
    r2: vec4f,
}

@group(0) @binding(0) var<storage, read>       inp:  array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@group(0) @binding(2) var<uniform>             u:    U;

// NaN -> 0, +/-Inf -> finite f32 extremes. Stops a non-finite raw sample from
// propagating through the matrix and the later log encode, where Apple GPU
// families (M1 vs M3) diverge and tint clipped highlights (M1 cyan bug). No-op
// for finite inputs.
fn sanitize(v: f32) -> f32 {
    let n = select(v, 0.0, v != v);
    return clamp(n, -3.4e38, 3.4e38);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let px = id.y * 16776960u + id.x;   // 65535 * 256 — 2D dispatch for large images
    let count = arrayLength(&inp) / 3u;
    if px >= count { return; }
    let base = px * 3u;
    let rgb = vec3f(sanitize(inp[base]), sanitize(inp[base + 1u]), sanitize(inp[base + 2u]));
    outp[base]      = sanitize(dot(u.r0.xyz, rgb));
    outp[base + 1u] = sanitize(dot(u.r1.xyz, rgb));
    outp[base + 2u] = sanitize(dot(u.r2.xyz, rgb));
}
