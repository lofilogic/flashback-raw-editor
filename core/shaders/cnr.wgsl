// Chroma noise reduction in Lab. See effects.reduce_color_noise_chroma.
//   main_to_lab     ACEScg -> Lab
//   main_despike    3x3 median clamp on a*/b* (optional)
//   main_bilateral  bilateral on a*/b*, L* untouched
//   main_to_acescg  Lab -> ACEScg
//
// The bilateral uses the same two Gaussians as cv2.bilateralFilter, with
// clamp-to-edge borders.

// --- Lab <-> ACEScg constants (D60 white) ---
const M_RGB2XYZ_0 = vec3f( 0.6624541811,  0.1340042065,  0.1561876744);
const M_RGB2XYZ_1 = vec3f( 0.2722287168,  0.6740817658,  0.0536895174);
const M_RGB2XYZ_2 = vec3f(-0.0055746495,  0.0040607335,  1.0103391685);
const M_XYZ2RGB_0 = vec3f( 1.6410233797, -0.3248032942, -0.2364246952);
const M_XYZ2RGB_1 = vec3f(-0.6636628587,  1.6153315917,  0.0167563477);
const M_XYZ2RGB_2 = vec3f( 0.0117218943, -0.0082844420,  0.9883948585);
const WHITE = vec3f(0.95265, 1.0, 1.00883);
const LAB_DELTA3: f32 = 0.00885645167;   // (6/29)^3
const LAB_DELTA:  f32 = 0.20689655172;   // 6/29
const LAB_SLOPE:  f32 = 7.78703703704;   // (29/6)^2 / 3
const LAB_OFFSET: f32 = 0.13793103448;   // 4/29

fn f_lab(t: f32) -> f32 {
    if t > LAB_DELTA3 { return pow(max(t, 0.0), 0.3333333333); }
    return LAB_SLOPE * t + LAB_OFFSET;
}

fn f_lab_inv(t: f32) -> f32 {
    if t > LAB_DELTA { return t * t * t; }
    return (t - LAB_OFFSET) / LAB_SLOPE;
}

fn to_lab(rgb: vec3f) -> vec3f {
    var xyz = vec3f(dot(M_RGB2XYZ_0, rgb), dot(M_RGB2XYZ_1, rgb), dot(M_RGB2XYZ_2, rgb));
    xyz = max(xyz, vec3f(0.0)) / WHITE;
    let fx = f_lab(xyz.x);
    let fy = f_lab(xyz.y);
    let fz = f_lab(xyz.z);
    return vec3f(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz));
}

fn to_acescg(lab: vec3f) -> vec3f {
    let fy = (lab.x + 16.0) / 116.0;
    let xyz = vec3f(
        f_lab_inv(lab.y / 500.0 + fy),
        f_lab_inv(fy),
        f_lab_inv(fy - lab.z / 200.0),
    ) * WHITE;
    return vec3f(dot(M_XYZ2RGB_0, xyz), dot(M_XYZ2RGB_1, xyz), dot(M_XYZ2RGB_2, xyz));
}

// ---- pass 1: ACEScg -> Lab ----
@group(0) @binding(0) var src_a: texture_2d<f32>;
@group(0) @binding(1) var dst_a: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main_to_lab(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(src_a);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    textureStore(dst_a, p, vec4f(to_lab(textureLoad(src_a, p, 0).rgb), 1.0));
}

// ---- pass 3: Lab -> ACEScg ----
@compute @workgroup_size(8, 8)
fn main_to_acescg(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(src_a);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    textureStore(dst_a, p, vec4f(to_acescg(textureLoad(src_a, p, 0).rgb), 1.0));
}

// ---- uniform shared by despike and bilateral; each reads its own fields ----
struct U {
    sigma_space: f32,   // bilateral spatial sigma
    sigma_color: f32,   // bilateral range sigma
    radius:      f32,   // bilateral window radius
    thr_green:   f32,   // despike clamp limit for the green direction (-a*)
    thr_other:   f32,   // despike clamp limit for every other direction
    _pad0:       f32,
    _pad1:       f32,
    _pad2:       f32,
}

@group(0) @binding(0) var          src_b: texture_2d<f32>;
@group(0) @binding(1) var          dst_b: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> u:     U;

// Sort a pair into (lo, hi). Returns by value: passing `&p[i]` into a function
// makes Naga's SPIR-V backend panic ("Expression is not cached!").
fn so(a: f32, b: f32) -> vec2f {
    return vec2f(min(a, b), max(a, b));
}

// Median of 9, 19-comparison network.
fn med9(v: array<f32, 9>) -> f32 {
    var p = v;
    var s: vec2f;
    s = so(p[1], p[2]); p[1] = s.x; p[2] = s.y;
    s = so(p[4], p[5]); p[4] = s.x; p[5] = s.y;
    s = so(p[7], p[8]); p[7] = s.x; p[8] = s.y;
    s = so(p[0], p[1]); p[0] = s.x; p[1] = s.y;
    s = so(p[3], p[4]); p[3] = s.x; p[4] = s.y;
    s = so(p[6], p[7]); p[6] = s.x; p[7] = s.y;
    s = so(p[1], p[2]); p[1] = s.x; p[2] = s.y;
    s = so(p[4], p[5]); p[4] = s.x; p[5] = s.y;
    s = so(p[7], p[8]); p[7] = s.x; p[8] = s.y;
    s = so(p[0], p[3]); p[0] = s.x; p[3] = s.y;
    s = so(p[5], p[8]); p[5] = s.x; p[8] = s.y;
    s = so(p[4], p[7]); p[4] = s.x; p[7] = s.y;
    s = so(p[3], p[6]); p[3] = s.x; p[6] = s.y;
    s = so(p[1], p[4]); p[1] = s.x; p[4] = s.y;
    s = so(p[2], p[5]); p[2] = s.x; p[5] = s.y;
    s = so(p[4], p[7]); p[4] = s.x; p[7] = s.y;
    s = so(p[4], p[2]); p[4] = s.x; p[2] = s.y;
    s = so(p[6], p[4]); p[6] = s.x; p[4] = s.y;
    s = so(p[4], p[2]); p[4] = s.x; p[2] = s.y;
    return p[4];
}

// ---- pass 2a: despike ----
// Clamp a*/b* to median ± thr of the 3x3 neighbourhood. The bilateral keeps
// single-pixel spikes because they look like edges. a* below the median uses
// thr_green, everything else thr_other.
@compute @workgroup_size(8, 8)
fn main_despike(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(src_b));
    if gid.x >= u32(dims.x) || gid.y >= u32(dims.y) { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    let center = textureLoad(src_b, p, 0).rgb;     // (L, a, b)

    var va: array<f32, 9>;
    var vb: array<f32, 9>;
    var idx = 0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = clamp(p + vec2i(dx, dy), vec2i(0), dims - vec2i(1));
            let s = textureLoad(src_b, q, 0).rgb;
            va[idx] = s.y;
            vb[idx] = s.z;
            idx++;
        }
    }
    let med_a = med9(va);
    let med_b = med9(vb);
    let a = clamp(center.y, med_a - u.thr_green, med_a + u.thr_other);
    let b = clamp(center.z, med_b - u.thr_other, med_b + u.thr_other);
    textureStore(dst_b, p, vec4f(center.x, a, b, 1.0));
}

// ---- pass 2b: bilateral on a*/b* ----

@compute @workgroup_size(8, 8)
fn main_bilateral(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(src_b));
    if gid.x >= u32(dims.x) || gid.y >= u32(dims.y) { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    let center = textureLoad(src_b, p, 0).rgb;     // (L, a, b)

    let rad = i32(u.radius);
    let r2 = rad * rad;       // circular window, like cv2
    let cs = -0.5 / (u.sigma_space * u.sigma_space);
    let cr = -0.5 / (u.sigma_color * u.sigma_color);

    var acc = vec2f(0.0);     // weighted a, b
    var wsum = vec2f(0.0);
    for (var dy = -rad; dy <= rad; dy++) {
        for (var dx = -rad; dx <= rad; dx++) {
            if dx * dx + dy * dy > r2 { continue; }
            let q = clamp(p + vec2i(dx, dy), vec2i(0), dims - vec2i(1));
            let s = textureLoad(src_b, q, 0).rgb;
            let ws = exp(f32(dx * dx + dy * dy) * cs);
            let da = s.y - center.y;
            let db = s.z - center.z;
            let wa = ws * exp(da * da * cr);
            let wb = ws * exp(db * db * cr);
            acc  += vec2f(s.y * wa, s.z * wb);
            wsum += vec2f(wa, wb);
        }
    }
    let ab = acc / max(wsum, vec2f(1e-12));
    textureStore(dst_b, p, vec4f(center.x, ab.x, ab.y, 1.0));
}
