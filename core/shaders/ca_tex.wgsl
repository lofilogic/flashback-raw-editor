// Spectral chromatic aberration, texture-resident.
//
// Models lateral (transverse) CA: a real lens magnifies long wavelengths less
// and short wavelengths more, so colour fringes grow with distance from the
// optical centre and vanish at it. Instead of three discrete R/G/B copies, we
// integrate `samples` points across the visible spectrum: each is displaced
// radially by its own magnification and weighted by that band's RGB
// sensitivity, producing a smooth purple->green fringe like real glass.
//
// Strength matches the legacy effect's envelope: red (longest wavelength) stays
// ~unshifted and blue (shortest) is magnified outward by (1 + `scale`), where
// scale = ca_pixels / (width/2). So existing ca_pixels values carry straight
// over. (The per-sample magnification is the reciprocal 1/(1+scale*t) — see the
// loop — which is what gives the physically-correct fringe direction.)
//
// rgba32float is not filterable, so sampling is manual bilinear with
// clamp-to-edge (== cv2 BORDER_REPLICATE).

struct U {
    scale:   f32,   // blue-edge magnification minus 1
    samples: f32,   // spectral sample count
    _p0:     f32,
    _p1:     f32,
}

@group(0) @binding(0) var          src: texture_2d<f32>;
@group(0) @binding(1) var          dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> u:   U;

fn sample_edge(pos: vec2f, dims: vec2f) -> vec3f {
    let p  = clamp(pos, vec2f(0.0), dims - vec2f(1.0));
    let p0 = floor(p);
    let fr = p - p0;
    let i0 = vec2i(p0);
    let i1 = vec2i(min(p0 + vec2f(1.0), dims - vec2f(1.0)));
    let c00 = textureLoad(src, vec2i(i0.x, i0.y), 0).rgb;
    let c10 = textureLoad(src, vec2i(i1.x, i0.y), 0).rgb;
    let c01 = textureLoad(src, vec2i(i0.x, i1.y), 0).rgb;
    let c11 = textureLoad(src, vec2i(i1.x, i1.y), 0).rgb;
    return mix(mix(c00, c10, fr.x), mix(c01, c11, fr.x), fr.y);
}

// Per-channel spectral sensitivity: smooth Gaussian bands centred at t=0 (red),
// t=0.5 (green), t=1 (blue). Per-channel normalisation in main() keeps a neutral
// input neutral.
fn band(t: f32) -> vec3f {
    let s2 = 2.0 * 0.25 * 0.25;
    let dr = t - 0.0;
    let dg = t - 0.5;
    let db = t - 1.0;
    return vec3f(exp(-dr * dr / s2), exp(-dg * dg / s2), exp(-db * db / s2));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dimsu = textureDimensions(src);
    if gid.x >= dimsu.x || gid.y >= dimsu.y { return; }
    let dims = vec2f(dimsu);
    let p = vec2f(f32(gid.x), f32(gid.y));
    let c = dims * 0.5;                 // matches cv2 getRotationMatrix2D centre
    let d = p - c;

    let n = max(i32(u.samples), 1);
    var acc  = vec3f(0.0);
    var wsum = vec3f(0.0);
    for (var i: i32 = 0; i < n; i++) {
        let t  = select(0.0, f32(i) / f32(n - 1), n > 1);
        // Reciprocal so the magnification matches a real lens (and the legacy
        // cv2.warpAffine, which inverts its matrix): blue (t=1) samples inward,
        // so blue *content* lands at larger radius. Net: light->dark edges going
        // outward fringe blue, dark->light fringe red — as on the film scans.
        let sc = 1.0 / (1.0 + u.scale * t);
        let w  = band(t);
        acc  += sample_edge(c + d * sc, dims) * w;
        wsum += w;
    }
    let outc = acc / max(wsum, vec3f(1e-8));
    textureStore(dst, vec2i(i32(gid.x), i32(gid.y)), vec4f(outc, 1.0));
}
