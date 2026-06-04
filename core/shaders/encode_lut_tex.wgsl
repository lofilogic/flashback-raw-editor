// Fused ACEScct-encode + tetrahedral LUT, texture-resident (GPU V2 Phase 3).
//
// One pass replacing the encode_tex -> lut_tex chain: encode each channel to
// ACEScct log, then run the LUT's Sakamoto tetrahedral interpolation on the
// encoded value, writing the display result. Saves one full-image rgba32float
// texture round-trip and one dispatch per render (encode + LUT are the only
// always-adjacent point-op pair in the default render). Math is identical to
// encode_tex.wgsl `encode` + lut_tex.wgsl `tetrahedral`, kept in sync by hand.

struct Uniforms {
    lut_size: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var                src:      texture_2d<f32>;
@group(0) @binding(1) var<storage, read> lut_data: array<f32>;
@group(0) @binding(2) var                dst:      texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform>       u:        Uniforms;

const CUT_ENCODE: f32 = 0.0078125;
const A: f32 = 10.5402377416545;
const B: f32 = 0.0729055341958355;

fn encode(vin: f32) -> f32 {
    let v = max(vin, 1e-10);
    if v <= CUT_ENCODE {
        return A * v + B;
    }
    return (log2(v) + 9.72) / 17.52;
}

fn lut3(r: u32, g: u32, b: u32) -> vec3f {
    let base = ((r * u.lut_size + g) * u.lut_size + b) * 3u;
    return vec3f(lut_data[base], lut_data[base + 1u], lut_data[base + 2u]);
}

fn tetrahedral(rgb: vec3f) -> vec3f {
    let scaled = clamp(rgb, vec3f(0.0), vec3f(1.0)) * f32(u.lut_size - 1u);
    let r0 = min(u32(scaled.x), u.lut_size - 2u);
    let g0 = min(u32(scaled.y), u.lut_size - 2u);
    let b0 = min(u32(scaled.z), u.lut_size - 2u);
    let rx = scaled.x - f32(r0);
    let gx = scaled.y - f32(g0);
    let bx = scaled.z - f32(b0);

    let c000 = lut3(r0,     g0,     b0    );
    let c111 = lut3(r0 + 1u, g0 + 1u, b0 + 1u);

    if rx >= gx && gx >= bx {
        let c100 = lut3(r0 + 1u, g0,     b0    );
        let c110 = lut3(r0 + 1u, g0 + 1u, b0    );
        return (1.0 - rx) * c000 + (rx - gx) * c100 + (gx - bx) * c110 + bx * c111;
    } else if rx >= bx && bx > gx {
        let c100 = lut3(r0 + 1u, g0,     b0    );
        let c101 = lut3(r0 + 1u, g0,     b0 + 1u);
        return (1.0 - rx) * c000 + (rx - bx) * c100 + (bx - gx) * c101 + gx * c111;
    } else if gx > rx && rx >= bx {
        let c010 = lut3(r0,      g0 + 1u, b0    );
        let c110 = lut3(r0 + 1u, g0 + 1u, b0    );
        return (1.0 - gx) * c000 + (gx - rx) * c010 + (rx - bx) * c110 + bx * c111;
    } else if bx > rx && rx >= gx {
        let c001 = lut3(r0,      g0,     b0 + 1u);
        let c101 = lut3(r0 + 1u, g0,     b0 + 1u);
        return (1.0 - bx) * c000 + (bx - rx) * c001 + (rx - gx) * c101 + gx * c111;
    } else if gx >= bx && bx > rx {
        let c010 = lut3(r0,      g0 + 1u, b0    );
        let c011 = lut3(r0,      g0 + 1u, b0 + 1u);
        return (1.0 - gx) * c000 + (gx - bx) * c010 + (bx - rx) * c011 + rx * c111;
    } else {
        let c001 = lut3(r0,      g0,     b0 + 1u);
        let c011 = lut3(r0,      g0 + 1u, b0 + 1u);
        return (1.0 - bx) * c000 + (bx - gx) * c001 + (gx - rx) * c011 + rx * c111;
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(src);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let p = vec2i(i32(gid.x), i32(gid.y));
    let c = textureLoad(src, p, 0).rgb;
    let encoded = vec3f(encode(c.r), encode(c.g), encode(c.b));
    textureStore(dst, p, vec4f(tetrahedral(encoded), 1.0));
}
