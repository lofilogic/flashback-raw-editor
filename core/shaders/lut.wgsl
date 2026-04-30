// Tetrahedral 3D LUT interpolation (Sakamoto algorithm).
//
// Tetrahedral uses 4 cell corners instead of trilinear's 8, and divides
// each cube cell into 6 tetrahedra. This eliminates the color discontinuities
// at cell boundaries that trilinear produces, and is the algorithm used by
// DaVinci Resolve and other professional color tools.
//
// LUT layout: flat array[lut_size * lut_size * lut_size * 3]
//   index = ((r_idx * lut_size + g_idx) * lut_size + b_idx) * 3 + channel

struct Uniforms {
    width:    u32,
    height:   u32,
    lut_size: u32,
    _pad:     u32,
}

@group(0) @binding(0) var<storage, read>       img_in:   array<f32>;
@group(0) @binding(1) var<storage, read>       lut_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> img_out:  array<f32>;
@group(0) @binding(3) var<uniform>             u:        Uniforms;

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

    // Sakamoto: sort rx, gx, bx to pick the enclosing tetrahedron (6 cases)
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
        // bx > gx > rx
        let c001 = lut3(r0,      g0,     b0 + 1u);
        let c011 = lut3(r0,      g0 + 1u, b0 + 1u);
        return (1.0 - bx) * c000 + (bx - gx) * c001 + (gx - rx) * c011 + rx * c111;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let pixel = id.x;
    if pixel >= u.width * u.height { return; }
    let base = pixel * 3u;
    let result = tetrahedral(vec3f(img_in[base], img_in[base + 1u], img_in[base + 2u]));
    img_out[base]       = result.x;
    img_out[base + 1u]  = result.y;
    img_out[base + 2u]  = result.z;
}
