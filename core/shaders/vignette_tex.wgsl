// Cosine vignette with a cool-edge tint, texture-resident.
//
// Twin of effects.apply_vignette (runs on linear ACEScg, pre-LUT). All channels
// share the same `dark` falloff; per-channel edge offsets push the periphery
// slightly cooler (red darkens a touch more, blue a touch less). Normalised
// coords match numpy linspace(-1, 1, n): pixel i -> -1 + 2i/(n-1).
//
//   r_norm  = clamp(length(xy) / sqrt(2), 0, 1)
//   falloff = pow(0.5*(1+cos(pi*r_norm)), feather)
//   dark    = 1 - strength*(1-falloff)
//   edge    = 1 - falloff
//   r = max(0, in.r * (dark - color_shift*edge))
//   g = max(0, in.g *  dark)
//   b = max(0, in.b * (dark + color_shift*0.4*edge))

struct U {
    strength:    f32,
    color_shift: f32,
    feather:     f32,
    _p:          f32,
}

@group(0) @binding(0) var          src: texture_2d<f32>;
@group(0) @binding(1) var          dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> u:   U;

const PI: f32 = 3.14159265358979;
const INV_SQRT2: f32 = 0.70710678118655;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dimsu = textureDimensions(src);
    if gid.x >= dimsu.x || gid.y >= dimsu.y { return; }
    let dims = vec2f(dimsu);
    let denom = max(dims - vec2f(1.0), vec2f(1.0));
    let xy = vec2f(f32(gid.x), f32(gid.y)) / denom * 2.0 - vec2f(1.0);

    let r_norm  = clamp(length(xy) * INV_SQRT2, 0.0, 1.0);
    let falloff = pow(0.5 * (1.0 + cos(PI * r_norm)), u.feather);
    let dark    = 1.0 - u.strength * (1.0 - falloff);
    let edge    = 1.0 - falloff;

    let pi = vec2i(i32(gid.x), i32(gid.y));
    let c = textureLoad(src, pi, 0).rgb;
    let outc = max(vec3f(0.0), c * vec3f(
        dark - u.color_shift * edge,
        dark,
        dark + u.color_shift * 0.4 * edge,
    ));
    textureStore(dst, pi, vec4f(outc, 1.0));
}
