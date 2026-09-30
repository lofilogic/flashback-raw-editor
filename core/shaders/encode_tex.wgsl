// ACEScct encode on textures. Clamps to >= 1e-10 like kernels.acescct_encode.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;

const CUT_ENCODE: f32 = 0.0078125;
const A: f32 = 10.5402377416545;
const B: f32 = 0.0729055341958355;

// NaN -> 0, Inf -> finite. See acescct.wgsl.
fn sanitize(v: f32) -> f32 {
    let n = select(v, 0.0, v != v);
    return clamp(n, -3.4e38, 3.4e38);
}

fn encode(vin: f32) -> f32 {
    let v = max(sanitize(vin), 1e-10);
    if v <= CUT_ENCODE {
        return A * v + B;
    }
    return (log2(v) + 9.72) / 17.52;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(src);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let p = vec2i(i32(gid.x), i32(gid.y));
    let c = textureLoad(src, p, 0);
    textureStore(dst, p, vec4f(encode(c.r), encode(c.g), encode(c.b), 1.0));
}
