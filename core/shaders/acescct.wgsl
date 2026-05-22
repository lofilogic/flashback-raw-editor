// ACEScct piecewise log encode/decode.
// Two entry points: encode (linear→ACEScct) and decode (ACEScct→linear).
// Operates element-wise on a flat f32 array (image flattened to 1D).

@group(0) @binding(0) var<storage, read>       data_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> data_out: array<f32>;

const CUT_ENCODE: f32 = 0.0078125;
const CUT_DECODE: f32 = 0.155251141552511;
const A: f32 = 10.5402377416545;
const B: f32 = 0.0729055341958355;

fn decode(v: f32) -> f32 {
    if v < CUT_DECODE {
        return (v - B) / A;
    }
    return pow(2.0, v * 17.52 - 9.72);
}

fn encode(v: f32) -> f32 {
    if v <= CUT_ENCODE {
        return A * v + B;
    }
    return (log2(max(v, 1e-10)) + 9.72) / 17.52;
}

@compute @workgroup_size(256)
fn main_decode(@builtin(global_invocation_id) id: vec3u) {
    let i = id.y * 16776960u + id.x; // 65535 * 256 — supports 2D dispatch for large images
    if i >= arrayLength(&data_in) { return; }
    data_out[i] = decode(data_in[i]);
}

@compute @workgroup_size(256)
fn main_encode(@builtin(global_invocation_id) id: vec3u) {
    let i = id.y * 16776960u + id.x;
    if i >= arrayLength(&data_in) { return; }
    data_out[i] = encode(data_in[i]);
}
