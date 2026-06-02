// Halation highlight mask: sigmoid of the ACEScct-encoded luma above a
// threshold. Matches the numpy mask in effects._halation_glow:
//   gray = max(r,g,b);  m = 1/(1+exp(-k*(encode(gray) - threshold)))

struct U { threshold: f32, k: f32, _p0: f32, _p1: f32, }

@group(0) @binding(0) var                  src: texture_2d<f32>;
@group(0) @binding(1) var                  dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform>         u:   U;

const CUT_ENCODE: f32 = 0.0078125;
const A: f32 = 10.5402377416545;
const B: f32 = 0.0729055341958355;

fn encode(vin: f32) -> f32 {
    let v = max(vin, 1e-10);
    if v <= CUT_ENCODE { return A * v + B; }
    return (log2(v) + 9.72) / 17.52;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(src);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    let c = textureLoad(src, p, 0).rgb;
    let gray = max(max(c.r, c.g), c.b);
    let m = 1.0 / (1.0 + exp(-u.k * (encode(gray) - u.threshold)));
    textureStore(dst, p, vec4f(m, m, m, 1.0));
}
