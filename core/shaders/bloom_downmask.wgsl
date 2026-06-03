// Bloom stage 1 (texture-resident): area-downsample + highlight mask.
//
// Twin of the first half of effects.apply_bloom (linear path). Each output texel
// area-averages its src block (~4x4), then keeps only the highlights: luma is
// taken with ACEScg/AP1 weights, ACEScct-encoded, and turned into a soft mask
// above `threshold`; the masked, downsampled colour is written out for blurring.
//
//   small   = mean(src over block)
//   luma    = dot(small, AP1)
//   mask    = clamp((encode(luma) - threshold) / max(0.01, 1 - threshold), 0, 1)
//   out     = small * mask

struct U {
    threshold: f32,
    _p0: f32, _p1: f32, _p2: f32,
}

@group(0) @binding(0) var          src: texture_2d<f32>;
@group(0) @binding(1) var          dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> u:   U;

const CUT: f32 = 0.0078125;
const A:   f32 = 10.5402377416545;
const B:   f32 = 0.0729055341958355;

fn encode(vin: f32) -> f32 {
    let v = max(vin, 1e-10);
    if v <= CUT { return A * v + B; }
    return (log2(v) + 9.72) / 17.52;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let sdim = textureDimensions(dst);          // small (downsampled) size
    if gid.x >= sdim.x || gid.y >= sdim.y { return; }
    let fdim = textureDimensions(src);          // full size
    let sx = f32(fdim.x) / f32(sdim.x);
    let sy = f32(fdim.y) / f32(sdim.y);

    let x0 = i32(floor(f32(gid.x) * sx));
    let x1 = max(x0 + 1, i32(floor(f32(gid.x + 1u) * sx)));
    let y0 = i32(floor(f32(gid.y) * sy));
    let y1 = max(y0 + 1, i32(floor(f32(gid.y + 1u) * sy)));
    let xe = min(x1, i32(fdim.x));
    let ye = min(y1, i32(fdim.y));

    var sum = vec3f(0.0);
    var cnt = 0.0;
    for (var y = y0; y < ye; y++) {
        for (var x = x0; x < xe; x++) {
            sum += textureLoad(src, vec2i(x, y), 0).rgb;
            cnt += 1.0;
        }
    }
    let small = sum / max(cnt, 1.0);

    let luma = dot(small, vec3f(0.2722, 0.6741, 0.0537));
    let mask = clamp((encode(luma) - u.threshold) / max(0.01, 1.0 - u.threshold), 0.0, 1.0);
    textureStore(dst, vec2i(i32(gid.x), i32(gid.y)), vec4f(small * mask, 1.0));
}
