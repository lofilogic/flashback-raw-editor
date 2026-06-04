// Area-average downsample, texture-resident (rgba32float).
//
// Each output texel averages its source block; dst is smaller than src. Used to
// shrink a layer before a wide blur (halation glow) so the expensive blur runs
// on far fewer pixels — the glow is low-frequency, so the lost detail is
// imperceptible. Same area-average as bloom_downmask.wgsl, without the mask.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;

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
    textureStore(dst, vec2i(i32(gid.x), i32(gid.y)), vec4f(sum / max(cnt, 1.0), 1.0));
}
