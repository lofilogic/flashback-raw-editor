// Halation highlight extraction. Matches effects._halation_glow's pre-blur
// highlight layer: red = img.r * mask, green = img.g * mask * 0.2, blue = 0
// (the blue glow channel is left at zero, as on real film halation).

@group(0) @binding(0) var img:  texture_2d<f32>;
@group(0) @binding(1) var mask: texture_2d<f32>;
@group(0) @binding(2) var dst:  texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(img);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    let c = textureLoad(img, p, 0).rgb;
    let m = textureLoad(mask, p, 0).r;
    textureStore(dst, p, vec4f(c.r * m, c.g * m * 0.2, 0.0, 1.0));
}
