// out = img + (img - blurred) * strength
// Not clamped; the render clips once at the end.

struct Uniforms {
    strength: f32,
    _pad0:    f32,
    _pad1:    f32,
    _pad2:    f32,
}

@group(0) @binding(0) var               img:     texture_2d<f32>;
@group(0) @binding(1) var               blurred: texture_2d<f32>;
@group(0) @binding(2) var               dst:     texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform>      u:       Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(img);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    let a = textureLoad(img, p, 0);
    let b = textureLoad(blurred, p, 0);
    let o = a + (a - b) * u.strength;
    textureStore(dst, p, vec4f(o.rgb, 1.0));
}
