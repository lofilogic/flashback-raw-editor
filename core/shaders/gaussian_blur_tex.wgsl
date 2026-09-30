// Separable blur on textures: main_h, then main_v. Clamp-to-edge. The kernel
// buffer's length is the tap count; it's also used for the exponential blur.

@group(0) @binding(0) var                  src:    texture_2d<f32>;
@group(0) @binding(1) var<storage, read>   kernel: array<f32>;
@group(0) @binding(2) var                  dst:    texture_storage_2d<rgba32float, write>;

fn blur(p: vec2i, dir: vec2i) -> vec4f {
    let dims = vec2i(textureDimensions(src));
    let n    = i32(arrayLength(&kernel));
    let half = n / 2;
    var acc  = vec4f(0.0);
    for (var k: i32 = 0; k < n; k++) {
        let s = clamp(p + dir * (k - half), vec2i(0), dims - vec2i(1));
        acc += textureLoad(src, s, 0) * kernel[k];
    }
    return acc;
}

@compute @workgroup_size(8, 8)
fn main_h(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(src);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    textureStore(dst, p, blur(p, vec2i(1, 0)));
}

@compute @workgroup_size(8, 8)
fn main_v(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(src);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let p = vec2i(i32(gid.x), i32(gid.y));
    textureStore(dst, p, blur(p, vec2i(0, 1)));
}
