// Separable Gaussian blur, texture-resident twin of gaussian_blur.wgsl.
//
// Run main_h (horizontal) then main_v (vertical). Reads/writes rgba32float
// textures so blurs chain with neighbouring resident stages without a readback.
// Boundary handling is clamp-to-edge, matching the buffer version. Kernel is a
// pre-normalised 1-D Gaussian storage buffer; its length is the tap count.
//
// All four channels are blurred uniformly: for 3-channel data alpha is a
// constant (1.0) and blurs to 1.0; for single-channel masks the value is
// replicated across channels, so .r carries the result either way.

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
