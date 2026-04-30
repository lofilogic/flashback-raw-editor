// Screen blend (used for halation) and unsharp mask.
// Two entry points operating on flat f32 arrays.

struct UnsharpUniforms {
    strength: f32,
    _pad0:    f32,
    _pad1:    f32,
    _pad2:    f32,
}

@group(0) @binding(0) var<storage, read>       a:       array<f32>;
@group(0) @binding(1) var<storage, read>       b:       array<f32>;
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;
@group(0) @binding(3) var<uniform>             u:       UnsharpUniforms;

// Screen blend: 1 - (1-a)*(1-b)
@compute @workgroup_size(256)
fn main_screen(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= arrayLength(&a) { return; }
    output[i] = 1.0 - (1.0 - a[i]) * (1.0 - b[i]);
}

// Unsharp mask: image + (image - blurred) * strength
@compute @workgroup_size(256)
fn main_unsharp(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= arrayLength(&a) { return; }
    output[i] = a[i] + (a[i] - b[i]) * u.strength;
}
