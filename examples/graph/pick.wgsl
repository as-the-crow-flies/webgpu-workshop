// Step 2: Write a Shader
//
// Finds the node under the mouse. Runs once when the left button goes down.
// Every node that the mouse ray hits proposes itself with atomicMin; the
// node closest to the camera wins.
#import "types.wgsl"
#import "lib/globals.wgsl"
#import "lib/intersect.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read_write> pick: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= arrayLength(&nodes)) {
        return;
    }
    let node = nodes[i];
    // A slightly larger sphere makes small nodes easier to grab.
    let t = ray_sphere(mouse_ray(), node.pos, node.radius * 1.5);
    if (t > 0.0) {
        // For positive floats, the bits sort in the same order as the values,
        // so the top 16 bits of `t` work as a (coarse) depth.
        let key = (bitcast<u32>(t) & 0xffff0000u) | (i & 0xffffu);
        atomicMin(&pick, key);
    }
}
