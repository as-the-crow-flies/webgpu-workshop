// Step 2: Write a Shader
//
// Copy the fixed-point grid into a 3D texture, so rendering can use
// hardware trilinear filtering. One thread per voxel, in 4x4x4 blocks.
#import "types.wgsl"
#import "lib/globals.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> grid: array<u32>;
@group(0) @binding(3) var volume: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (any(id >= vec3u(params.grid))) {
        return;
    }
    let value = f32(grid[voxel_index(id, params.grid)]) / SCALE;
    textureStore(volume, id, vec4f(value, 0.0, 0.0, 0.0));
}
