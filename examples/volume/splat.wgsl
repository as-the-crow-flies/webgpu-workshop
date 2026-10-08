// Step 2: Write a Shader
//
// Add every particle's weight to the voxel it is in.
// Many particles can land in the same voxel at the same time, hence atomicAdd.
#import "types.wgsl"
#import "lib/globals.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> particles: array<Particle>;
@group(0) @binding(3) var<storage, read_write> grid: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= arrayLength(&particles)) {
        return;
    }
    let particle = particles[i];
    let voxel = vec3u(clamp(particle.pos, vec3f(0.0), vec3f(0.9999)) * f32(params.grid));
    atomicAdd(&grid[voxel_index(voxel, params.grid)], u32(particle.weight * params.splat * SCALE + 0.5));
}
