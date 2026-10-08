// Step 2: Write a Shader
//
// Generates the boids: random positions and directions.
#import "types.wgsl"
#import "lib/globals.wgsl"
#import "lib/random.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> boids: array<Boid>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= arrayLength(&boids)) {
        return;
    }
    let r = rand3(i * 3u + params.seed * 7919u);
    let angle = r.z * 6.28318530718;
    boids[i].pos = (r.xy * 2.0 - 1.0) * WORLD;
    boids[i].vel = vec2f(cos(angle), sin(angle)) * params.max_speed * 0.5;
}
