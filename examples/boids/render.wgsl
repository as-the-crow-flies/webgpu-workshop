// Step 2: Write a Shader
//
// Each boid is one triangle (3 vertices), pointing where it flies.
#import "types.wgsl"
#import "lib/globals.wgsl"
#import "lib/colormap.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> boids: array<Boid>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
}

@vertex
fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let boid = boids[instance];
    // A triangle pointing along +x ...
    var shape = array<vec2f, 3>(vec2f(1.0, 0.0), vec2f(-0.6, 0.5), vec2f(-0.6, -0.5));
    // ... rotated to point along the velocity.
    let forward = normalize(boid.vel + vec2f(1e-6, 0.0));
    let side = vec2f(-forward.y, forward.x);
    let p = shape[vertex];
    let world = boid.pos + (forward * p.x + side * p.y) * params.size;

    var out: VertexOutput;
    out.position = globals.view_proj * vec4f(world, 0.0, 1.0);
    // Color by heading.
    out.color = plasma(0.5 + 0.5 * atan2(forward.y, forward.x) / 3.14159265);
    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
