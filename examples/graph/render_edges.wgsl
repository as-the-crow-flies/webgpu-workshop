// Step 2: Write a Shader
//
// Edges: one quad per edge, drawn as a ray-traced cylinder (an "impostor", see lib/intersect.wgsl).
// The fragment shader runs once per MSAA sample, so the cylinder's outline is anti-aliased.
#import "types.wgsl"
#import "lib/globals.wgsl"
#import "lib/math.wgsl"
#import "lib/camera.wgsl"
#import "lib/intersect.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> edges: array<Edge>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    // The two ends of the cylinder (the centers of the two nodes), in world space.
    @location(0) @interpolate(flat) a: vec3f,
    @location(1) @interpolate(flat) b: vec3f,
}

@vertex
fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let edge = edges[instance];
    let a = nodes[edge.a].pos;
    let b = nodes[edge.b].pos;
    // A quad that covers the cylinder on screen (see lib/intersect.wgsl).
    let position = cylinder_billboard(a, b, params.edge_radius, quad_corner(vertex));
    return VertexOutput(position, a, b);
}

// What the fragment shader returns: a color, and the depth to test and store.
struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs(in: VertexOutput, @builtin(sample_index) sample: u32) -> FragmentOutput {
    // Shoot a ray through this sample (not the pixel center) and intersect it with the cylinder.
    let ray = camera_ray(sample_position_4x(in.position.xy, sample));
    let hit = ray_cylinder(ray, in.a, in.b, params.edge_radius); // (t, normal)
    if (hit.x < 0.0) {
        discard; // the ray misses the cylinder
    }
    let point = ray.origin + hit.x * ray.dir;
    let light = 0.3 + 0.7 * max(dot(hit.yzw, -ray.dir), 0.0); // lit from the camera
    return FragmentOutput(vec4f(vec3f(0.5) * light, 1.0), depth_of(point));
}
