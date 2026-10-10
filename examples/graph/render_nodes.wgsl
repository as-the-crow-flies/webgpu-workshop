// Step 2: Write a Shader
//
// Nodes: a camera-facing quad per node, with an exact ray-sphere intersection
// in the fragment shader (a sphere "impostor"). The fragment shader runs once per
// MSAA sample, so the sphere's outline is anti-aliased.
#import "types.wgsl"
#import "lib/globals.wgsl"
#import "lib/math.wgsl"
#import "lib/camera.wgsl"
#import "lib/intersect.wgsl"
#import "lib/colormap.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> pick: u32;
@group(0) @binding(3) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) node: u32,
}

@vertex
fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let node = nodes[instance];
    var out: VertexOutput;
    // A quad that covers the sphere on screen (see lib/camera.wgsl).
    out.position = sphere_billboard(node.pos, node.radius * params.node_radius, quad_corner(vertex));
    out.node = instance;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs(in: VertexOutput, @builtin(sample_index) sample: u32) -> FragmentOutput {
    let node = nodes[in.node];
    // Shoot a ray through this sample (not the pixel center).
    let ray = camera_ray(sample_position_4x(in.position.xy, sample));

    let t = ray_sphere(ray, node.pos, node.radius * params.node_radius);
    if (t < 0.0) {
        discard; // the ray misses the sphere
    }
    let hit = ray.origin + t * ray.dir;
    let normal = normalize(hit - node.pos);

    var color = category_color(node.group);
    if (picked_node(pick) == i32(in.node)) {
        color = vec3f(1.0);
    }
    let light = 0.3 + 0.7 * max(dot(normal, -ray.dir), 0.0);

    var out: FragmentOutput;
    out.color = vec4f(color * light, 1.0);
    out.depth = depth_of(hit);
    return out;
}
