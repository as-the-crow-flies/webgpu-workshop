// Step 2: Write a Shader
//
// One quad (6 vertices) per body, drawn as a ray-traced sphere (as in examples/template).
#import "types.wgsl"
#import "lib/globals.wgsl"
#import "lib/math.wgsl"
#import "lib/camera.wgsl"
#import "lib/intersect.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> bodies: array<Body>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) @interpolate(flat) center: vec3f, // the body, in world space
    @location(2) @interpolate(flat) radius: f32,   // the size of the body
}

@vertex
fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let body = bodies[instance];
    let radius = body.size * params.point_scale;

    var out: VertexOutput;
    // A quad that covers the sphere of this radius on screen (see lib/camera.wgsl).
    out.position = sphere_billboard(body.pos, radius, quad_corner(vertex));
    out.color = body.color;
    out.center = body.pos;
    out.radius = radius;
    return out;
}

// What the fragment shader returns: a color, and the depth to test and store.
struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs(in: VertexOutput) -> FragmentOutput {
    // Shoot a ray from the camera through this pixel and intersect it with the sphere.
    let ray = camera_ray(in.position.xy);
    let t = ray_sphere(ray, in.center, in.radius);
    if (t < 0.0) {
        discard; // the ray misses the sphere
    }
    let hit = ray.origin + t * ray.dir;
    let normal = normalize(hit - in.center);
    let light = 0.3 + 0.7 * max(dot(normal, -ray.dir), 0.0); // lit from the camera
    return FragmentOutput(vec4f(in.color.rgb * light, 1.0), depth_of(hit));
}
