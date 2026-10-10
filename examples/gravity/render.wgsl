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

// What the vertex shader passes on to the fragment shader, per vertex.
// Between the two, the GPU interpolates every field across the triangle.
struct VertexOutput {
    // Where the vertex lands on screen, in clip space (written by `vs`).
    // Clip space is [xy: (-1..1), z: (0..1), w: 1]
    // In `fs` the same field holds the pixel's position: xy in pixels, z its depth (0..1).
    @builtin(position) position: vec4f,
    // Which body this quad belongs to. `flat`: not interpolated, every pixel gets the
    // value of the triangle's first vertex (integers can't be interpolated).
    @location(0) @interpolate(flat) instance: u32,
}

@vertex
fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let body = bodies[instance];
    let radius = body.size * params.point_scale;

    // A quad that covers the sphere of this radius on screen (see lib/camera.wgsl)
    let position = sphere_billboard(body.pos, radius, quad_corner(vertex));

    return VertexOutput(position, instance);
}

// What the fragment shader returns: a color, and the depth to test and store.
struct FragmentOutput {
    // The pixel's color (rgba), written to color target 0: the window.
    @location(0) color: vec4f,
    // The pixel's depth (0 = near, 1 = far), replacing the depth of the flat quad, so a
    // sphere hides what is behind it at the right place. Closest pixel wins.
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs(in: VertexOutput) -> FragmentOutput {
    let body = bodies[in.instance];
    let radius = body.size * params.point_scale;

    // Cast a ray and find intersection with sphere
    let ray = camera_ray(in.position.xy);
    let t = ray_sphere(ray, body.pos, radius);

    if (t < 0.0) { discard; } // discard if the ray misses the sphere

    let hit = ray.origin + t * ray.dir; // find hit location
    let normal = normalize(hit - body.pos); // find sphere normal at hit location

    // Lit from the camera
    let light = 0.3 + 0.7 * max(dot(normal, -ray.dir), 0.0);

    return FragmentOutput(vec4f(body.color.rgb * light, 1.0), depth_of(hit));
}
