// Step 2: Write a Shader
//
// One quad (6 vertices) per point. The fragment shader decides what it looks
// like: pick one of `square`, `disk` or `sphere` in `fs` at the bottom.
#import "types.wgsl"
#import "lib/globals.wgsl"
#import "lib/math.wgsl"
#import "lib/camera.wgsl"
#import "lib/intersect.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> points: array<Point>;

// What the vertex shader passes on to the fragment shader, per vertex.
// Between the two, the GPU interpolates every field across the triangle.
struct VertexOutput {
    // Where the vertex lands on screen, in clip space (written by `vs`).
    // Clip space is [xy: (-1..1), z: (0..1), w: 1]
    // In `fs` the same field holds the pixel's position: xy in pixels, z its depth (0..1).
    @builtin(position) position: vec4f,
    // The corner of the quad, (-1, -1)..(1, 1); in `fs`, where in the quad this pixel is.
    @location(0) uv: vec2f,
    // Which point this quad belongs to. `flat`: not interpolated, every pixel gets the
    // value of the triangle's first vertex (integers can't be interpolated).
    @location(1) @interpolate(flat) instance: u32,
}

@vertex
fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let point = points[instance];
    let uv = quad_corner(vertex);
    let radius = point.size * params.point_scale;

    // A quad that covers the sphere of this radius on screen (see lib/camera.wgsl)
    let position = sphere_billboard(point.pos, radius, uv);

    return VertexOutput(position, uv, instance);
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
    // Try the others: `return square(in);` or `return disk(in);`
    return sphere(in);
}

// The whole quad, facing the camera.
fn square(in: VertexOutput) -> FragmentOutput {
    let point = points[in.instance];
    return FragmentOutput(point.color, in.position.z);
}

// A flat disk: throw away the corners of the quad.
fn disk(in: VertexOutput) -> FragmentOutput {
    if (length(in.uv) > 1.0) { discard; }

    let point = points[in.instance];
    return FragmentOutput(point.color, in.position.z);
}

// A sphere
fn sphere(in: VertexOutput) -> FragmentOutput {
    let point = points[in.instance];
    let radius = point.size * params.point_scale;

    // Cast a ray and find intersection with sphere
    let ray = camera_ray(in.position.xy);
    let t = ray_sphere(ray, point.pos, radius);

    if (t < 0.0) { discard; } // discard if the ray misses the sphere

    let hit = ray.origin + t * ray.dir; // find hit location
    let normal = normalize(hit - point.pos); // find sphere normal at hit location

    // Compute fake light source from the 'up' direction
    let light = 0.3 + 0.7 * max(dot(normal, vec3(0.0, 1.0, 0.0)), 0.0);

    return FragmentOutput(vec4f(point.color.rgb * light, 1.0), depth_of(hit));
}
