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

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,                         // (-1, -1) .. (1, 1) across the quad
    @location(1) color: vec4f,
    @location(2) @interpolate(flat) center: vec3f,  // the point, in world space
    @location(3) @interpolate(flat) radius: f32,    // the size of the point
}

@vertex
fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let point = points[instance];
    let corner = quad_corner(vertex);
    let radius = point.size * params.point_scale;

    var out: VertexOutput;
    // A quad that covers the sphere of this radius on screen (see lib/camera.wgsl).
    out.position = sphere_billboard(point.pos, radius, corner);
    out.uv = corner;
    out.color = point.color;
    out.center = point.pos;
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
    // Try the others: `return square(in);` or `return disk(in);`
    return sphere(in);
}

// The whole quad, facing the camera.
fn square(in: VertexOutput) -> FragmentOutput {
    return FragmentOutput(in.color, in.position.z);
}

// A flat disk: throw away the corners of the quad.
fn disk(in: VertexOutput) -> FragmentOutput {
    if (length(in.uv) > 1.0) {
        discard;
    }
    return FragmentOutput(in.color, in.position.z);
}

// A sphere
fn sphere(in: VertexOutput) -> FragmentOutput {
    let ray = camera_ray(in.position.xy);
    let t = ray_sphere(ray, in.center, in.radius);
    if (t < 0.0) {
        discard; // the ray misses the sphere
    }
    let hit = ray.origin + t * ray.dir;
    let normal = normalize(hit - in.center);
    let up = vec3f(globals.view[0].y, globals.view[1].y, globals.view[2].y);
    let light = 0.3 + 0.7 * max(dot(normal, up), 0.0);
    return FragmentOutput(vec4f(in.color.rgb * light, 1.0), depth_of(hit));
}
