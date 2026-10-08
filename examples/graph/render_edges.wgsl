// Step 2: Write a Shader
//
// Edges: one quad per edge, as a line of `line_width` pixels.
#import "types.wgsl"
#import "lib/globals.wgsl"
#import "lib/math.wgsl"
#import "lib/sdf2d.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> edges: array<Edge>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) a: vec2f,   // in pixels
    @location(1) @interpolate(flat) b: vec2f,
}

// Clip position -> (pixel x, pixel y, depth).
fn to_pixels(clip: vec4f) -> vec3f {
    let ndc = clip.xyz / clip.w;
    return vec3f(vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * globals.resolution, ndc.z);
}

@vertex
fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let edge = edges[instance];
    let a = to_pixels(globals.view_proj * vec4f(nodes[edge.a].pos, 1.0));
    let b = to_pixels(globals.view_proj * vec4f(nodes[edge.b].pos, 1.0));

    let along = normalize(b.xy - a.xy + vec2f(1e-4, 0.0));
    let across = vec2f(-along.y, along.x);
    let r = 0.5 * params.line_width + 1.0;
    let corner = quad_corner(vertex);
    let f = corner.x * 0.5 + 0.5;
    let pixel = mix(a.xy, b.xy, f) + along * corner.x * r + across * corner.y * r;

    var out: VertexOutput;
    let ndc = vec2f(pixel.x / globals.resolution.x * 2.0 - 1.0, 1.0 - pixel.y / globals.resolution.y * 2.0);
    out.position = vec4f(ndc, mix(a.z, b.z, f), 1.0);
    out.a = a.xy;
    out.b = b.xy;
    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
    // Distance (in pixels) from this pixel to the line; outside the line width: discard.
    if (sd_segment(in.position.xy, in.a, in.b) > 0.5 * params.line_width) {
        discard;
    }
    return vec4f(vec3f(0.4), 1.0);
}
