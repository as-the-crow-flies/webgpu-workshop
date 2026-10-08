// Step 2: Write a Shader
//
// Draws every series as a polyline: one quad per line segment, with an
// anti-aliased signed distance to the segment computed in pixels.
#import "types.wgsl"
#import "lib/globals.wgsl"
#import "lib/math.wgsl"
#import "lib/sdf2d.wgsl"
#import "lib/aa.wgsl"
#import "lib/colormap.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> samples: array<Sample>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) a: vec2f,   // segment start, in pixels
    @location(1) @interpolate(flat) b: vec2f,   // segment end, in pixels
    @location(2) @interpolate(flat) series: u32,
    @location(3) world_x: f32,
}

// World position -> pixel position (y down, like @builtin(position)).
fn to_pixels(world: vec2f) -> vec2f {
    let clip = globals.view_proj * vec4f(world, 0.0, 1.0);
    let ndc = clip.xy / clip.w;
    return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * globals.resolution;
}

@vertex
fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    // Instance i is segment t of series s.
    let segments = params.samples - 1u;
    let s = instance / segments;
    let t = instance % segments;
    let p0 = sample_position(s, t, samples[s * params.samples + t].value, params);
    let p1 = sample_position(s, t + 1u, samples[s * params.samples + t + 1u].value, params);

    // A quad around the segment, in pixels, a bit larger than the line.
    let a = to_pixels(p0);
    let b = to_pixels(p1);
    let along = normalize(b - a + vec2f(1e-4, 0.0));
    let across = vec2f(-along.y, along.x);
    let r = 0.5 * params.line_width + 1.0;
    let corner = quad_corner(vertex);
    let center = mix(a, b, corner.x * 0.5 + 0.5);
    let pixel = center + along * corner.x * r + across * corner.y * r;

    var out: VertexOutput;
    let ndc = vec2f(pixel.x / globals.resolution.x * 2.0 - 1.0, 1.0 - pixel.y / globals.resolution.y * 2.0);
    out.position = vec4f(ndc, 0.0, 1.0);
    out.a = a;
    out.b = b;
    out.series = s;
    out.world_x = mix(p0.x, p1.x, corner.x * 0.5 + 0.5);
    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
    let d = sd_segment(in.position.xy, in.a, in.b) - 0.5 * params.line_width;
    let alpha = aa_fill_px(d);

    var color = category_color(in.series);
    // A playhead that sweeps over the data every 10 seconds (uses globals.time).
    let playhead = -2.0 + 4.0 * fract(globals.time / 10.0);
    color *= 0.55 + 0.45 * smoothstep(0.3, 0.0, abs(in.world_x - playhead));
    if (i32(in.series) == params.hovered) {
        color = vec3f(1.0);
    }
    return vec4f(color * alpha, alpha);
}
