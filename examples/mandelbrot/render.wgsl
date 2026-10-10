// Step 2: Write a Shader
//
// Draws the Mandelbrot set as one quad covering the whole screen; the fragment
// shader iterates z = z^2 + c for the point c under each pixel.
#import "types.wgsl"
#import "lib/globals.wgsl"
#import "lib/math.wgsl"
#import "lib/colormap.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) world: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vertex: u32) -> VertexOutput {
    // The corners of the screen, in clip space: (-1, -1) .. (1, 1).
    let corner = quad_corner(vertex);
    var out: VertexOutput;
    out.position = vec4f(corner, 0.0, 1.0);
    // The world point at that corner of the screen (the reverse of view_proj).
    let world = globals.inv_view_proj * out.position;
    out.world = world.xy / world.w;
    return out;
}

// How many steps of z = z^2 + c it takes for z to escape, or -1 if it never does
// (c is inside the set). The count is smoothed, so the colors have no bands.
fn mandelbrot(c: vec2f) -> f32 {
    var z = vec2f(0.0);
    for (var i = 0u; i < params.iterations; i++) {
        // Complex numbers as vec2f(real, imaginary).
        z = vec2f(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        if (dot(z, z) > 256.0) {
            return f32(i) + 1.0 - log2(log2(dot(z, z)) * 0.5);
        }
    }
    return -1.0;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
    let steps = mandelbrot(in.world);
    if (steps < 0.0) {
        return vec4f(0.0, 0.0, 0.0, 1.0); // inside the set
    }
    // cos() cycles through the colormap.
    return vec4f(magma(0.5 + 0.5 * cos(steps * 0.15)), 1.0);
}
