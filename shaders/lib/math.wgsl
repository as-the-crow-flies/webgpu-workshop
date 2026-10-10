// Constants, small helpers and quad generation for instanced drawing.

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

// Map `x` from [a, b] to [c, d].
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return c + (x - a) * (d - c) / (b - a);
}

// Rotate a 2D vector by `angle` radians.
fn rotate2(v: vec2f, angle: f32) -> vec2f {
    let c = cos(angle);
    let s = sin(angle);
    return vec2f(c * v.x - s * v.y, s * v.x + c * v.y);
}

// Corner of a quad made of two triangles: draw 6 vertices per instance and
// call this with the vertex index. Returns (-1, -1) .. (1, 1).
fn quad_corner(vertex_index: u32) -> vec2f {
    var corners = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
        vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
    );
    return corners[vertex_index % 6u];
}

// A triangle that covers the whole screen: draw 3 vertices.
fn fullscreen_triangle(vertex_index: u32) -> vec4f {
    let uv = vec2f(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Per-sample shading with 4x MSAA: a fragment shader that takes `@builtin(sample_index)`
// runs once per sample instead of once per pixel. This is where that sample is, in pixels
// (pass `@builtin(position).xy`), using WebGPU's standard 4x sample positions.
fn sample_position_4x(frag_coord: vec2f, sample_index: u32) -> vec2f {
    var offsets = array<vec2f, 4>(
        vec2f(0.375, 0.125), vec2f(0.875, 0.375), vec2f(0.125, 0.625), vec2f(0.625, 0.875),
    );
    return floor(frag_coord) + offsets[sample_index % 4u];
}
