// Step 2: Write a Shader
//
// Draws the grid as one quad covering [-1, 1]^2; the fragment shader looks up the cell.
#import "types.wgsl"
#import "lib/globals.wgsl"
#import "lib/math.wgsl"
#import "lib/colormap.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> cells: array<Cell>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) world: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vertex: u32) -> VertexOutput {
    let corner = quad_corner(vertex);
    var out: VertexOutput;
    out.position = globals.view_proj * vec4f(corner, 0.0, 1.0);
    out.world = corner;
    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
    let size = vec2f(f32(params.width), f32(params.height));
    let cell = vec2i(floor((in.world * 0.5 + 0.5) * size));
    let c = cells[cell_index(cell, params)];
    if (c.alive == 0u) {
        return vec4f(0.03, 0.03, 0.05, 1.0);
    }
    // Young cells are bright, old cells fade to purple.
    return vec4f(magma(1.0 - min(f32(c.age) / 60.0, 0.8)), 1.0);
}
