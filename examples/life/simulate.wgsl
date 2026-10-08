// Step 2: Write a Shader
//
// One step of Conway's Game of Life, plus painting with the mouse.
// Runs one thread per cell, in 8x8 tiles.
#import "types.wgsl"
#import "lib/globals.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> ping: array<Cell>;
@group(0) @binding(3) var<storage, read_write> pong: array<Cell>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let cell = vec2i(id.xy);
    let me = ping[cell_index(cell, params)];

    // Count the living neighbours.
    var neighbours = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                neighbours += ping[cell_index(cell + vec2i(dx, dy), params)].alive;
            }
        }
    }

    // The rules: a living cell survives with 2 or 3 neighbours; a dead cell comes alive with 3.
    var alive = (me.alive == 1u && (neighbours == 2u || neighbours == 3u)) || (me.alive == 0u && neighbours == 3u);

    // Paint living cells with the left mouse button.
    if (mouse_down(MOUSE_LEFT)) {
        let mouse = mouse_on_plane(0.0).xy;   // world position under the mouse
        let mouse_cell = (mouse * 0.5 + 0.5) * vec2f(f32(params.width), f32(params.height));
        if (distance(vec2f(cell) + 0.5, mouse_cell) < params.brush) {
            alive = true;
        }
    }

    let age = select(0u, me.age + 1u, alive);
    pong[cell_index(cell, params)] = Cell(u32(alive), age);
}
