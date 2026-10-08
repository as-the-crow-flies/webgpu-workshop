// Step 2: Write a Shader
//
// Fills the grid with random cells.
#import "types.wgsl"
#import "lib/globals.wgsl"
#import "lib/random.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> cells: array<Cell>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= arrayLength(&cells)) {
        return;
    }
    let alive = rand(i + params.seed * 104729u) < params.density;
    cells[i] = Cell(u32(alive), 0u);
}
