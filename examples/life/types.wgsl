// The data layouts of this example, shared by all of its shaders.

// One cell of the grid. The grid is stored row by row: cell (x, y) is at index y * width + x.
struct Cell {
    alive: u32,   // 1 or 0
    age: u32,     // number of steps the cell has been alive (for coloring)
}

// Values controlled by main.rs. `Params` there must have the same layout.
struct Params {
    width: u32,
    height: u32,
    density: f32,   // fraction of cells alive after "Reset"
    seed: u32,
    brush: f32,     // radius (in cells) of the mouse brush
}

// The grid covers the square [-1, 1]^2 in world space.
fn cell_index(cell: vec2i, params: Params) -> u32 {
    // Wrap around the edges.
    let size = vec2i(i32(params.width), i32(params.height));
    let c = (cell % size + size) % size;
    return u32(c.y) * params.width + u32(c.x);
}
