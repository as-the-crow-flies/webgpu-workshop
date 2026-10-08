// The data layouts of this example, shared by all of its shaders.

// One sample. timeseries.npy is a plain float32 array of shape (series, samples),
// so every float is one `Sample`; sample t of series s is at index s * samples + t.
struct Sample {
    value: f32,
}

// Values controlled by main.rs. `Params` there must have the same layout.
struct Params {
    series: u32,      // number of series (rows of the .npy array)
    samples: u32,     // samples per series (columns)
    line_width: f32,  // in pixels
    amplitude: f32,   // vertical scale of each series
    hovered: i32,     // series under the mouse, or -1
}

// Layout in world space: time runs from x = -2 to x = 2, and each series
// gets its own horizontal lane, LANE apart (series 0 at the top).
const LANE: f32 = 0.5;

fn sample_position(s: u32, t: u32, value: f32, params: Params) -> vec2f {
    let x = -2.0 + 4.0 * f32(t) / f32(params.samples - 1u);
    let lane = (0.5 * f32(params.series - 1u) - f32(s)) * LANE;
    return vec2f(x, lane + value * params.amplitude);
}
