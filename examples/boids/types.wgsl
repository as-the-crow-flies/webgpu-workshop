// The data layouts of this example, shared by all of its shaders.

// One boid. There is no .npy file: init.wgsl generates the boids on the GPU.
struct Boid {
    pos: vec2f,
    vel: vec2f,
}

// Values controlled by the sliders. `Params` in main.rs must have the same layout.
struct Params {
    separation: f32,   // steer away from boids closer than this
    alignment: f32,    // strength of matching the neighbours' heading
    cohesion: f32,     // strength of moving towards the neighbours' center
    radius: f32,       // how far a boid can see
    max_speed: f32,
    mouse_force: f32,  // pull towards the mouse while the left button is down (negative: push away)
    seed: u32,         // changes on "Reset"
    size: f32,         // drawing size
}

// The boids live in the square [-WORLD, WORLD]^2 and wrap around at the edges.
const WORLD: f32 = 1.5;
