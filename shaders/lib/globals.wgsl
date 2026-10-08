// Camera, mouse and time. Filled by the framework every frame (see `Globals`
// in src/lib.rs). Import this file and declare the binding in your shader:
//   #import "lib/globals.wgsl"
//   @group(0) @binding(0) var<uniform> globals: Globals;
// By convention, binding 0 of every bind group is `ctx.globals_buffer`;
// your own resources start at binding 1.
// (The helpers below, and lib/camera.wgsl, use that `globals` variable.)

struct Globals {
    view: mat4x4f,          // world -> camera
    proj: mat4x4f,          // camera -> clip
    view_proj: mat4x4f,     // world -> clip
    inv_view_proj: mat4x4f, // clip -> world
    eye: vec3f,             // camera position (world)
    time: f32,              // seconds since start (stops while paused)
    ray_origin: vec3f,      // ray from the camera through the mouse (world)
    dt: f32,                // seconds since the previous frame (0 while paused)
    ray_dir: vec3f,
    frame: u32,             // frame counter
    resolution: vec2f,      // window size in pixels
    mouse: vec2f,           // mouse position in pixels, from the top-left
    mouse_ndc: vec2f,       // mouse position in [-1, 1], y up
    buttons: u32,           // bit 0: left, bit 1: right, bit 2: middle
    orthographic: u32,      // 1 if the camera is orthographic
}

const MOUSE_LEFT: u32 = 1u;
const MOUSE_RIGHT: u32 = 2u;
const MOUSE_MIDDLE: u32 = 4u;

fn mouse_down(button: u32) -> bool {
    return (globals.buttons & button) != 0u;
}

// Where the mouse ray hits the plane z = `z` (e.g. the mouse in 2D world coordinates).
fn mouse_on_plane(z: f32) -> vec3f {
    let t = (z - globals.ray_origin.z) / globals.ray_dir.z;
    return globals.ray_origin + t * globals.ray_dir;
}
