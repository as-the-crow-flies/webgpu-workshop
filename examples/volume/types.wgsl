// The data layouts of this example, shared by all of its shaders.

// One particle, exactly as stored in particles.npy (positions in [0, 1]^3):
//   np.dtype([("pos", "<f4", 3), ("weight", "<f4")])
struct Particle {
    pos: vec3f,
    weight: f32,
}

// Values controlled by the sliders. `Params` in main.rs must have the same layout.
struct Params {
    grid: u32,         // the grid has grid x grid x grid voxels
    splat: f32,        // multiplies every particle's weight
    density: f32,      // how opaque the volume is
    brightness: f32,   // how quickly the colormap saturates
}

// WebGPU has no floating-point atomics, so the splat pass adds weights as
// fixed-point integers: weight * SCALE, rounded.
const SCALE: f32 = 256.0;

fn voxel_index(v: vec3u, grid: u32) -> u32 {
    return v.x + grid * (v.y + grid * v.z);
}
