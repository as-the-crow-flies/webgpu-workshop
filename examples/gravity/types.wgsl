// The data layouts of this example, shared by all of its shaders.

// One body, exactly as stored in bodies.npy. In numpy this dtype is:
//   np.dtype([("pos", "<f4", 3), ("mass", "<f4"), ("vel", "<f4", 3), ("size", "<f4"), ("color", "<f4", 4)])
// The file must match this struct byte for byte (see make_data.py).
struct Body {
    pos: vec3f,    // bytes 0..12
    mass: f32,     // bytes 12..16 (fills the gap after the vec3, which is 16-byte aligned)
    vel: vec3f,    // bytes 16..28
    size: f32,     // bytes 28..32
    color: vec4f,  // bytes 32..48
}

// Values controlled by the sliders. `Params` in main.rs must have the same layout.
struct Params {
    point_scale: f32,
    gravity: f32,      // the gravitational constant G
    softening: f32,    // keeps the force finite when two bodies get very close
    central_mass: f32, // the heavy mass at the origin
}
