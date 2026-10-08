// The data layouts of this example, shared by all of its shaders.

// One point, exactly as stored in points.npy. In numpy this dtype is:
//   np.dtype([("pos", "<f4", 3), ("size", "<f4"), ("color", "<f4", 4)])
// The file must match this struct byte for byte (see make_data.py).
struct Point {
    pos: vec3f,   // bytes 0..12
    size: f32,    // bytes 12..16 (fills the gap after the vec3, which is 16-byte aligned)
    color: vec4f, // bytes 16..32
}

// Values controlled by the sliders. `Params` in main.rs must have the same layout.
struct Params {
    point_scale: f32,
}
