// The data layouts of this example, shared by all of its shaders.

// One node, exactly as stored in nodes.npy:
//   np.dtype([("pos", "<f4", 3), ("radius", "<f4"), ("vel", "<f4", 3), ("group", "<u4")])
struct Node {
    pos: vec3f,
    radius: f32,
    vel: vec3f,
    group: u32,   // community, used for the color
}

// One edge between two nodes (indices into the node array), as stored in edges.npy:
//   np.dtype([("a", "<u4"), ("b", "<u4")])
struct Edge {
    a: u32,
    b: u32,
}

// Values controlled by the sliders. `Params` in main.rs must have the same layout.
struct Params {
    repulsion: f32,      // how strongly all nodes push each other away
    spring_length: f32,  // rest length of an edge
    spring: f32,         // stiffness of an edge
    damping: f32,        // fraction of the velocity kept per frame
    gravity: f32,        // pull towards the origin
    edge_radius: f32,    // thickness of the edge cylinders
    node_radius: f32,    // scales the radius of every node (from nodes.npy: larger for more edges)
    flat: u32,           // 1: keep the layout in the plane z = 0
}

// The node under the mouse when the left button went down (see pick.wgsl).
// Packed as (depth bits << 16) | node index, so it supports up to 65536 nodes.
const NO_PICK: u32 = 0xffffffffu;

fn picked_node(pick: u32) -> i32 {
    if (pick == NO_PICK) {
        return -1;
    }
    return i32(pick & 0xffffu);
}
