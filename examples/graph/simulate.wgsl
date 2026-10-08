// Step 2: Write a Shader
//
// A force-directed layout: all nodes repel each other, edges act as springs.
// The picked node follows the mouse while the left button is down.
#import "types.wgsl"
#import "lib/globals.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> ping: array<Node>;
@group(0) @binding(3) var<storage, read_write> pong: array<Node>;
@group(0) @binding(4) var<storage, read> edges: array<Edge>;
@group(0) @binding(5) var<storage, read> pick: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let n = arrayLength(&ping);
    if (i >= n) {
        return;
    }
    var me = ping[i];
    var force = vec3f(0.0);

    // Repulsion: every node pushes every other node away (1 / distance^2).
    for (var j = 0u; j < n; j++) {
        let d = me.pos - ping[j].pos;
        let dist2 = dot(d, d) + 0.001;
        force += d / (dist2 * sqrt(dist2)) * params.repulsion;
    }

    // Springs: look through all edges for the ones that touch this node.
    for (var e = 0u; e < arrayLength(&edges); e++) {
        let edge = edges[e];
        if (edge.a == i || edge.b == i) {
            let other = ping[select(edge.a, edge.b, edge.a == i)];
            let d = other.pos - me.pos;
            let len = length(d) + 1e-6;
            force += d / len * (len - params.spring_length) * params.spring;
        }
    }

    force -= me.pos * params.gravity;

    let dt = min(globals.dt, 1.0 / 30.0);
    me.vel = me.vel * pow(params.damping, dt * 60.0) + force * dt;
    me.pos += me.vel * dt;
    if (params.flat == 1u) {
        me.pos.z = 0.0;
        me.vel.z = 0.0;
    }

    // Dragging: move the picked node to the mouse, in the plane through the
    // node that faces the camera.
    if (picked_node(pick) == i32(i) && mouse_down(MOUSE_LEFT)) {
        let forward = vec3f(globals.view[0].z, globals.view[1].z, globals.view[2].z);
        let t = dot(ping[i].pos - globals.ray_origin, forward) / dot(globals.ray_dir, forward);
        me.pos = globals.ray_origin + t * globals.ray_dir;
        me.vel = vec3f(0.0);
    }

    pong[i] = me;
}
