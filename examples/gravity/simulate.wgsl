// Step 2: Write a Shader
//
// Gravity: every body pulls on every other body, and a
// heavy mass sits at the origin. Reads the bodies from `ping` and writes the
// next state to `pong`; main.rs swaps the two buffers every frame.
#import "types.wgsl"
#import "lib/globals.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> ping: array<Body>;
@group(0) @binding(3) var<storage, read_write> pong: array<Body>;

// Acceleration of a body at `pos` caused by a mass `mass` at `other`.
fn pull(pos: vec3f, other: vec3f, mass: f32) -> vec3f {
    let d = other - pos;
    let r2 = dot(d, d) + params.softening * params.softening;
    return d * (mass / (r2 * sqrt(r2)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let n = arrayLength(&ping);
    if (i >= n) {
        return;
    }

    var body = ping[i];

    // Add up the pull of all other bodies (O(n^2): fine for a few thousand).
    var acc = pull(body.pos, vec3f(0.0), params.central_mass);
    for (var j = 0u; j < n; j++) {
        if (j != i) {
            acc += pull(body.pos, ping[j].pos, ping[j].mass);
        }
    }
    acc *= params.gravity;

    // Semi-implicit Euler: first the velocity, then the position with the new velocity.
    let dt = min(globals.dt, 1.0 / 30.0);
    body.vel += acc * dt;
    body.pos += body.vel * dt;

    // Every element must be written, even when it does not change.
    pong[i] = body;
}
