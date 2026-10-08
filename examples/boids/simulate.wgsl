// Step 2: Write a Shader
//
// The flocking rules: separation, alignment and cohesion (Reynolds 1987),
// plus a force towards (or away from) the mouse.
#import "types.wgsl"
#import "lib/globals.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> ping: array<Boid>;
@group(0) @binding(3) var<storage, read_write> pong: array<Boid>;

// The shortest offset from a to b in a world that wraps around.
fn wrapped(d: vec2f) -> vec2f {
    return d - 2.0 * WORLD * round(d / (2.0 * WORLD));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let n = arrayLength(&ping);
    if (i >= n) {
        return;
    }
    var me = ping[i];

    var separation = vec2f(0.0);
    var heading = vec2f(0.0);
    var center = vec2f(0.0);
    var neighbours = 0.0;

    // Every boid looks at every other boid: simple, and fast enough for a few thousand.
    for (var j = 0u; j < n; j++) {
        if (j == i) { continue; }
        let other = ping[j];
        let d = wrapped(other.pos - me.pos);
        let dist = length(d);
        if (dist < params.radius) {
            heading += other.vel;
            center += d;
            neighbours += 1.0;
            if (dist < params.separation) {
                separation -= d / max(dist * dist, 1e-4);
            }
        }
    }

    var force = separation * 0.002;
    if (neighbours > 0.0) {
        force += (heading / neighbours - me.vel) * params.alignment;
        force += (center / neighbours) * params.cohesion;
    }

    // The mouse: `mouse_force` > 0 attracts, < 0 repels (set from main.rs).
    if (mouse_down(MOUSE_LEFT)) {
        let to_mouse = wrapped(mouse_on_plane(0.0).xy - me.pos);
        force += normalize(to_mouse) * params.mouse_force / (0.2 + length(to_mouse));
    }

    me.vel += force * globals.dt * 60.0;
    let speed = length(me.vel);
    if (speed > params.max_speed) {
        me.vel *= params.max_speed / speed;
    }
    me.pos += me.vel * globals.dt;
    me.pos = wrapped(me.pos);   // wrap around the edges

    pong[i] = me;
}
