// Random numbers on the GPU: hash an integer (index, frame, ...) to a random value.
// Seed with e.g. `index * 7919u + globals.frame`.

// PCG hash: a well-mixed u32 from a u32.
fn pcg(seed: u32) -> u32 {
    let state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Uniform float in [0, 1).
fn rand(seed: u32) -> f32 {
    return f32(pcg(seed)) / 4294967296.0;
}

fn rand2(seed: u32) -> vec2f {
    return vec2f(rand(seed), rand(pcg(seed)));
}

fn rand3(seed: u32) -> vec3f {
    let s = pcg(seed);
    return vec3f(rand(seed), rand(s), rand(pcg(s)));
}

// Uniformly distributed direction on the unit sphere.
fn rand_direction(seed: u32) -> vec3f {
    let r = rand2(seed);
    let z = 1.0 - 2.0 * r.x;
    let a = 6.28318530718 * r.y;
    let s = sqrt(max(0.0, 1.0 - z * z));
    return vec3f(s * cos(a), s * sin(a), z);
}
