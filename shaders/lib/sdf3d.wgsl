// 3D signed distance functions (after Inigo Quilez, https://iquilezles.org/articles/distfunctions/).
// Negative inside, positive outside. Use them for ray marching:
//   var t = 0.0;
//   for (var i = 0; i < 128; i++) {
//       let d = scene(ray.origin + t * ray.dir);
//       if (d < 0.001) { break; }
//       t += d;
//   }

fn sd_sphere(p: vec3f, radius: f32) -> f32 {
    return length(p) - radius;
}

// Box with half-size `b`, centered at the origin.
fn sd_box3(p: vec3f, b: vec3f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_round_box3(p: vec3f, b: vec3f, r: f32) -> f32 {
    return sd_box3(p, b - vec3f(r)) - r;
}

// Capsule from a to b with radius r.
fn sd_capsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// Torus in the xz-plane: `t.x` is the main radius, `t.y` the tube radius.
fn sd_torus(p: vec3f, t: vec2f) -> f32 {
    let q = vec2f(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

// Cylinder along y with radius `r` and half-height `h`.
fn sd_cylinder(p: vec3f, r: f32, h: f32) -> f32 {
    let d = abs(vec2f(length(p.xz), p.y)) - vec2f(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

// Plane with unit normal `n` at distance `h` from the origin.
fn sd_plane(p: vec3f, n: vec3f, h: f32) -> f32 {
    return dot(p, n) + h;
}

// Smooth union of two distances; `k` is the blend size.
fn smooth_min3(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}
