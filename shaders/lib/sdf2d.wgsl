// 2D signed distance functions (after Inigo Quilez, https://iquilezles.org/articles/distfunctions2d/).
// Negative inside, positive outside. Combine with aa_fill / aa_stroke from lib/aa.wgsl.

fn sd_circle(p: vec2f, radius: f32) -> f32 {
    return length(p) - radius;
}

// Box with half-size `b`, centered at the origin.
fn sd_box(p: vec2f, b: vec2f) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}

// Box with half-size `b` and corner radius `r`.
fn sd_round_box(p: vec2f, b: vec2f, r: f32) -> f32 {
    return sd_box(p, b - vec2f(r)) - r;
}

// Distance to the line segment a-b (add a radius for a thick line: sd_segment(...) - r).
fn sd_segment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

fn sd_triangle(p: vec2f, p0: vec2f, p1: vec2f, p2: vec2f) -> f32 {
    let e0 = p1 - p0;
    let e1 = p2 - p1;
    let e2 = p0 - p2;
    let v0 = p - p0;
    let v1 = p - p1;
    let v2 = p - p2;
    let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
    let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
    let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
    let s = sign(e0.x * e2.y - e0.y * e2.x);
    let d = min(min(vec2f(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
                    vec2f(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
                    vec2f(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));
    return -sqrt(d.x) * sign(d.y);
}

// A ring (annulus) of radius `r` and thickness `w`.
fn sd_ring(p: vec2f, r: f32, w: f32) -> f32 {
    return abs(length(p) - r) - 0.5 * w;
}

// Smooth union of two distances; `k` is the blend size.
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}
