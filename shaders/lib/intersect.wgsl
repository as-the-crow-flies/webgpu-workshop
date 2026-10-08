// Ray intersections (after Inigo Quilez, https://iquilezles.org/articles/intersectors/).
// All functions return the distance `t` along the ray to the first hit, or a
// negative value if there is no hit. The hit point is `ray.origin + t * ray.dir`.
//
// Typical use in a fragment shader (an "impostor"): draw a quad around the
// object, then intersect the camera ray with the exact shape:
//   let ray = camera_ray(in.position.xy);
//   let t = ray_sphere(ray, center, radius);
//   if (t < 0.0) { discard; }
#import "lib/camera.wgsl"

fn ray_sphere(ray: Ray, center: vec3f, radius: f32) -> f32 {
    let oc = ray.origin - center;
    let b = dot(oc, ray.dir);
    let c = dot(oc, oc) - radius * radius;
    let h = b * b - c;
    if (h < 0.0) { return -1.0; }
    return -b - sqrt(h);
}

fn sphere_normal(p: vec3f, center: vec3f) -> vec3f {
    return normalize(p - center);
}

// Axis-aligned box. Returns (t_enter, t_exit); the ray misses if t_enter > t_exit
// or t_exit < 0. Useful for volume rendering: march from max(t_enter, 0) to t_exit.
fn ray_box(ray: Ray, box_min: vec3f, box_max: vec3f) -> vec2f {
    let inv = 1.0 / ray.dir;
    let t0 = (box_min - ray.origin) * inv;
    let t1 = (box_max - ray.origin) * inv;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    return vec2f(max(max(lo.x, lo.y), lo.z), min(min(hi.x, hi.y), hi.z));
}

// Normal of an axis-aligned box at surface point `p`.
fn box_normal(p: vec3f, box_min: vec3f, box_max: vec3f) -> vec3f {
    let q = (p - 0.5 * (box_min + box_max)) / (0.5 * (box_max - box_min));
    let a = abs(q);
    if (a.x > a.y && a.x > a.z) { return vec3f(sign(q.x), 0.0, 0.0); }
    if (a.y > a.z) { return vec3f(0.0, sign(q.y), 0.0); }
    return vec3f(0.0, 0.0, sign(q.z));
}

// Plane through `point` with unit normal `n`.
fn ray_plane(ray: Ray, point: vec3f, n: vec3f) -> f32 {
    return dot(point - ray.origin, n) / dot(ray.dir, n);
}

// Disk at `center` with unit normal `n` and radius `r`.
fn ray_disk(ray: Ray, center: vec3f, n: vec3f, r: f32) -> f32 {
    let t = ray_plane(ray, center, n);
    let q = ray.origin + t * ray.dir - center;
    return select(-1.0, t, dot(q, q) < r * r);
}

// Capsule from a to b with radius r.
fn ray_capsule(ray: Ray, a: vec3f, b: vec3f, r: f32) -> f32 {
    let ba = b - a;
    let oa = ray.origin - a;
    let baba = dot(ba, ba);
    let bard = dot(ba, ray.dir);
    let baoa = dot(ba, oa);
    let rdoa = dot(ray.dir, oa);
    let oaoa = dot(oa, oa);
    let k2 = baba - bard * bard;
    let k1 = baba * rdoa - baoa * bard;
    let k0 = baba * oaoa - baoa * baoa - r * r * baba;
    let h = k1 * k1 - k2 * k0;
    if (h < 0.0) { return -1.0; }
    let t = (-k1 - sqrt(h)) / k2;
    let y = baoa + t * bard;
    if (y > 0.0 && y < baba) { return t; }   // the body
    // one of the two spherical caps
    let oc = select(ray.origin - b, oa, y <= 0.0);
    let bc = dot(ray.dir, oc);
    let hc = bc * bc - (dot(oc, oc) - r * r);
    if (hc > 0.0) { return -bc - sqrt(hc); }
    return -1.0;
}

fn capsule_normal(p: vec3f, a: vec3f, b: vec3f, r: f32) -> vec3f {
    let ba = b - a;
    let pa = p - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return (pa - h * ba) / r;
}

// Capped cylinder from a to b with radius r. Returns (t, normal); t < 0 on a miss.
fn ray_cylinder(ray: Ray, a: vec3f, b: vec3f, r: f32) -> vec4f {
    let ba = b - a;
    let oc = ray.origin - a;
    let baba = dot(ba, ba);
    let bard = dot(ba, ray.dir);
    let baoc = dot(ba, oc);
    let k2 = baba - bard * bard;
    let k1 = baba * dot(oc, ray.dir) - baoc * bard;
    let k0 = baba * dot(oc, oc) - baoc * baoc - r * r * baba;
    var h = k1 * k1 - k2 * k0;
    if (h < 0.0) { return vec4f(-1.0); }
    h = sqrt(h);
    var t = (-k1 - h) / k2;
    let y = baoc + t * bard;
    if (y > 0.0 && y < baba) {   // the body
        return vec4f(t, (oc + t * ray.dir - ba * y / baba) / r);
    }
    t = (select(baba, 0.0, y < 0.0) - baoc) / bard;   // the caps
    if (abs(k1 + k2 * t) < h) {
        return vec4f(t, ba * sign(y) / sqrt(baba));
    }
    return vec4f(-1.0);
}

// Ellipsoid at `center` with radii `radii`. Returns (t_enter, t_exit), both -1 on a miss.
fn ray_ellipsoid(ray: Ray, center: vec3f, radii: vec3f) -> vec2f {
    let o = (ray.origin - center) / radii;
    let d = ray.dir / radii;
    let a = dot(d, d);
    let b = dot(o, d);
    let c = dot(o, o);
    let h = b * b - a * (c - 1.0);
    if (h < 0.0) { return vec2f(-1.0); }
    let s = sqrt(h);
    return vec2f(-b - s, -b + s) / a;
}

fn ellipsoid_normal(p: vec3f, center: vec3f, radii: vec3f) -> vec3f {
    return normalize((p - center) / (radii * radii));
}

// Triangle v0, v1, v2. Returns (t, u, v) with barycentric u, v; t < 0 on a miss.
fn ray_triangle(ray: Ray, v0: vec3f, v1: vec3f, v2: vec3f) -> vec3f {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let o = ray.origin - v0;
    let n = cross(e1, e2);
    let q = cross(o, ray.dir);
    let d = 1.0 / dot(ray.dir, n);
    let u = d * dot(-q, e2);
    let v = d * dot(q, e1);
    var t = d * dot(-n, o);
    if (u < 0.0 || v < 0.0 || u + v > 1.0) { t = -1.0; }
    return vec3f(t, u, v);
}

// Closest distance between a ray and a point (for picking).
fn ray_point_distance(ray: Ray, p: vec3f) -> f32 {
    let t = max(dot(p - ray.origin, ray.dir), 0.0);
    return length(ray.origin + t * ray.dir - p);
}

// Billboards: a quad (from `quad_corner`, 6 vertices) that covers a shape on
// screen, for the impostor pattern above. Each one puts the shape in a sphere
// and uses `sphere_billboard` (lib/camera.wgsl), so the quad can be a bit
// larger than the shape: the fragment shader discards the pixels it misses.
// For a sphere itself, use `sphere_billboard` directly.

fn box_billboard(box_min: vec3f, box_max: vec3f, corner: vec2f) -> vec4f {
    return sphere_billboard(0.5 * (box_min + box_max), 0.5 * length(box_max - box_min), corner);
}

fn disk_billboard(center: vec3f, r: f32, corner: vec2f) -> vec4f {
    return sphere_billboard(center, r, corner);
}

fn capsule_billboard(a: vec3f, b: vec3f, r: f32, corner: vec2f) -> vec4f {
    return sphere_billboard(0.5 * (a + b), 0.5 * length(b - a) + r, corner);
}

fn cylinder_billboard(a: vec3f, b: vec3f, r: f32, corner: vec2f) -> vec4f {
    let half_length = 0.5 * length(b - a);
    return sphere_billboard(0.5 * (a + b), sqrt(half_length * half_length + r * r), corner);
}

fn ellipsoid_billboard(center: vec3f, radii: vec3f, corner: vec2f) -> vec4f {
    return sphere_billboard(center, max(radii.x, max(radii.y, radii.z)), corner);
}

fn triangle_billboard(v0: vec3f, v1: vec3f, v2: vec3f, corner: vec2f) -> vec4f {
    let center = (v0 + v1 + v2) / 3.0;
    let r = sqrt(max(dot(v0 - center, v0 - center), max(dot(v1 - center, v1 - center), dot(v2 - center, v2 - center))));
    return sphere_billboard(center, r, corner);
}
