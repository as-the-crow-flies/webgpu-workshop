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
// screen, for the impostor pattern above. Most put the shape in a sphere and use
// `sphere_billboard` (lib/camera.wgsl), so the quad can be a bit larger than the
// shape: the fragment shader discards the pixels it misses. Capsules and cylinders
// use `tube_billboard`, which fits long, thin shapes much more tightly.
// For a sphere itself, use `sphere_billboard` directly.

fn box_billboard(box_min: vec3f, box_max: vec3f, corner: vec2f) -> vec4f {
    return sphere_billboard(0.5 * (box_min + box_max), 0.5 * length(box_max - box_min), corner);
}

fn disk_billboard(center: vec3f, r: f32, corner: vec2f) -> vec4f {
    return sphere_billboard(center, r, corner);
}

// A tight quad around a tube: spheres of radius `ra` at `a` and `rb` at `b`, and the cone
// that connects them (a "rounded cone"). After Groß & Gumhold, "Advanced Rendering of Line
// Data with Ambient Occlusion and Transparency" (IEEE TVCG, 2021).
//
// The quad lies in the plane through the tube's axis that faces the eye. Across the tube it
// is as wide as the end spheres look (`sphere_billboard`); along the tube it runs from the
// outline of one end sphere to the outline of the other.
fn tube_billboard(a: vec3f, b: vec3f, ra: f32, rb: f32, corner: vec2f) -> vec4f {
    let forward = -vec3f(globals.view[0].z, globals.view[1].z, globals.view[2].z); // camera looks along this
    let orthographic = globals.orthographic != 0u;

    // From each end towards the eye (in orthographic views, the same for every point).
    let to_eye_a = select(globals.eye - a, -forward, orthographic);
    let to_eye_b = select(globals.eye - b, -forward, orthographic);
    let la = length(to_eye_a);
    let lb = length(to_eye_b);
    // How much larger than its radius each end sphere looks, per unit of distance: seen from
    // distance l, a sphere of radius r covers a circle of radius l * s (see `sphere_billboard`).
    let sa = select(ra / sqrt(max(la * la - ra * ra, 1e-12)), ra, orthographic);
    let sb = select(rb / sqrt(max(lb * lb - rb * rb, 1e-12)), rb, orthographic);

    // `across`: perpendicular to the tube and to the eye, so across the tube on screen.
    var across = cross(b - a, to_eye_a);
    if (dot(across, across) < 1e-12) { // looking straight down the tube: any perpendicular
        let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normalize(to_eye_a).y) > 0.99);
        across = cross(helper, to_eye_a);
    }
    across = normalize(across);
    // In the quad's plane, perpendicular to the view ray to each end: the end sphere's outline
    // reaches l * s along these, both ways.
    let up_a = normalize(cross(across, to_eye_a));
    let up_b = normalize(cross(across, to_eye_b));
    let a_plus = a + la * sa * up_a;
    let a_minus = a - la * sa * up_a;
    let b_plus = b + lb * sb * up_b;
    let b_minus = b - lb * sb * up_b;

    // Along the tube on screen: which of those outline points is the first and the last.
    let along = cross(across, forward);
    let s = max(sa, sb);
    var start = b_plus;
    var start_width = lb * s;
    if (screen_along(a_plus, along) <= screen_along(b_plus, along)) {
        start = a_plus;
        start_width = la * s;
    }
    var end = a_minus;
    var end_width = la * s;
    if (screen_along(a_minus, along) <= screen_along(b_minus, along)) {
        end = b_minus;
        end_width = lb * s;
    }

    // corner.x: -1 at the start, 1 at the end. corner.y: -1 .. 1 across.
    var world = end + across * corner.y * end_width;
    if (corner.x < 0.0) {
        world = start + across * corner.y * start_width;
    }
    return globals.view_proj * vec4f(world, 1.0);
}

// Where `p` is on screen along `direction`, for comparing points in `tube_billboard`.
fn screen_along(p: vec3f, direction: vec3f) -> f32 {
    if (globals.orthographic != 0u) {
        return dot(p, direction);
    }
    return dot(normalize(p - globals.eye), direction);
}

fn capsule_billboard(a: vec3f, b: vec3f, r: f32, corner: vec2f) -> vec4f {
    return tube_billboard(a, b, r, r, corner);
}

// A capped cylinder fits inside the capsule of the same radius.
fn cylinder_billboard(a: vec3f, b: vec3f, r: f32, corner: vec2f) -> vec4f {
    return tube_billboard(a, b, r, r, corner);
}

fn ellipsoid_billboard(center: vec3f, radii: vec3f, corner: vec2f) -> vec4f {
    return sphere_billboard(center, max(radii.x, max(radii.y, radii.z)), corner);
}

fn triangle_billboard(v0: vec3f, v1: vec3f, v2: vec3f, corner: vec2f) -> vec4f {
    let center = (v0 + v1 + v2) / 3.0;
    let r = sqrt(max(dot(v0 - center, v0 - center), max(dot(v1 - center, v1 - center), dot(v2 - center, v2 - center))));
    return sphere_billboard(center, r, corner);
}
