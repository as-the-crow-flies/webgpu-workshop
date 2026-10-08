// Camera helpers: billboards, camera rays and depth.
#import "lib/globals.wgsl"

// A world-space quad around `center` that always faces the camera.
// `corner` comes from `quad_corner` (lib/math.wgsl). Returns the clip position.
fn billboard(center: vec3f, half_size: f32, corner: vec2f) -> vec4f {
    let right = vec3f(globals.view[0].x, globals.view[1].x, globals.view[2].x);
    let up = vec3f(globals.view[0].y, globals.view[1].y, globals.view[2].y);
    let world = center + (right * corner.x + up * corner.y) * half_size;
    return globals.view_proj * vec4f(world, 1.0);
}

// A quad that covers a sphere on screen exactly, also up close and at the
// edges of the view (where `billboard` would clip it). The quad faces the eye,
// and is a bit larger than the sphere: seen from distance d, a sphere of radius
// r covers a circle of radius r * d / sqrt(d^2 - r^2) around its center.
fn sphere_billboard(center: vec3f, radius: f32, corner: vec2f) -> vec4f {
    var to_eye = vec3f(globals.view[0].z, globals.view[1].z, globals.view[2].z); // orthographic
    var size = radius;
    if (globals.orthographic == 0u) {
        let d = distance(globals.eye, center);
        to_eye = (globals.eye - center) / d;
        size = radius * d / sqrt(max(d * d - radius * radius, 1e-6));
    }
    // Two directions along the quad, perpendicular to `to_eye`.
    let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(to_eye.y) > 0.99);
    let right = normalize(cross(helper, to_eye));
    let up = cross(to_eye, right);
    let world = center + (right * corner.x + up * corner.y) * size;
    return globals.view_proj * vec4f(world, 1.0);
}

struct Ray {
    origin: vec3f,
    dir: vec3f,
}

// The ray from the camera through a pixel. Pass `@builtin(position).xy` from a
// fragment shader. Works for perspective and orthographic cameras.
fn camera_ray(frag_coord: vec2f) -> Ray {
    let ndc = vec2f(2.0 * frag_coord.x / globals.resolution.x - 1.0,
                    1.0 - 2.0 * frag_coord.y / globals.resolution.y);
    let near = globals.inv_view_proj * vec4f(ndc, 0.0, 1.0);
    let far = globals.inv_view_proj * vec4f(ndc, 1.0, 1.0);
    let origin = near.xyz / near.w;
    return Ray(origin, normalize(far.xyz / far.w - origin));
}

// The ray under the mouse.
fn mouse_ray() -> Ray {
    return Ray(globals.ray_origin, globals.ray_dir);
}

// Depth-buffer value of a world-space point, for `@builtin(frag_depth)`.
fn depth_of(world: vec3f) -> f32 {
    let clip = globals.view_proj * vec4f(world, 1.0);
    return clip.z / clip.w;
}
