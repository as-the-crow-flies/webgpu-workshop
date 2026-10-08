// Step 2: Write a Shader
//
// Volume rendering. A full-screen triangle; every pixel marches its
// camera ray through the unit cube [0, 1]^3 and composites front to back.
#import "types.wgsl"
#import "lib/globals.wgsl"
#import "lib/math.wgsl"
#import "lib/camera.wgsl"
#import "lib/intersect.wgsl"
#import "lib/volume.wgsl"
#import "lib/colormap.wgsl"

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var volume: texture_3d<f32>;
@group(0) @binding(3) var volume_sampler: sampler;

@vertex
fn vs(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4f {
    return fullscreen_triangle(vertex);
}

@fragment
fn fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let ray = camera_ray(position.xy);
    let range = ray_box(ray, vec3f(0.0), vec3f(1.0));
    let t_enter = max(range.x, 0.0);
    if (range.y < t_enter) {
        return vec4f(0.0);   // the ray misses the cube
    }

    // Half a voxel per step, starting at a random offset to hide banding.
    let step = 0.5 / f32(params.grid);
    let jitter = fract(sin(dot(position.xy, vec2f(12.9898, 78.233))) * 43758.5453);
    var acc = vec4f(0.0);
    for (var t = t_enter + jitter * step; t < range.y && acc.a < 0.99; t += step) {
        let p = ray.origin + t * ray.dir;
        let value = textureSampleLevel(volume, volume_sampler, p, 0.0).r;
        let color = magma(1.0 - exp(-value * params.brightness));
        acc = composite(acc, color, step_alpha(value * params.density, step));
    }
    return acc;   // premultiplied
}
