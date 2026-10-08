// Volume rendering helpers: march along a ray and composite front to back.
//
//   var acc = vec4f(0.0);
//   for (var t = t_enter; t < t_exit && acc.a < 0.99; t += step) {
//       let density = textureSampleLevel(volume, volume_sampler, p(t), 0.0).r;
//       acc = composite(acc, color(density), alpha(density) );
//   }

// Add a sample with `color` and opacity `alpha` behind what was accumulated so far.
// `acc` is premultiplied (rgb already multiplied by alpha).
fn composite(acc: vec4f, color: vec3f, alpha: f32) -> vec4f {
    let a = saturate(alpha) * (1.0 - acc.a);
    return vec4f(acc.rgb + color * a, acc.a + a);
}

// Opacity of one step of length `step` through a medium of `density`
// (Beer-Lambert), so the result does not depend on the step size.
fn step_alpha(density: f32, step: f32) -> f32 {
    return 1.0 - exp(-max(density, 0.0) * step);
}
