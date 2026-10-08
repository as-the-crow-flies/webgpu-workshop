// Anti-aliasing for signed distances (negative inside, positive outside).
// The derivative (fwidth) tells how much `d` changes per pixel, so the edge is
// blurred over exactly one pixel at any zoom level.
//
// Call these at the top level of a fragment shader, not inside an `if`:
// derivatives need all neighbouring pixels to run the same code.

// Coverage of a filled shape: 1 inside, 0 outside, smooth over one pixel.
fn aa_fill(d: f32) -> f32 {
    let w = max(fwidth(d), 1e-6);
    return clamp(0.5 - d / w, 0.0, 1.0);
}

// Coverage of an outline of `width` (in the same units as `d`).
fn aa_stroke(d: f32, width: f32) -> f32 {
    return aa_fill(abs(d) - 0.5 * width);
}

// Same as aa_fill, for a distance that is already in pixels.
fn aa_fill_px(d_px: f32) -> f32 {
    return clamp(0.5 - d_px, 0.0, 1.0);
}

// Premultiplied-alpha "over": put `top` on top of `bottom`.
fn over(bottom: vec4f, top: vec4f) -> vec4f {
    return top + bottom * (1.0 - top.a);
}
