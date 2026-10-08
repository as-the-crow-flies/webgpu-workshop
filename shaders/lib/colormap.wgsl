// Colormaps: map a value in [0, 1] to a color.
// Polynomial fits (max error ~2%) of matplotlib's viridis, magma, plasma (CC0)
// and coolwarm (diverging; use it for data centered around zero).

fn poly6(c0: vec3f, c1: vec3f, c2: vec3f, c3: vec3f, c4: vec3f, c5: vec3f, c6: vec3f, t: f32) -> vec3f {
    return saturate(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))));
}

fn viridis(t: f32) -> vec3f {
    return poly6(
        vec3f(0.274455, 0.005768, 0.332664), vec3f(0.107708, 1.396470, 1.386771),
        vec3f(-0.327241, 0.214814, 0.091977), vec3f(-4.599932, -5.758238, -19.291809),
        vec3f(6.203736, 14.153965, 56.656300), vec3f(4.751787, -13.749439, -65.320968),
        vec3f(-5.432077, 4.641571, 26.272108), saturate(t));
}

fn magma(t: f32) -> vec3f {
    return poly6(
        vec3f(-0.002067, -0.000688, -0.009548), vec3f(0.250486, 0.694455, 2.495287),
        vec3f(8.345901, -3.596031, 0.329057), vec3f(-27.666969, 14.253853, -13.646583),
        vec3f(52.170684, -27.944584, 12.881091), vec3f(-50.758572, 29.053880, 4.269936),
        vec3f(18.664253, -11.490027, -5.570769), saturate(t));
}

fn plasma(t: f32) -> vec3f {
    return poly6(
        vec3f(0.064053, 0.024812, 0.534900), vec3f(2.142438, 0.244749, 0.742966),
        vec3f(-2.653255, -7.461101, 3.108382), vec3f(6.094711, 42.308428, -28.491792),
        vec3f(-11.065106, -82.644718, 60.093584), vec3f(9.974645, 71.408341, -54.020563),
        vec3f(-3.623823, -22.914405, 18.193381), saturate(t));
}

fn coolwarm(t: f32) -> vec3f {
    return poly6(
        vec3f(0.228493, 0.289034, 0.754451), vec3f(1.204425, 2.299764, 1.558905),
        vec3f(0.093671, -7.278712, -1.875136), vec3f(2.240691, 32.222794, -1.585472),
        vec3f(-5.130376, -74.669002, -3.741622), vec3f(1.414258, 72.799673, 9.572374),
        vec3f(0.657454, -25.633429, -4.537389), saturate(t));
}

// A distinct color per category (golden-ratio hue steps).
fn category_color(i: u32) -> vec3f {
    let h = fract(f32(i) * 0.618034);
    return saturate(abs(fract(vec3f(h) + vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0) * 0.7 + 0.25;
}
