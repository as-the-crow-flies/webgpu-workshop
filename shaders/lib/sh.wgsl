// Real, symmetric spherical harmonics in the MRtrix3 convention (even orders only),
// for rendering fibre orientation distributions (FODs).
//
// Coefficient (l, m) is at index l * (l + 1) / 2 + m, with l = 0, 2, 4, 6, 8 and -l <= m <= l.
// lmax 8 gives 45 coefficients (lmax 6: 28, lmax 4: 15).
// The amplitude in direction `dir` is: sum over i of coefficient[i] * sh_basis(dir)[i].

const SH_LMAX: i32 = 8;
const SH_COUNT: u32 = 45u;

// Associated Legendre functions for one m, normalized as in MRtrix3 (Plm_sph).
fn sh_legendre(m: i32, x: f32) -> array<f32, 9> {
    var p: array<f32, 9>;
    let x2 = x * x;
    if (m > 0 && x2 >= 1.0) { return p; }
    var start = 0.282094791773878;
    if (m > 0) {
        var product = 1.0;
        for (var j = 1; j <= m; j++) {
            product *= (1.0 - x2) * f32(2 * j - 1) / f32(2 * j);
        }
        start *= sqrt(f32(2 * m + 1) * product);
    }
    if ((m & 1) == 1) { start = -start; }
    p[m] = start;
    if (m == SH_LMAX) { return p; }
    var f = sqrt(f32(2 * m + 3));
    p[m + 1] = x * f * p[m];
    for (var n = m + 2; n <= SH_LMAX; n++) {
        p[n] = x * p[n - 1] - p[n - 2] / f;
        f = sqrt(f32(4 * n * n - 1) / f32(n * n - m * m));
        p[n] *= f;
    }
    return p;
}

// All 45 basis functions evaluated in unit direction `dir`.
fn sh_basis(dir: vec3f) -> array<f32, 45> {
    var y: array<f32, 45>;
    let rxy = length(dir.xy);
    let cp = select(1.0, dir.x / rxy, rxy > 0.0);
    let sp = select(0.0, dir.y / rxy, rxy > 0.0);

    var p = sh_legendre(0, dir.z);
    for (var l = 0; l <= SH_LMAX; l += 2) {
        y[l * (l + 1) / 2] = p[l];
    }

    var c0 = 1.0;
    var s0 = 0.0;
    for (var m = 1; m <= SH_LMAX; m++) {
        p = sh_legendre(m, dir.z);
        let c = c0 * cp - s0 * sp;
        let s = s0 * cp + c0 * sp;
        for (var l = m + (m & 1); l <= SH_LMAX; l += 2) {
            let center = l * (l + 1) / 2;
            y[center + m] = p[l] * 1.41421356237 * c;
            y[center - m] = p[l] * 1.41421356237 * s;
        }
        c0 = c;
        s0 = s;
    }
    return y;
}
