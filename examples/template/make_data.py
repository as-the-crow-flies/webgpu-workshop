"""Writes points.npy: a spiral galaxy of 20,000 points.

One row of the file must be exactly one `Point` from types.wgsl:

    struct Point {          numpy dtype field
        pos: vec3f,         ("pos",   "<f4", 3)   bytes 0..12
        size: f32,          ("size",  "<f4")      bytes 12..16  (fills the gap after the vec3)
        color: vec4f,       ("color", "<f4", 4)   bytes 16..32
    }

"<f4" is a little-endian 32-bit float (f32); "<u4" would be a u32.
Run with: python make_data.py
"""
from pathlib import Path

import numpy as np

point = np.dtype([("pos", "<f4", 3), ("size", "<f4"), ("color", "<f4", 4)])
assert point.itemsize == 32  # the size of `struct Point` in WGSL

rng = np.random.default_rng(7)
n = 20000
arm = rng.integers(0, 3, n)
r = rng.gamma(2.0, 0.25, n)
angle = arm * 2 * np.pi / 3 + r * 2.5 + rng.normal(0, 0.25, n)

points = np.zeros(n, dtype=point)
points["pos"][:, 0] = r * np.cos(angle)
points["pos"][:, 1] = rng.normal(0, 0.04, n) * np.exp(-r)
points["pos"][:, 2] = r * np.sin(angle)
points["size"] = rng.uniform(0.004, 0.012, n)
palette = np.array([[1.0, 0.55, 0.3, 1], [0.4, 0.7, 1.0, 1], [0.9, 0.9, 1.0, 1]])
points["color"] = palette[arm] * rng.uniform(0.6, 1.0, (n, 1))
points["color"][:, 3] = 1.0

np.save(Path(__file__).parent / "points.npy", points)
