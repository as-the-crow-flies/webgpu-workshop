"""Writes bodies.npy: 4096 bodies in a rotating disk around a heavy center.

One row per `Body` from types.wgsl:

    struct Body {           numpy dtype field
        pos: vec3f,         ("pos",   "<f4", 3)   bytes 0..12
        mass: f32,          ("mass",  "<f4")      bytes 12..16  (fills the gap after the vec3)
        vel: vec3f,         ("vel",   "<f4", 3)   bytes 16..28
        size: f32,          ("size",  "<f4")      bytes 28..32
        color: vec4f,       ("color", "<f4", 4)   bytes 32..48
    }

The velocity is part of the data: the simulation needs a state to update.
Units: G = 1, the central mass is 1 (CENTRAL_MASS in simulate.wgsl), the disk mass is 0.5.
Run with: python make_data.py
"""
from pathlib import Path

import numpy as np

body = np.dtype([("pos", "<f4", 3), ("mass", "<f4"), ("vel", "<f4", 3), ("size", "<f4"), ("color", "<f4", 4)])
assert body.itemsize == 48  # the size of `struct Body` in WGSL

rng = np.random.default_rng(7)
n = 4096
central_mass, disk_mass = 1.0, 0.5

# Radii between 0.2 and 1.5, denser towards the center; a thin disk in the xz-plane.
r = 0.2 + rng.gamma(2.0, 0.3, n).clip(0, 1.3)
angle = rng.uniform(0, 2 * np.pi, n)
pos = np.stack([r * np.cos(angle), rng.normal(0, 0.02, n), r * np.sin(angle)], axis=1)

# Circular orbits: v = sqrt(G * M(<r) / r), with M(<r) the center plus the disk mass inside r.
enclosed = central_mass + disk_mass * np.argsort(np.argsort(r)) / n
speed = np.sqrt(enclosed / r)
tangent = np.stack([-np.sin(angle), np.zeros(n), np.cos(angle)], axis=1)

bodies = np.zeros(n, dtype=body)
bodies["pos"] = pos
bodies["mass"] = disk_mass / n
bodies["vel"] = tangent * speed[:, None]
bodies["size"] = rng.uniform(0.006, 0.014, n)
inner = np.array([1.0, 0.8, 0.5, 1.0])
outer = np.array([0.4, 0.6, 1.0, 1.0])
t = ((r - 0.2) / 1.3)[:, None]
bodies["color"] = (1 - t) * inner + t * outer

np.save(Path(__file__).parent / "bodies.npy", bodies)
