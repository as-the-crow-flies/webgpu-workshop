"""Writes particles.npy: 150,000 particles (blobs, a ring and a filament).

One row per `Particle` from types.wgsl:

    struct Particle {       numpy dtype field
        pos: vec3f,         ("pos",    "<f4", 3)   bytes 0..12, inside [0, 1]^3
        weight: f32,        ("weight", "<f4")      bytes 12..16
    }

The shaders expect positions in the unit cube, so scale your own data first:
    pos = (pos - pos.min(0)) / (pos.max(0) - pos.min(0)).max()
Run with: python make_data.py
"""
from pathlib import Path

import numpy as np

particle = np.dtype([("pos", "<f4", 3), ("weight", "<f4")])
assert particle.itemsize == 16

rng = np.random.default_rng(7)
parts = []
for center, spread, count in [((0.3, 0.6, 0.4), 0.06, 40000), ((0.7, 0.4, 0.6), 0.1, 40000), ((0.5, 0.5, 0.5), 0.02, 10000)]:
    parts.append(rng.normal(center, spread, (count, 3)))

a = rng.uniform(0, 2 * np.pi, 40000)
ring = np.stack([0.5 + 0.3 * np.cos(a), 0.5 + 0.05 * np.sin(5 * a), 0.5 + 0.3 * np.sin(a)], axis=1)
parts.append(ring + rng.normal(0, 0.015, ring.shape))

u = rng.uniform(0, 1, 20000)
filament = np.stack([0.15 + 0.7 * u, 0.2 + 0.6 * u * u, 0.8 - 0.5 * u], axis=1)
parts.append(filament + rng.normal(0, 0.01, filament.shape))

particles = np.zeros(sum(len(p) for p in parts), dtype=particle)
particles["pos"] = np.clip(np.concatenate(parts), 0.0, 1.0)
particles["weight"] = 1.0

np.save(Path(__file__).parent / "particles.npy", particles)
