"""Writes timeseries.npy: 8 signals of 2048 samples each.

This is a plain float32 array of shape (series, samples), not a structured
dtype. Every float is one `Sample` from types.wgsl:

    struct Sample {
        value: f32,     a single "<f4"
    }

so sample t of series s is element s * samples + t of `array<Sample>`.
main.rs reads the number of series and samples from the shape.
Run with: python make_data.py
"""
from pathlib import Path

import numpy as np

rng = np.random.default_rng(7)
t = np.linspace(0, 20, 2048)
series = [
    np.sin(t),
    np.sin(3 * t) * np.exp(-0.1 * t),
    np.cumsum(rng.normal(0, 0.05, t.size)),            # a random walk
    np.sign(np.sin(1.3 * t)) * 0.8,                     # a square wave
    rng.normal(0, 0.3, t.size),                         # noise
    np.sin(t * t * 0.05),                               # a chirp
    (rng.random(t.size) > 0.98) * rng.uniform(0.5, 1.0, t.size),  # spikes
    np.cos(0.7 * t) + 0.3 * np.sin(5.3 * t),
]
# Scale every series to [-1, 1].
series = [s / (np.abs(s).max() + 1e-9) for s in series]

# numpy would use float64 by default; WGSL needs float32.
data = np.array(series, dtype=np.float32)
assert data.shape == (8, 2048)

np.save(Path(__file__).parent / "timeseries.npy", data)
