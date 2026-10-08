"""Writes nodes.npy and edges.npy: a graph of 360 nodes in 6 communities.

nodes.npy, one row per `Node` from types.wgsl:

    struct Node {           numpy dtype field
        pos: vec3f,         ("pos",    "<f4", 3)   bytes 0..12
        radius: f32,        ("radius", "<f4")      bytes 12..16  (fills the gap after the vec3)
        vel: vec3f,         ("vel",    "<f4", 3)   bytes 16..28
        group: u32,         ("group",  "<u4")      bytes 28..32
    }

The velocity is part of the file so the simulation can keep it in the same
buffer; it simply starts at zero.

edges.npy, one row per `Edge`:

    struct Edge {
        a: u32,             ("a", "<u4")   index into nodes.npy
        b: u32,             ("b", "<u4")
    }

If you have a networkx graph `G`, the edges are:
    np.array(list(G.edges()), dtype="<u4").view(edge).reshape(-1)
(after relabelling the nodes to 0 .. n-1 with nx.convert_node_labels_to_integers).
Run with: python make_data.py
"""
from pathlib import Path

import numpy as np

node = np.dtype([("pos", "<f4", 3), ("radius", "<f4"), ("vel", "<f4", 3), ("group", "<u4")])
edge = np.dtype([("a", "<u4"), ("b", "<u4")])
assert node.itemsize == 32 and edge.itemsize == 8

rng = np.random.default_rng(7)
n, groups = 360, 6
group = np.sort(rng.integers(0, groups, n))

# Mostly connect nodes within their own community, sometimes across.
pairs = set()
for i in range(n):
    for _ in range(3):
        if rng.random() < 0.92:
            j = rng.choice(np.flatnonzero(group == group[i]))
        else:
            j = rng.integers(0, n)
        if i != j:
            pairs.add((min(i, j), max(i, j)))

# An (E, 2) uint32 array, viewed as E structs of two u32s.
edges = np.array(sorted(pairs), dtype="<u4").view(edge).reshape(-1)

nodes = np.zeros(n, dtype=node)
nodes["pos"] = rng.uniform(-1, 1, (n, 3))   # random start; the simulation lays it out
nodes["group"] = group
degree = np.bincount(np.concatenate([edges["a"], edges["b"]]), minlength=n)
nodes["radius"] = 0.02 + 0.006 * np.sqrt(degree)

folder = Path(__file__).parent
np.save(folder / "nodes.npy", nodes)
np.save(folder / "edges.npy", edges)
