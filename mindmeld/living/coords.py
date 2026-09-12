"""Map annotated MaleCNS bodies with soma coordinates onto graph indices."""
from __future__ import annotations

import numpy as np
import pyarrow.feather as feather

from ..download import ROOT
from ..preprocess import ANN_FILE, RAW


def load_viz_neurons(nodes: np.ndarray, max_points: int = 8000, seed: int = 0) -> dict:
    """Return display neuron indices + 2D normalized coordinates from somaLocation."""
    ann = feather.read_table(RAW / ANN_FILE, memory_map=True)
    bodies = np.asarray(ann["bodyId"], dtype=np.int64)
    soma = ann["somaLocation"].to_pylist()
    superclass = np.array([str(x) if x is not None else "" for x in ann["superclass"].to_pylist()], dtype=object)

    xyz = []
    body_ok = []
    sc_ok = []
    for b, s, sc in zip(bodies, soma, superclass):
        if s is None or len(s) < 2:
            continue
        xyz.append((float(s[0]), float(s[1]), float(s[2]) if len(s) > 2 else 0.0))
        body_ok.append(int(b))
        sc_ok.append(sc)
    body_ok = np.asarray(body_ok, dtype=np.int64)
    xyz = np.asarray(xyz, dtype=np.float32)
    sc_ok = np.asarray(sc_ok, dtype=object)

    # Intersect with graph nodes
    pos = np.searchsorted(nodes, body_ok)
    in_graph = pos < len(nodes)
    in_graph[in_graph] = nodes[pos[in_graph]] == body_ok[in_graph]
    body_ok, xyz, sc_ok, pos = body_ok[in_graph], xyz[in_graph], sc_ok[in_graph], pos[in_graph]

    rng = np.random.default_rng(seed)
    if len(body_ok) > max_points:
        # Prefer central brain, then random fill
        prefer = np.where(sc_ok == "cb_intrinsic")[0]
        other = np.where(sc_ok != "cb_intrinsic")[0]
        n_pref = min(len(prefer), max_points // 2)
        take = list(rng.choice(prefer, size=n_pref, replace=False)) if len(prefer) else []
        remain = max_points - len(take)
        pool = np.concatenate([prefer, other]) if len(prefer) else other
        pool = np.setdiff1d(pool, np.asarray(take, dtype=np.int64), assume_unique=False)
        if remain > 0 and len(pool):
            take.extend(rng.choice(pool, size=min(remain, len(pool)), replace=False).tolist())
        take = np.asarray(take, dtype=np.int64)
        body_ok, xyz, sc_ok, pos = body_ok[take], xyz[take], sc_ok[take], pos[take]

    # Keep raw soma XYZ (µm-ish) and a normalized copy for projection.
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    xyz_n = ((xyz - lo) / span).astype(np.float32)
    # Legacy single-panel XY (same as triad XY plane).
    xy = xyz_n[:, :2].copy()

    # Region codes for community view
    regions = sorted(set(sc_ok.tolist()) | {""})
    region_to_id = {r: i for i, r in enumerate(regions)}
    region_ids = np.array([region_to_id[r] for r in sc_ok], dtype=np.int16)

    return {
        "body_id": body_ok,
        "index": pos.astype(np.int64),
        "xy": xy.astype(np.float32),
        "xyz": xyz_n,
        "region": sc_ok,
        "region_id": region_ids,
        "region_names": regions,
        "n": int(len(body_ok)),
    }


def sample_display_edges(csr, viz_index: np.ndarray, max_edges: int = 4000, seed: int = 1) -> np.ndarray:
    """Sample edges among display neurons as pairs of local viz indices. Shape [E,2]."""
    global_to_local = {int(g): i for i, g in enumerate(viz_index)}
    rng = np.random.default_rng(seed)
    pairs = []
    # sample neurons and take a few outgoing edges that land in viz set
    order = rng.permutation(len(viz_index))
    for li in order:
        gi = int(viz_index[li])
        start, end = int(csr.indptr[gi]), int(csr.indptr[gi + 1])
        if start == end:
            continue
        cols = csr.indices[start:end]
        for c in cols[:: max(1, len(cols) // 3)][:3]:
            loc = global_to_local.get(int(c))
            if loc is not None and loc != li:
                pairs.append((li, loc))
                if len(pairs) >= max_edges:
                    return np.asarray(pairs, dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32) if pairs else np.zeros((0, 2), dtype=np.int32)
