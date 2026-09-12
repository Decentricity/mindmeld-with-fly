"""Connection weighting modes for MaleCNS edges."""
from __future__ import annotations

import numpy as np

SIGNED_NT = {
    "acetylcholine": 1.0,
    "ach": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "glu": -1.0,
}


def normalize_nt(name: str | None) -> str | None:
    if name is None:
        return None
    s = str(name).strip().lower()
    return s or None


def unsigned_weights(counts: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(counts, dtype=np.float32))


def signed_from_nt(counts: np.ndarray, pre_nt: np.ndarray) -> tuple[np.ndarray, dict]:
    """ACh+/GABA-/Glu-; other transmitters get weight 0 and are reported."""
    counts = np.asarray(counts, dtype=np.float32)
    labels = np.asarray(pre_nt, dtype=object).astype(str)
    uniq, inv, freq = np.unique(labels, return_inverse=True, return_counts=True)
    sign_of = np.array(
        [SIGNED_NT.get(normalize_nt(u) or "unknown", 0.0) for u in uniq],
        dtype=np.float32,
    )
    signs = sign_of[inv]
    weights = signs * np.log1p(counts)
    by_nt = {normalize_nt(u) or "unknown": int(c) for u, c in zip(uniq, freq)}
    kept = int((signs != 0).sum())
    report = {
        "edges_total": int(len(counts)),
        "edges_signed_kept": kept,
        "edges_zeroed_unknown_nt": int(len(counts) - kept),
        "pre_nt_counts": dict(sorted(by_nt.items(), key=lambda kv: -kv[1])),
        "sign_map": {"acetylcholine": +1, "gaba": -1, "glutamate": -1},
        "policy": "unknown/other transmitters zeroed (not invented)",
    }
    return weights, report
