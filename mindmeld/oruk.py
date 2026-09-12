"""Oruk-like 499-neuron subgraph reconstruction from MaleCNS (approximate)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .download import ROOT
from .preprocess import ANN_FILE, NT_FILE, PROCESSED, RAW, WEIGHT_FILE, REPORTS, column_map, load_tables
from .spectral import scale_to_radius
from .weights import SIGNED_NT, normalize_nt, signed_from_nt

ORUK_PUBLISHED = {
    "source": "https://oruk.ai/research/we-taught-a-fruit-fly-to-read-human-emotion",
    "meta": "https://oruk.ai/connectome-reservoir/v1/meta.json",
    "candidates": 512,
    "scc_neurons": 499,
    "connections": 15865,
    "synapses": 867344,
    "min_contacts": 5,
    "alpha": 0.15,
    "target_spectral_radius": 0.9,
    "signs": {"acetylcholine": +1, "gaba": -1, "glutamate": -1},
}


def _ann_filter(ann, cols_ann_body: str) -> tuple[np.ndarray, dict]:
    """Select central-brain intrinsic candidates (MaleCNS superclass=cb_intrinsic)."""
    body_col = cols_ann_body
    bodies = np.asarray(ann[body_col], dtype=np.int64)
    notes = {"filters_applied": []}
    mask = np.ones(len(bodies), dtype=bool)

    if "superclass" in ann.column_names:
        vals = np.array([str(v) if v is not None else "" for v in ann["superclass"].to_pylist()], dtype=object)
        sm = vals == "cb_intrinsic"
        if int(sm.sum()) >= 512:
            mask &= sm
            notes["filters_applied"].append("superclass == cb_intrinsic")
        else:
            notes["filters_applied"].append(f"superclass==cb_intrinsic only {int(sm.sum())}; not applied alone")

    status_col = "statusLabel" if "statusLabel" in ann.column_names else ("status" if "status" in ann.column_names else None)
    if status_col is not None:
        vals = np.array([str(v).lower() if v is not None else "" for v in ann[status_col].to_pylist()], dtype=object)
        traced = np.array([("traced" in v) or ("reviewed" in v) or ("leaves" in v) for v in vals], dtype=bool)
        if int((mask & traced).sum()) >= 512:
            mask &= traced
            notes["filters_applied"].append(f"{status_col} in traced/reviewed/leaves")

    if int(mask.sum()) < 512:
        notes["fallback"] = f"filters left {int(mask.sum())} bodies (<512); using all annotated bodies"
        mask = np.ones(len(bodies), dtype=bool)

    notes["eligible_after_ann_filter"] = int(mask.sum())
    return bodies[mask], notes


def build_oruk499(target_radius: float = 0.9, min_contacts: int = 5, top_k: int = 512) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    weights_t, nt_t, ann_t = load_tables()
    cols = column_map(weights_t, nt_t, ann_t)

    cand_bodies, ann_notes = _ann_filter(ann_t, cols["ann_body"])
    cand_set = set(int(b) for b in cand_bodies)

    # Prefer consensus_nt when present, else predicted_nt
    nt_col = "consensus_nt" if "consensus_nt" in nt_t.column_names else cols["nt_pred"]
    nt_bodies = np.asarray(nt_t[cols["nt_body"]], dtype=np.int64)
    nt_preds = np.array([normalize_nt(p) for p in nt_t[nt_col].to_pylist()], dtype=object)
    order = np.argsort(nt_bodies)
    nt_bodies_s = nt_bodies[order]
    nt_preds_s = nt_preds[order]

    def lookup_nt(ids: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(nt_bodies_s, ids)
        out = np.full(len(ids), "unknown", dtype=object)
        ok = pos < len(nt_bodies_s)
        ok[ok] = nt_bodies_s[pos[ok]] == ids[ok]
        out[ok] = nt_preds_s[pos[ok]]
        return out

    modeled = set(SIGNED_NT)
    cand_arr = np.fromiter(cand_set, dtype=np.int64, count=len(cand_set))
    cand_nt = lookup_nt(cand_arr)
    nt_ok = set(int(b) for b, nt in zip(cand_arr, cand_nt) if nt in modeled)
    if len(nt_ok) >= top_k:
        cand_set = nt_ok
        ann_notes["filters_applied"].append(f"{nt_col} in {{acetylcholine,gaba,glutamate}}")
    else:
        ann_notes["filters_applied"].append(
            f"NT-modeled filter left {len(nt_ok)}; keeping broader candidate pool"
        )

    pre = np.asarray(weights_t[cols["pre"]], dtype=np.int64)
    post = np.asarray(weights_t[cols["post"]], dtype=np.int64)
    counts = np.asarray(weights_t[cols["weight"]], dtype=np.float32)

    # strength within candidates via two passes of bincount after mapping
    cand_list = np.fromiter(cand_set, dtype=np.int64, count=len(cand_set))
    cand_list.sort()
    # map pre/post that land in cand_set
    pre_pos = np.searchsorted(cand_list, pre)
    post_pos = np.searchsorted(cand_list, post)
    pre_in = pre_pos < len(cand_list)
    post_in = post_pos < len(cand_list)
    pre_in[pre_in] = cand_list[pre_pos[pre_in]] == pre[pre_in]
    post_in[post_in] = cand_list[post_pos[post_in]] == post[post_in]
    strength = np.zeros(len(cand_list), dtype=np.float64)
    np.add.at(strength, pre_pos[pre_in], counts[pre_in])
    np.add.at(strength, post_pos[post_in], counts[post_in])
    top_idx = np.argsort(-strength)[:top_k]
    ranked = cand_list[top_idx]
    ranked_set = set(int(b) for b in ranked)

    both = pre_in & post_in
    # both currently means in cand_list; restrict to ranked + min contacts
    ranked_sorted = np.sort(ranked)
    pre_r = np.searchsorted(ranked_sorted, pre)
    post_r = np.searchsorted(ranked_sorted, post)
    pre_ok = pre_r < len(ranked_sorted)
    post_ok = post_r < len(ranked_sorted)
    pre_ok[pre_ok] = ranked_sorted[pre_r[pre_ok]] == pre[pre_ok]
    post_ok[post_ok] = ranked_sorted[post_r[post_ok]] == post[post_ok]
    mask = pre_ok & post_ok & (counts >= min_contacts)
    pre_e, post_e, cnt_e = pre[mask], post[mask], counts[mask]
    pre_nt = lookup_nt(pre_e)
    wdata, nt_report = signed_from_nt(cnt_e, pre_nt)
    keep = wdata != 0
    pre_e, post_e, wdata, cnt_e = pre_e[keep], post_e[keep], wdata[keep], cnt_e[keep]

    # map to dense indices over ranked
    idx = {int(b): i for i, b in enumerate(ranked)}
    n = len(ranked)
    rows = np.fromiter((idx[int(b)] for b in post_e), dtype=np.int32, count=len(post_e))
    cols_i = np.fromiter((idx[int(b)] for b in pre_e), dtype=np.int32, count=len(pre_e))
    w = sp.coo_matrix((wdata, (rows, cols_i)), shape=(n, n)).tocsr()
    w.sum_duplicates()
    w.sort_indices()

    # largest SCC
    n_comp, labels = connected_components(w, directed=True, connection="strong")
    sizes = np.bincount(labels)
    scc_id = int(np.argmax(sizes))
    scc_mask = labels == scc_id
    scc_nodes = np.array(ranked, dtype=np.int64)[scc_mask]
    remap = -np.ones(n, dtype=np.int32)
    remap[np.where(scc_mask)[0]] = np.arange(int(scc_mask.sum()), dtype=np.int32)
    r = remap[rows]
    c = remap[cols_i]
    edge_keep = (r >= 0) & (c >= 0)
    w2 = sp.coo_matrix((wdata[edge_keep], (r[edge_keep], c[edge_keep])), shape=(int(scc_mask.sum()), int(scc_mask.sum()))).tocsr()
    w2.sum_duplicates()
    w2.sort_indices()
    w2, before, after = scale_to_radius(w2, target=target_radius)

    report = {
        "published": ORUK_PUBLISHED,
        "reconstruction_notes": {
            **ann_notes,
            "strength_definition": "total synaptic contact count as pre or post within candidate pool (proxy for unspecified 'strongest')",
            "exact_neuron_ids": "Oruk did not publish the 499 body IDs; this is a reproducible approximation, not a bit-exact replica",
            "nt_report": nt_report,
        },
        "candidates_top_k": top_k,
        "scc_neurons": int(scc_mask.sum()),
        "connections": int(w2.nnz),
        "synapses": float(cnt_e[edge_keep].sum()),
        "spectral_radius_before": before,
        "spectral_radius_after": after,
        "target_spectral_radius": target_radius,
        "min_contacts": min_contacts,
        "deviation_from_published": {
            "neurons_delta": int(scc_mask.sum()) - ORUK_PUBLISHED["scc_neurons"],
            "connections_delta": int(w2.nnz) - ORUK_PUBLISHED["connections"],
            "synapses_delta": float(cnt_e[edge_keep].sum()) - ORUK_PUBLISHED["synapses"],
        },
    }

    out = PROCESSED / "oruk499"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "nodes.npy", scc_nodes)
    np.save(out / "csr_indptr.npy", w2.indptr.astype(np.int32, copy=False))
    np.save(out / "csr_indices.npy", w2.indices.astype(np.int32, copy=False))
    np.save(out / "edge_weights.npy", w2.data.astype(np.float32, copy=False))
    (out / "metadata.json").write_text(json.dumps(report, indent=2, default=str))
    (REPORTS / "oruk499.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def load_oruk499():
    out = PROCESSED / "oruk499"
    meta = json.loads((out / "metadata.json").read_text())
    indptr = np.load(out / "csr_indptr.npy")
    indices = np.load(out / "csr_indices.npy")
    data = np.load(out / "edge_weights.npy")
    nodes = np.load(out / "nodes.npy")
    w = sp.csr_matrix((data, indices, indptr), shape=(len(nodes), len(nodes)))
    return w, nodes, meta


def main():
    print(json.dumps(build_oruk499(), indent=2, default=str))


if __name__ == "__main__":
    main()
