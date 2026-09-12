"""Inspect Feather schemas / build cached sparse MaleCNS tensors.

Memory-conscious: chunked edge scans, int8 NT signs (no object string arrays),
reuse unsigned node index when present, and abort if MemAvailable gets too low.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import scipy.sparse as sp

from .download import FILES, ROOT
from .spectral import scale_to_radius
from .weights import SIGNED_NT, normalize_nt, unsigned_weights

RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports"
TMP = ROOT / "data/tmp"

WEIGHT_FILE = FILES[0]
NT_FILE = FILES[1]
ANN_FILE = FILES[2]

# Leave headroom for other tmux panes + systemd-oomd (fires ~50% pressure).
MIN_AVAILABLE_GB = 10.0
CHUNK = 1_000_000


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def guard_ram(label: str, min_gb: float = MIN_AVAILABLE_GB) -> None:
    avail = mem_available_gb()
    if avail >= min_gb:
        print(f"[mem] {label}: MemAvailable={avail:.1f}G", flush=True)
        return
    gc.collect()
    time.sleep(1)
    avail = mem_available_gb()
    print(f"[mem] {label}: low MemAvailable={avail:.1f}G after gc", flush=True)
    if avail < min_gb:
        raise MemoryError(
            f"Aborting at {label}: MemAvailable {avail:.1f}G < {min_gb}G "
            "(refusing to risk systemd-oomd killing a tmux pane)"
        )


def _pick(names, *candidates):
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"None of {candidates} in {list(names)}")


def load_tables():
    weights = feather.read_table(RAW / WEIGHT_FILE, memory_map=True)
    nt = feather.read_table(RAW / NT_FILE, memory_map=True)
    ann = feather.read_table(RAW / ANN_FILE, memory_map=True)
    return weights, nt, ann


def column_map(weights, nt, ann):
    wnames = weights.column_names
    nnames = nt.column_names
    anames = ann.column_names
    return {
        "pre": _pick(wnames, "bodyId_pre", "body_pre", "pre", "i", "src"),
        "post": _pick(wnames, "bodyId_post", "body_post", "post", "j", "dst"),
        "weight": _pick(wnames, "weight", "Weight", "count", "n", "synapses"),
        "nt_body": _pick(nnames, "bodyId", "body_id", "body", "id"),
        "nt_pred": _pick(
            nnames,
            "predicted_nt",
            "neurotransmitter",
            "neurotransmitter_predicted",
            "top_nt",
            "nt",
            "consensus",
        ),
        "ann_body": _pick(anames, "bodyId", "body_id", "body", "id"),
    }


def inspect_schemas() -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in FILES:
        table = feather.read_table(RAW / name, memory_map=True)
        info = {
            "rows": table.num_rows,
            "schema": str(table.schema),
            "columns": table.column_names,
            "sample": table.slice(0, 3).to_pylist(),
        }
        report[name] = info
    (REPORTS / "schemas.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _nt_sign_tables(nt_t, cols) -> tuple[np.ndarray, np.ndarray, str, dict]:
    """Sorted body ids + int8 signs (+1/-1/0). No per-edge string arrays."""
    nt_col = "consensus_nt" if "consensus_nt" in nt_t.column_names else cols["nt_pred"]
    bodies = np.asarray(nt_t[cols["nt_body"]], dtype=np.int64)
    labels = nt_t[nt_col].to_pylist()
    signs = np.zeros(len(bodies), dtype=np.int8)
    counts: dict[str, int] = {}
    for i, lab in enumerate(labels):
        key = normalize_nt(lab) or "unknown"
        counts[key] = counts.get(key, 0) + 1
        signs[i] = int(SIGNED_NT.get(key, 0.0))
    order = np.argsort(bodies, kind="mergesort")
    return bodies[order], signs[order], nt_col, counts


def _lookup_signs(nt_bodies: np.ndarray, nt_signs: np.ndarray, pre: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(nt_bodies, pre)
    out = np.zeros(len(pre), dtype=np.int8)
    ok = pos < len(nt_bodies)
    # avoid advanced-index side effects on boolean mask
    check = np.zeros(len(pre), dtype=bool)
    check[ok] = nt_bodies[pos[ok]] == pre[ok]
    out[check] = nt_signs[pos[check]]
    return out


def _save_csr(tag: str, bodies: np.ndarray, w: sp.csr_matrix, report: dict) -> dict:
    out = PROCESSED / tag
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "nodes.npy", bodies)
    np.save(out / "csr_indptr.npy", w.indptr.astype(np.int32, copy=False))
    np.save(out / "csr_indices.npy", w.indices.astype(np.int32, copy=False))
    np.save(out / "edge_weights.npy", w.data.astype(np.float32, copy=False))
    np.savez_compressed(out / "body_id_to_index.npz", body_id=bodies, index=np.arange(len(bodies), dtype=np.int32))
    (out / "metadata.json").write_text(json.dumps(report, indent=2, default=str))
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"preprocess_{tag}.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def build_full_graph(weighting: str = "unsigned", target_radius: float = 0.9) -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    guard_ram(f"start {weighting}")

    if weighting == "fast_nt_signed":
        return _build_signed_chunked(target_radius=target_radius)
    if weighting != "unsigned":
        raise ValueError(f"Unknown weighting: {weighting}")

    weights_t, nt_t, ann_t = load_tables()
    cols = column_map(weights_t, nt_t, ann_t)
    report = {
        "weighting": weighting,
        "n_edges_raw": weights_t.num_rows,
        "columns": cols,
        "ann_rows": ann_t.num_rows,
        "nt_rows": nt_t.num_rows,
        "weight_policy": "all edges positive; magnitude=log1p(count); engineering baseline",
    }
    del nt_t, ann_t
    gc.collect()
    guard_ram("unsigned after metadata")

    # Chunked unique + CSR assembly via temp memmaps to avoid giant Python peaks.
    n_edges = weights_t.num_rows
    # Pass 1: collect unique bodies via sorted merge of chunk uniques (disk-backed).
    uniq_path = TMP / "unsigned_chunk_uniqs"
    if uniq_path.exists():
        import shutil

        shutil.rmtree(uniq_path)
    uniq_path.mkdir()
    chunk_files = []
    for start in range(0, n_edges, CHUNK):
        guard_ram(f"unsigned uniq chunk {start}")
        sl = weights_t.slice(start, min(CHUNK, n_edges - start))
        pre = np.asarray(sl[cols["pre"]], dtype=np.int64)
        post = np.asarray(sl[cols["post"]], dtype=np.int64)
        u = np.unique(np.concatenate([pre, post]))
        fp = uniq_path / f"{start:09d}.npy"
        np.save(fp, u)
        chunk_files.append(fp)
        del pre, post, u, sl
    # merge unique files hierarchically
    bodies = None
    for fp in chunk_files:
        u = np.load(fp)
        bodies = u if bodies is None else np.unique(np.concatenate([bodies, u]))
        del u
        guard_ram("unsigned merging uniques")
    assert bodies is not None
    n = len(bodies)
    report["n_nodes"] = n
    print(f"unsigned nodes={n}", flush=True)

    # Pass 2: write COO pieces then build CSR
    rows_mm = np.memmap(TMP / "u_rows.mmap", dtype=np.int32, mode="w+", shape=(n_edges,))
    cols_mm = np.memmap(TMP / "u_cols.mmap", dtype=np.int32, mode="w+", shape=(n_edges,))
    data_mm = np.memmap(TMP / "u_data.mmap", dtype=np.float32, mode="w+", shape=(n_edges,))
    for start in range(0, n_edges, CHUNK):
        guard_ram(f"unsigned fill chunk {start}")
        end = min(start + CHUNK, n_edges)
        sl = weights_t.slice(start, end - start)
        pre = np.asarray(sl[cols["pre"]], dtype=np.int64)
        post = np.asarray(sl[cols["post"]], dtype=np.int64)
        counts = np.asarray(sl[cols["weight"]], dtype=np.float32)
        rows_mm[start:end] = np.searchsorted(bodies, post).astype(np.int32, copy=False)
        cols_mm[start:end] = np.searchsorted(bodies, pre).astype(np.int32, copy=False)
        data_mm[start:end] = unsigned_weights(counts)
        del pre, post, counts, sl
    rows_mm.flush()
    cols_mm.flush()
    data_mm.flush()
    del weights_t
    gc.collect()
    guard_ram("unsigned before CSR")

    w = sp.csr_matrix((np.asarray(data_mm), (np.asarray(rows_mm), np.asarray(cols_mm))), shape=(n, n))
    del rows_mm, cols_mm, data_mm
    gc.collect()
    w.sum_duplicates()
    w.sort_indices()
    guard_ram("unsigned before spectral")
    w, before, after = scale_to_radius(w, target=target_radius)
    report["spectral_radius_before"] = before
    report["spectral_radius_after"] = after
    report["target_spectral_radius"] = target_radius
    report["nnz"] = int(w.nnz)
    _save_csr("unsigned", bodies, w, report)
    # cleanup temp
    for p in TMP.glob("u_*.mmap"):
        p.unlink(missing_ok=True)
    import shutil

    shutil.rmtree(uniq_path, ignore_errors=True)
    return report


def _build_signed_chunked(target_radius: float = 0.9) -> dict:
    """Build signed graph without a 152M object string array."""
    guard_ram("signed start")
    unsigned_nodes = PROCESSED / "unsigned" / "nodes.npy"
    if not unsigned_nodes.exists():
        raise FileNotFoundError("Need unsigned nodes cache first (run unsigned preprocess)")

    bodies = np.load(unsigned_nodes)
    n = len(bodies)
    print(f"reusing unsigned nodes={n}", flush=True)

    weights_t = feather.read_table(RAW / WEIGHT_FILE, memory_map=True)
    nt_t = feather.read_table(RAW / NT_FILE, memory_map=True)
    ann_t = feather.read_table(RAW / ANN_FILE, memory_map=True)
    cols = column_map(weights_t, nt_t, ann_t)
    ann_rows = ann_t.num_rows
    del ann_t
    nt_bodies, nt_signs, nt_col, nt_counts = _nt_sign_tables(nt_t, cols)
    del nt_t
    gc.collect()
    guard_ram("signed after NT tables")

    n_edges = weights_t.num_rows
    # Over-allocate temp kept-edge buffers on disk; we fill densely then truncate.
    # First count kept edges with a cheap pass.
    kept = 0
    by_sign = {1: 0, -1: 0}
    for start in range(0, n_edges, CHUNK):
        guard_ram(f"signed count chunk {start}")
        end = min(start + CHUNK, n_edges)
        sl = weights_t.slice(start, end - start)
        pre = np.asarray(sl[cols["pre"]], dtype=np.int64)
        signs = _lookup_signs(nt_bodies, nt_signs, pre)
        m = signs != 0
        kept += int(m.sum())
        by_sign[1] += int((signs == 1).sum())
        by_sign[-1] += int((signs == -1).sum())
        del pre, signs, m, sl
    print(f"signed kept edges={kept} / {n_edges}", flush=True)
    guard_ram("signed after count pass")

    rows_mm = np.memmap(TMP / "s_rows.mmap", dtype=np.int32, mode="w+", shape=(kept,))
    cols_mm = np.memmap(TMP / "s_cols.mmap", dtype=np.int32, mode="w+", shape=(kept,))
    data_mm = np.memmap(TMP / "s_data.mmap", dtype=np.float32, mode="w+", shape=(kept,))
    cursor = 0
    for start in range(0, n_edges, CHUNK):
        guard_ram(f"signed fill chunk {start}")
        end = min(start + CHUNK, n_edges)
        sl = weights_t.slice(start, end - start)
        pre = np.asarray(sl[cols["pre"]], dtype=np.int64)
        post = np.asarray(sl[cols["post"]], dtype=np.int64)
        counts = np.asarray(sl[cols["weight"]], dtype=np.float32)
        signs = _lookup_signs(nt_bodies, nt_signs, pre)
        m = signs != 0
        if not np.any(m):
            del pre, post, counts, signs, m, sl
            continue
        n_keep = int(m.sum())
        rows_mm[cursor : cursor + n_keep] = np.searchsorted(bodies, post[m]).astype(np.int32, copy=False)
        cols_mm[cursor : cursor + n_keep] = np.searchsorted(bodies, pre[m]).astype(np.int32, copy=False)
        data_mm[cursor : cursor + n_keep] = signs[m].astype(np.float32) * np.log1p(counts[m])
        cursor += n_keep
        del pre, post, counts, signs, m, sl
    assert cursor == kept
    rows_mm.flush()
    cols_mm.flush()
    data_mm.flush()
    del weights_t, nt_bodies, nt_signs
    gc.collect()
    guard_ram("signed before CSR")

    w = sp.csr_matrix((np.asarray(data_mm), (np.asarray(rows_mm), np.asarray(cols_mm))), shape=(n, n))
    del rows_mm, cols_mm, data_mm
    gc.collect()
    w.sum_duplicates()
    w.sort_indices()
    guard_ram("signed before spectral")
    w, before, after = scale_to_radius(w, target=target_radius)

    report = {
        "weighting": "fast_nt_signed",
        "n_nodes": n,
        "n_edges_raw": n_edges,
        "n_edges_after_sign_filter": kept,
        "ann_rows": ann_rows,
        "nt_rows": int(sum(nt_counts.values())),
        "columns": cols,
        "weight_policy": {
            "nt_column": nt_col,
            "edges_total": n_edges,
            "edges_signed_kept": kept,
            "edges_zeroed_unknown_nt": n_edges - kept,
            "pre_edges_positive_ach": by_sign[1],
            "pre_edges_negative_gaba_glu": by_sign[-1],
            "neuron_nt_counts": dict(sorted(nt_counts.items(), key=lambda kv: -kv[1])),
            "sign_map": {"acetylcholine": +1, "gaba": -1, "glutamate": -1},
            "policy": "unknown/other transmitters zeroed (not invented)",
        },
        "spectral_radius_before": before,
        "spectral_radius_after": after,
        "target_spectral_radius": target_radius,
        "nnz": int(w.nnz),
        "memory_strategy": "chunked memmap; int8 NT signs; reused unsigned nodes",
    }
    _save_csr("fast_nt_signed", bodies, w, report)
    for p in TMP.glob("s_*.mmap"):
        p.unlink(missing_ok=True)
    return report


def load_processed(weighting: str):
    out = PROCESSED / weighting
    meta = json.loads((out / "metadata.json").read_text())
    indptr = np.load(out / "csr_indptr.npy")
    indices = np.load(out / "csr_indices.npy")
    data = np.load(out / "edge_weights.npy")
    nodes = np.load(out / "nodes.npy")
    w = sp.csr_matrix((data, indices, indptr), shape=(len(nodes), len(nodes)))
    return w, nodes, meta


def main():
    schemas = inspect_schemas()
    print(json.dumps({k: {"rows": v["rows"], "columns": v["columns"]} for k, v in schemas.items()}, indent=2))
    for mode in ("unsigned", "fast_nt_signed"):
        meta = PROCESSED / mode / "metadata.json"
        if meta.exists():
            print("Skipping existing", mode, flush=True)
            continue
        print("Building", mode, flush=True)
        print(json.dumps(build_full_graph(mode), indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
