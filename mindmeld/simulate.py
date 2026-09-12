"""Phase-1 reservoir simulation CLI."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .download import ROOT
from .oruk import build_oruk499, load_oruk499
from .preprocess import build_full_graph, load_processed
from .reservoir import Projection, signals, step, to_torch

REPORTS = ROOT / "reports"


def _load_graph(graph: str, weighting: str):
    if graph == "oruk499":
        path = ROOT / "data/processed/oruk499/metadata.json"
        if not path.exists():
            build_oruk499()
        return load_oruk499()
    if graph == "full":
        path = ROOT / "data/processed" / weighting / "metadata.json"
        if not path.exists():
            build_full_graph(weighting)
        return load_processed(weighting)
    raise ValueError(graph)


def run(graph: str, weighting: str, steps: int, device: str, seed: int = 42) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    t_load0 = time.perf_counter()
    scipy_w, nodes, meta = _load_graph(graph, weighting)
    load_s = time.perf_counter() - t_load0
    n = scipy_w.shape[0]
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device == "cuda" and dev.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    w = to_torch(scipy_w, str(dev))
    proj = Projection(n, str(dev), seed=seed)
    x = torch.zeros(n, dtype=torch.float32, device=dev)
    u = torch.as_tensor(signals(steps), device=dev)

    # warm-up
    for i in range(min(5, steps)):
        x = step(w, x, proj(u[i]))
    if dev.type == "cuda":
        torch.cuda.synchronize()

    x = torch.zeros(n, dtype=torch.float32, device=dev)
    trace = []
    t0 = time.perf_counter()
    for i in range(steps):
        x = step(w, x, proj(u[i]))
        if i % max(1, steps // 200) == 0 or i == steps - 1:
            trace.append(x[: min(8, n)].detach().float().cpu().numpy())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    x_cpu = x.detach().float().cpu()
    finite = bool(torch.isfinite(x_cpu).all())
    report = {
        "graph": graph,
        "weighting": weighting if graph == "full" else "oruk_signed",
        "device": str(dev),
        "n_nodes": n,
        "n_edges": int(scipy_w.nnz),
        "steps": steps,
        "load_seconds": load_s,
        "wall_seconds": wall,
        "steps_per_sec": steps / wall if wall > 0 else None,
        "state_mean": float(x_cpu.mean()),
        "state_std": float(x_cpu.std()),
        "state_min": float(x_cpu.min()),
        "state_max": float(x_cpu.max()),
        "nan_inf": 0 if finite else int((~torch.isfinite(x_cpu)).sum()),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if dev.type == "cuda" else None,
        "meta": {
            k: meta.get(k)
            for k in (
                "spectral_radius_before",
                "spectral_radius_after",
                "target_spectral_radius",
                "scc_neurons",
                "connections",
                "synapses",
                "deviation_from_published",
                "weight_policy",
                "n_nodes",
                "nnz",
            )
            if k in meta
        },
    }

    # activity plot
    arr = np.stack(trace, axis=0)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(arr)
    ax.set_title(f"{graph} activity (first {arr.shape[1]} units)")
    ax.set_xlabel("sampled step")
    fig.tight_layout()
    plot_path = REPORTS / f"activity_{graph}_{weighting if graph=='full' else 'signed'}_{steps}.png"
    fig.savefig(plot_path)
    plt.close(fig)
    report["plot"] = str(plot_path)

    out = REPORTS / f"simulate_{graph}_{weighting if graph=='full' else 'signed'}_{steps}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="MaleCNS sparse reservoir simulation")
    p.add_argument("--graph", choices=["oruk499", "full"], required=True)
    p.add_argument("--weighting", choices=["unsigned", "fast_nt_signed"], default="unsigned")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    report = run(args.graph, args.weighting, args.steps, args.device, args.seed)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
