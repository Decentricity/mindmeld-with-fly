# Phase 1 report — MaleCNS sparse reservoir

## Factual (measured / from official sources)

- Hardware: NVIDIA GeForce RTX 4070 Ti, 12 GB; PyTorch `2.10.0+cu128`; CUDA runtime 12.8; `torch.cuda.is_available() == true` (see `environment.txt`).
- Official FlyEM MaleCNS v1.0 flat files downloaded + MD5-verified under `data/raw/` (weights ≈1.0 GB, NT ≈42 MB, annotations ≈14 MB).
- Feather schemas recorded in `schemas.json`.
- **Weights table as published:** 151,856,684 edges among **88,384,522** segment body IDs (`body_pre`, `body_post`, `weight`). This is the full segment graph, not only the ~211k annotated neurons.
- Annotations: 211,577 rows; NT table: 1,835,518 bodies.
- Oruk public numbers (meta + article): 499 neurons, 15,865 connections, 867,344 synapses, α=0.15, target spectral radius 0.9, ACh+/GABA−/Glu−.
- Simulations (CUDA, float32 sparse CSR):
  - Oruk-like: 506 nodes, 10,751 edges, 10k steps, ~31k steps/s, peak VRAM negligible, no NaN/Inf.
  - Full unsigned: 88,384,522 nodes / 151,856,684 nnz, ~48 steps/s, peak VRAM ≈4.40 GB, 10k steps OK.
  - Full `fast_nt_signed`: 134,319,992 nnz, ~52 steps/s, peak VRAM ≈4.26 GB, 10k steps OK.
- Benchmarks: `benchmarks.json`.

## Assumed / reconstructed (not claimed as biology)

- Recurrent dynamics: leaky-tanh echo-state form with α=0.15 (Oruk-style engineering dynamics), **not** biophysical compartments.
- `unsigned` weights: `log1p(count)`, all excitatory — engineering baseline only.
- `fast_nt_signed`: consensus NT signs ACh=+ / GABA=− / Glu=−; other transmitters **zeroed**, not invented.
- Spectral radius scaled to 0.9 via sparse power iteration / eigs on small graphs.
- Oruk-499 reconstruction: `superclass==cb_intrinsic`, traced/reviewed status, consensus NT in {ACh,GABA,Glu}, top-512 by synaptic contact strength, min 5 contacts, largest SCC → **506** neurons (Δ=+7 vs published 499; fewer edges than Oruk’s 15,865). Oruk did not publish body IDs; this is a reproducible approximation.
- Input drive: fixed seeded per-neuron channel gains on synthetic multichannel sinusoids (O(N) projection).

## Memory note

An earlier in-RAM signed build spiked ~19 GiB RSS and triggered `systemd-oomd` on a tmux pane. Signed preprocess was rewritten to chunked memmaps + int8 NT signs; rebuild kept MemAvailable ≈22–23 G.
