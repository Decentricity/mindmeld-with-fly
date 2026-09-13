# mindmeld-with-fly

Sparse Drosophila MaleCNS reservoir + live Muse EEG mind-meld on libcaca (`living-caca`).

GitHub: https://github.com/Decentricity/mindmeld-with-fly  
Local: `/home/decentricity/mindmeld-with-fly`  
Env: `source /home/decentricity/mindmeld-with-fly/.venv/bin/activate`

## What this is

- **Phase 1:** build/preprocess MaleCNS graphs, run reservoir sims, accept.
- **Phase 1.5B:** terminal living connectome (triad/orbit, stim, cinema record).
- **Phase II:** Muse EEG → fixed seeded `Win` → **one** fly reservoir, with a HUMAN scalp/bands pane beside the fly viz.

Honest limits: reservoir dynamics are not biophysics; fly “EEG” bands are **spectral content of the digital reservoir**, not insect electrophysiology.

## Mind-meld quick start

In tmux, use the same libcaca policy as `mplay-caca` (`CACA_DRIVER=slang`):

```bash
cd /home/decentricity/mindmeld-with-fly
source .venv/bin/activate

# Live Muse (default MAC for Muse-E946)
./scripts/mindmeld-caca --mac 00:55:DA:B5:E9:46 --graph full --mute

# Synthetic bands (no headset)
./scripts/mindmeld-caca --eeg-source synthetic --graph full --mute --seconds 12

# Plain living connectome (no EEG)
./scripts/living-caca --graph full
```

Extra EEG keys: `e` inject on/off · `t` rest silent/tonic · `m` mark.  
Living keys unchanged: `Tab` camera · `r` reset · `R` record · `,`/`.`/`j`/`k` orbit.

On quit (or after a timed run), meld metrics land under `recordings/meld_<timestamp>/` (`events.jsonl`, `config.json`, `metrics.npz`).

## Drive vector (human → fly)

```text
e[t] = [delta, theta, alpha, beta, gamma]   # 5-D; same Hz edges as Muse viewer
drive = Win @ e[t]                          # sparse Win on full MaleCNS (~8192 targets)
```

Band edges (Hz): δ 1–4 · θ 4–8 · α 8–13 · β 13–30 · γ 30–45.

Changing to 5-D starts a **new experiment config** (`win_hash` differs from older 4-D runs).

## Fly spectral response

Each tick appends mean `|x|` on viz somata into a rolling buffer, then Welch/band-power with the **same band edges** as human EEG. The UI shows **HUM D/T/A/B/G** next to **FLY (reservoir)** meters/traces for stimulus vs response.

Logged in `metrics.npz`: `e`, `e_raw`, `fly_rel`, `fly_abs` (shape `(N, 5)`).

## Phase 1 reproduce

```bash
cd /home/decentricity/mindmeld-with-fly
source .venv/bin/activate
python -m mindmeld.download
python -m mindmeld.environment
python -m mindmeld.preprocess
python -m mindmeld.oruk
python -m mindmeld.simulate --graph oruk499 --steps 10000 --device cuda
python -m mindmeld.simulate --graph full --weighting unsigned --steps 10000 --device cuda
python -m mindmeld.simulate --graph full --weighting fast_nt_signed --steps 10000 --device cuda
pytest -q
python -m mindmeld.accept
```

Reports → `reports/`. Processed tensors → `data/processed/`.

## Changelog

### 2026-09-13 — Human gamma + fly spectral response

- **Human drive/display is 5-D:** gamma (30–45 Hz) is included in `BANDS` / `FEATURE_NAMES`, live Muse + synthetic sources, and `Win` (`n×5`). Older 4-band recordings pad to 5 on replay.
- **Fly reservoir spectra:** new `mindmeld/coupling/fly_spectrum.py`; `MeldSession` exposes `last_fly_abs` / `last_fly_rel` each tick.
- **UI:** HUMAN pane meters/graphs are **D/T/A/B/G**; side-by-side **FLY (reservoir)** meters + dual time traces; HUD shows both human and fly band summaries; disclaimer states fly bands are not bio EEG.
- **Logging:** throttled events and `metrics.npz` store human `e`/`e_raw` and fly `fly_rel`/`fly_abs`.
- **ASCII fly art:** side-view FlyBrains-derived art overlay (`mindmeld/living/ascii_fly.py`, `mindmeld/living/art/`).

### Earlier

- EEG mind-meld on top of living-caca (Muse → sparse `Win` → one MaleCNS reservoir).
- Repo seeded from working connectome living-caca (force-replaced history).
