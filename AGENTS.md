# mindmeld-with-fly

GitHub: https://github.com/Decentricity/mindmeld-with-fly

# Connectome (mindmeld Phase 1)

Memorable path: `/home/decentricity/mindmeld-with-fly`  
Package: `mindmeld`  
Env: `source /home/decentricity/mindmeld-with-fly/.venv/bin/activate`

## Setup

```bash
cd /home/decentricity/mindmeld-with-fly
source .venv/bin/activate
python -m mindmeld.download
python -m mindmeld.environment
python -m mindmeld.preprocess   # schemas + unsigned + fast_nt_signed caches
python -m mindmeld.oruk         # Oruk-like ~499 SCC approximation
```

## Simulate

```bash
python -m mindmeld.simulate --graph oruk499 --steps 10000 --device cuda
python -m mindmeld.simulate --graph full --weighting unsigned --steps 10000 --device cuda
python -m mindmeld.simulate --graph full --weighting fast_nt_signed --steps 10000 --device cuda
```

## Tests / acceptance

```bash
pytest -q
python -m mindmeld.accept
```

Phase 1 is a hard checkpoint. Do not start EEG / stick-figure work until acceptance prints `PHASE 1: PASS`.

## Phase 1.5B

In tmux, use the same libcaca policy as `mplay-caca` (`CACA_DRIVER=slang`):

```bash
living-caca --graph full
# or:
python -m mindmeld.living --renderer caca --graph full
python -m mindmeld.living --renderer caca --graph full --demo --seconds 12 --record recordings/living_demo.npz
python -m mindmeld.living --replay recordings/living_demo.npz --seconds 8
```

Camera: **Tab** toggles `triad` (XY|XZ|YZ + orbit peek) ↔ `orbit` (single rotatable view). In orbit: `,`/`.` yaw, `j`/`k` pitch.

Toggle announcements use `/home/decentricity/bin/say-alert` (RHVoice / speech-dispatcher — **not** Piper / GPU). Pass `--mute` to silence; `--demo` is muted by default.

**R** records activity NPZ **and** headless GPU cinema MP4 by default (`recordings/living_cinema_<timestamp>.mp4`). Opt out with `--no-cinema`. Overrides: `--cinema-mp4`, `--cinema-size 1280x720`, `--cinema-every N`.

Phase II EEG mind-meld is developed in `https://github.com/Decentricity/mindmeld-with-fly` (local: `/home/decentricity/mindmeld-with-fly`). Fruit-fly swarm remains back-burner.
