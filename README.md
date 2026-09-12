# mindmeld / connectome — Phase 1

Sparse MaleCNS reservoir on RTX 4070 Ti.

## Paths

- Project: `/home/decentricity/connectome`
- Symlink: `/home/decentricity/mindmeld`
- Env: `/home/decentricity/connectome/.venv`

## Exact reproduction

```bash
cd /home/decentricity/connectome
source .venv/bin/activate
python -m mindmeld.download
python -m mindmeld.environment
python -m mindmeld.preprocess
python -m mindmeld.oruk
python -m mindmeld.simulate --graph oruk499 --steps 10000 --device cuda
python -m mindmeld.simulate --graph full --weighting unsigned --steps 100 --device cuda
python -m mindmeld.simulate --graph full --weighting unsigned --steps 1000 --device cuda
python -m mindmeld.simulate --graph full --weighting unsigned --steps 10000 --device cuda
python -m mindmeld.simulate --graph full --weighting fast_nt_signed --steps 10000 --device cuda
pytest -q
python -m mindmeld.accept
```

Reports land in `reports/`. Processed sparse tensors in `data/processed/`.

## Phase 1.5B — living connectome (terminal)

In tmux, libcaca needs the **same policy as `mplay-caca`**: `CACA_DRIVER=slang` (not ncurses), keep `tmux*`/`screen*` TERM, and prefer pane `alternate-screen off`. That fix is documented in the login CLI reminder (`mplay-caca FILE`).

```bash
cd /home/decentricity/connectome
source .venv/bin/activate

# Preferred in tmux (wrapper applies mplay-caca env):
living-caca --graph full

# Equivalent:
python -m mindmeld.living --renderer caca --graph full

# ANSI-only (no libcaca display):
python -m mindmeld.living --renderer ansi --graph oruk499

# Demo / record / replay:
living-caca --graph full --demo --seconds 12 --record recordings/living_demo.npz
python -m mindmeld.living --replay recordings/living_demo.npz --seconds 8
# Optional: turn a screen capture into ASCII video with the existing tool:
#   mplay-caca recordings/some_capture.mp4
```

Keys: `space` pause · `n` stim mode · `v` view · `[`/`]` intensity · `+`/`-` speed · `i` impulse · `r` reset · `R` record · `q` quit

Phase II (fruit-fly swarm) and original Phase 2 (EEG mind-meld) stay on the back burner until you ask.
