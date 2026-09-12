# Decisions

- Project path `/home/decentricity/connectome` (symlink `~/mindmeld`); Python package name remains `mindmeld`.
- Use official minconf-0.5 flat weights rather than multi-GB synapse tables.
- Never densify the full adjacency; CSR only.
- Graph orientation: `W[post, pre]` so `state_post += W @ state_pre`.
- Full-graph node set = all body IDs appearing in the weights table (88.4M), as published — not silently restricted to annotations.
- Signed NT uses `consensus_nt` when present; unknown NT edges dropped.
- Abort preprocess if `MemAvailable < 10 GiB` to avoid systemd-oomd killing tmux panes.
