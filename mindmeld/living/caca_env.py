"""Tmux-safe libcaca environment — same rules as ~/bin/mplay-caca.

Do not invent a second policy. Inside tmux use slang (not ncurses) to avoid
alternate-screen flicker; keep tmux/screen TERM so libcaca skips blink hacks.
"""
from __future__ import annotations

import os


def apply_mplay_caca_env() -> dict[str, str]:
    """Apply mplay-caca defaults; return the effective env knobs for logging."""
    term = os.environ.get("TERM", "")
    if not (term.startswith("tmux") or term.startswith("screen")):
        os.environ.setdefault("TERM", "xterm-256color")

    in_tmux = bool(os.environ.get("TMUX"))
    if in_tmux:
        os.environ.setdefault("CACA_DRIVER", "slang")
    else:
        os.environ.setdefault("CACA_DRIVER", "ncurses")

    return {
        "TERM": os.environ.get("TERM", ""),
        "TMUX": "1" if in_tmux else "0",
        "CACA_DRIVER": os.environ.get("CACA_DRIVER", ""),
        "source": "/home/decentricity/bin/mplay-caca",
    }
