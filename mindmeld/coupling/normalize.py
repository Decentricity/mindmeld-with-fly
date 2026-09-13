"""Rolling baseline normalization for EEG feature vectors."""
from __future__ import annotations

import numpy as np


class RollingNormalizer:
    """Z-score with warm-up; returns clipped features for Win."""

    def __init__(self, dim: int = 5, window: int = 250, clip: float = 3.0):
        self.dim = dim
        self.window = max(8, window)
        self.clip = clip
        self.buf = np.zeros((self.window, dim), dtype=np.float32)
        self.n = 0
        self.i = 0

    def update(self, e: np.ndarray) -> np.ndarray:
        e = np.asarray(e, dtype=np.float32).reshape(self.dim)
        self.buf[self.i % self.window] = e
        self.i += 1
        self.n = min(self.n + 1, self.window)
        if self.n < 8:
            return e.copy()
        sl = self.buf[: self.n]
        mu = sl.mean(axis=0)
        sd = sl.std(axis=0) + 1e-6
        z = (e - mu) / sd
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "dim": self.dim,
            "window": self.window,
            "clip": self.clip,
            "buf": self.buf.copy(),
            "n": self.n,
            "i": self.i,
        }
