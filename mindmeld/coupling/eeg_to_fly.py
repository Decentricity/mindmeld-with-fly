"""Fixed seeded Win: EEG feature vector -> length-n fly drive."""
from __future__ import annotations

import hashlib

import numpy as np
import torch


class EEGToFly:
    """drive = Win @ e  (Win is n x k, sparse-ish dense row gains, frozen)."""

    def __init__(
        self,
        n: int,
        k: int = 4,
        seed: int = 0,
        scale: float = 0.25,
        density: float = 1.0,
        device: str = "cpu",
    ):
        self.n = n
        self.k = k
        self.seed = seed
        self.scale = scale
        self.density = float(np.clip(density, 0.01, 1.0))
        rng = np.random.default_rng(seed)
        w = rng.normal(0.0, scale, size=(n, k)).astype(np.float32)
        if self.density < 1.0:
            mask = rng.random(size=(n, k)) < self.density
            w = np.where(mask, w, 0.0).astype(np.float32)
        self.win = torch.as_tensor(w, device=device, dtype=torch.float32)
        self.device = device
        self.gain = 1.0
        self.enabled = True

    def hash(self) -> str:
        arr = self.win.detach().float().cpu().numpy().tobytes()
        return hashlib.sha256(arr).hexdigest()[:16]

    def project(self, e: np.ndarray) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(self.n, device=self.device, dtype=torch.float32)
        e_t = torch.as_tensor(np.asarray(e, dtype=np.float32).reshape(self.k), device=self.device)
        return (self.win @ e_t) * float(self.gain)
