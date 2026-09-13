"""Fixed seeded Win: EEG feature vector -> length-n fly drive."""
from __future__ import annotations

import hashlib

import numpy as np
import torch


class EEGToFly:
    """drive = Win @ e.

    For small graphs Win is dense (n x k). For large MaleCNS, inject into a
    fixed seeded subset of neurons only — a dense 88M×5 Win is unusable live.
    """

    def __init__(
        self,
        n: int,
        k: int = 5,
        seed: int = 0,
        scale: float = 0.25,
        density: float = 1.0,
        device: str = "cpu",
        max_targets: int = 8192,
    ):
        self.n = n
        self.k = k
        self.seed = seed
        self.scale = scale
        self.device = device
        self.gain = 1.0
        self.enabled = True
        rng = np.random.default_rng(seed)

        # Auto-sparse when the connectome is too large for a dense adapter.
        if n > 100_000:
            t = int(min(max_targets, n))
            self.density = t / float(n)
            idx = rng.choice(n, size=t, replace=False).astype(np.int64)
            w = rng.normal(0.0, scale, size=(t, k)).astype(np.float32)
            self.idx = torch.as_tensor(idx, device=device, dtype=torch.long)
            self.win = torch.as_tensor(w, device=device, dtype=torch.float32)
            self.sparse = True
        else:
            self.density = float(np.clip(density, 0.01, 1.0))
            w = rng.normal(0.0, scale, size=(n, k)).astype(np.float32)
            if self.density < 1.0:
                mask = rng.random(size=(n, k)) < self.density
                w = np.where(mask, w, 0.0).astype(np.float32)
            self.idx = None
            self.win = torch.as_tensor(w, device=device, dtype=torch.float32)
            self.sparse = False

    def hash(self) -> str:
        parts = [self.win.detach().float().cpu().numpy().tobytes()]
        if self.idx is not None:
            parts.append(self.idx.detach().cpu().numpy().tobytes())
        return hashlib.sha256(b"".join(parts)).hexdigest()[:16]

    def project(self, e: np.ndarray) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(self.n, device=self.device, dtype=torch.float32)
        e_t = torch.as_tensor(
            np.asarray(e, dtype=np.float32).reshape(self.k),
            device=self.device,
        )
        local = (self.win @ e_t) * float(self.gain)
        if not self.sparse:
            return local
        drive = torch.zeros(self.n, device=self.device, dtype=torch.float32)
        drive.index_copy_(0, self.idx, local)
        return drive
