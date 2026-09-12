"""Sparse spectral-radius estimate via power iteration (no densification)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spa


def estimate_spectral_radius(w: sp.spmatrix, iters: int = 40, seed: int = 0) -> float:
    """Return an estimate of the largest-magnitude eigenvalue of sparse W."""
    n = w.shape[0]
    if n == 0:
        return 0.0
    if n < 2000:
        try:
            vals = spa.eigs(w.astype(np.float64), k=1, which="LM", return_eigenvectors=False)
            return float(np.abs(vals[0]))
        except Exception:
            pass
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float64)
    x /= np.linalg.norm(x) + 1e-12
    lam = 0.0
    for _ in range(iters):
        y = w @ x
        lam = float(np.linalg.norm(y))
        if lam < 1e-15:
            return 0.0
        x = y / lam
    return lam


def scale_to_radius(w: sp.csr_matrix, target: float = 0.9, **kwargs) -> tuple[sp.csr_matrix, float, float]:
    before = estimate_spectral_radius(w, **kwargs)
    if before <= 0:
        return w, before, before
    scaled = w.copy()
    scaled.data = (scaled.data * (target / before)).astype(np.float32)
    after = estimate_spectral_radius(scaled, **kwargs)
    return scaled, before, after
