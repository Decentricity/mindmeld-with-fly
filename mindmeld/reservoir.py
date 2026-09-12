"""Sparse float32 leaky-tanh recurrence; W[POST, PRE], never dense adjacency."""
import numpy as np
import scipy.sparse as sp
import torch

def make_csr(pre, post, weights, n):
    w = sp.coo_matrix((np.asarray(weights, dtype=np.float32), (post, pre)), shape=(n, n)).tocsr()
    w.sum_duplicates()
    w.sort_indices()
    return w

def to_torch(w, device):
    return torch.sparse_csr_tensor(torch.from_numpy(w.indptr.astype(np.int32)),
        torch.from_numpy(w.indices.astype(np.int32)), torch.from_numpy(w.data.astype(np.float32)),
        size=w.shape, dtype=torch.float32, device=device)

def step(w, x, drive, alpha=0.15):
    return (1-alpha)*x + alpha*torch.tanh(torch.mv(w, x) + drive)

def signals(steps, channels=4):
    t = np.arange(steps, dtype=np.float32)[:, None]
    f = np.arange(1, channels+1, dtype=np.float32)[None, :]
    return (np.sin(t*f*0.013) + 0.3*np.cos(t*f*0.037)).astype(np.float32)

class Projection:
    """One seeded input channel and coefficient per node: O(N), not N×N."""
    def __init__(self, n, device, seed=42, channels=4, scale=0.3):
        rng = np.random.default_rng(seed)
        self.channels = torch.as_tensor(rng.integers(channels, size=n), device=device)
        self.gains = torch.as_tensor(rng.uniform(-scale, scale, size=n).astype(np.float32), device=device)

    def __call__(self, u):
        return self.gains * u[self.channels]
