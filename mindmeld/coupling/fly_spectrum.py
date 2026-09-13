"""Spectral band readout from fly reservoir activity (analogy to human EEG bands).

Not insect electrophysiology — Welch/band-power on a rolling buffer of mean
|viz| activity, using the same Hz edges as Muse (δ θ α β γ).
"""
from __future__ import annotations

import numpy as np

from ..eeg.features import BANDS, EMA_K, N_BANDS, NFFT


class FlySpectrum:
    """Ring buffer → relative/abs band shares for the digital fly."""

    def __init__(self, fs: float = 15.0, window_sec: float = 3.0):
        self.fs = max(4.0, float(fs))
        self.n_buf = max(NFFT // 4, int(self.fs * window_sec))
        self.buf = np.zeros(self.n_buf, dtype=np.float64)
        self.i = 0
        self.filled = 0
        self.last_abs = np.zeros(N_BANDS, dtype=np.float32)
        self.last_rel = np.zeros(N_BANDS, dtype=np.float32)
        self._ema_rel = np.zeros(N_BANDS, dtype=np.float32)
        self._ema_abs = np.zeros(N_BANDS, dtype=np.float32)

    def push(self, mean_abs_activity: float) -> tuple[np.ndarray, np.ndarray]:
        self.buf[self.i % self.n_buf] = float(mean_abs_activity)
        self.i += 1
        self.filled = min(self.filled + 1, self.n_buf)
        if self.filled < max(32, NFFT // 8):
            return self.last_abs, self.last_rel

        # Chronological window
        if self.filled < self.n_buf:
            sig = self.buf[: self.filled].copy()
        else:
            start = self.i % self.n_buf
            sig = np.concatenate([self.buf[start:], self.buf[:start]])

        abs_bp, rel_bp = _band_powers_1d(sig, self.fs)
        self._ema_abs = self._ema_abs + EMA_K * (abs_bp - self._ema_abs)
        self._ema_rel = self._ema_rel + EMA_K * (rel_bp - self._ema_rel)
        # renormalize ema rel
        s = float(self._ema_rel.sum()) + 1e-12
        self.last_abs = self._ema_abs.copy()
        self.last_rel = (self._ema_rel / s).astype(np.float32)
        return self.last_abs, self.last_rel


def _band_powers_1d(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch-ish band powers without BrainFlow (reservoir is not μV EEG)."""
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - sig.mean()
    n = len(sig)
    if n < 16:
        z = np.zeros(N_BANDS, dtype=np.float32)
        return z, z
    # Hann periodogram
    nfft = int(2 ** np.ceil(np.log2(max(n, 32))))
    nfft = min(max(nfft, 64), 512)
    window = np.hanning(n)
    spectrum = np.fft.rfft(sig * window, n=nfft)
    psd = (np.abs(spectrum) ** 2) / (np.sum(window**2) * fs + 1e-12)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    abs_bp = np.zeros(N_BANDS, dtype=np.float64)
    for i, (_, lo, hi) in enumerate(BANDS):
        # Map Muse Hz bands into analysis band relative to Nyquist of our fs.
        # If fs is low (e.g. 15 Hz), high bands collapse — still useful as
        # relative spectral shape of the reservoir trajectory.
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            # fallback: partition spectrum into N_BANDS equal log-ish bins
            continue
        integ = getattr(np, "trapezoid", None) or np.trapz
        abs_bp[i] = float(integ(psd[mask], freqs[mask]))
    if float(abs_bp.sum()) <= 0:
        # Low-fs fallback: split rfft into N_BANDS contiguous chunks
        m = max(1, len(psd) // N_BANDS)
        for i in range(N_BANDS):
            sl = psd[i * m : (i + 1) * m] if i < N_BANDS - 1 else psd[i * m :]
            abs_bp[i] = float(np.sum(sl)) + 1e-12
    rel = abs_bp / (abs_bp.sum() + 1e-12)
    return abs_bp.astype(np.float32), rel.astype(np.float32)
