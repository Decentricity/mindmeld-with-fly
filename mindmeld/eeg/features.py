"""EEG band feature helpers reused from neurofeedback-muse (BrainFlow Welch path)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Keep band edges identical to muse/neurofeedback-muse/muse_viewer.py
BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
)
# gamma exists in the viewer but Phase II e[t] is δ θ α β only
GAMMA = ("gamma", 30.0, 45.0)

NFFT = 256
EMA_K = 0.06
FEATURE_NAMES = ("delta", "theta", "alpha", "beta")


@dataclass
class BandFeatures:
    """Normalized-ready absolute + relative band powers and calm."""

    abs_power: np.ndarray  # (4,) δθαλβ
    rel_power: np.ndarray  # (4,)
    calm: float
    quality: float  # 0..1 rough contact heuristic from channel RMS
    channel_rms: np.ndarray  # (4,) TP9 AF7 AF8 TP10 order when available
    timestamp: float


def compute_band_powers_from_eeg(
    eeg: np.ndarray,
    fs: float,
    *,
    do_filter: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (abs_bp[4], rel_bp[4], chan_rms[n_ch]) using BrainFlow DataFilter.

    `eeg` shape: (n_channels, n_samples). Uses the same detrend/bandpass/notch
    + Welch path as muse_viewer.py (without Qt).
    """
    from brainflow.data_filter import (
        DataFilter,
        DetrendOperations,
        FilterTypes,
        NoiseTypes,
        WindowOperations,
    )

    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2:
        raise ValueError("eeg must be (channels, samples)")
    n_ch, n_samp = eeg.shape
    chan_rms = np.zeros(n_ch, dtype=np.float64)
    if n_samp < max(64, NFFT):
        return np.zeros(4), np.zeros(4), chan_rms

    psd_amp = None
    freqs = None
    for i in range(n_ch):
        sig = eeg[i].copy()
        if do_filter and sig.shape[0] > 32:
            DataFilter.detrend(sig, DetrendOperations.LINEAR.value)
            DataFilter.perform_bandpass(sig, int(fs), 1.0, 45.0, 4, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.remove_environmental_noise(sig, int(fs), NoiseTypes.FIFTY_AND_SIXTY.value)
        chan_rms[i] = float(np.std(sig))
        amp, freqs = DataFilter.get_psd_welch(
            sig, NFFT, NFFT // 2, int(fs), WindowOperations.HANNING.value
        )
        psd_amp = amp if psd_amp is None else psd_amp + amp
    assert psd_amp is not None and freqs is not None
    psd_amp = psd_amp / n_ch
    psd = (psd_amp, freqs)
    abs_bp = np.array(
        [DataFilter.get_band_power(psd, lo, hi) for _, lo, hi in BANDS], dtype=np.float64
    )
    rel_bp = abs_bp / (abs_bp.sum() + 1e-12)
    return abs_bp.astype(np.float32), rel_bp.astype(np.float32), chan_rms.astype(np.float32)


def calm_from_rel(rel_bp: np.ndarray) -> float:
    alpha, beta = float(rel_bp[2]), float(rel_bp[3])
    return float(alpha / (alpha + beta + 1e-9))


def quality_from_rms(chan_rms: np.ndarray) -> float:
    """Rough 0..1 contact score: mid RMS = good (mirrors viewer heuristics loosely)."""
    if chan_rms.size == 0:
        return 0.0
    # typical filtered Muse EEG std when seated: tens of uV; dead ~0; motion huge
    scores = []
    for s in chan_rms:
        s = float(s)
        if s < 1.0:
            scores.append(0.0)
        elif s > 200.0:
            scores.append(0.2)
        else:
            scores.append(float(np.clip(1.0 - abs(s - 40.0) / 80.0, 0.1, 1.0)))
    return float(np.mean(scores))
