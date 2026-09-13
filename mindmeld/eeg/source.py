"""EEGSource abstraction: live_muse / replay / synthetic."""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .features import (
    FEATURE_NAMES,
    N_BANDS,
    BandFeatures,
    calm_from_rel,
    coerce_bands,
    compute_band_powers_from_eeg,
    quality_from_rms,
)


@dataclass
class EEGSample:
    t: float
    e: np.ndarray  # (5,) δ θ α β γ — relative by default for drive stability
    abs_power: np.ndarray
    calm: float
    quality: float
    channel_rms: np.ndarray
    marker: str = ""


class EEGSource(ABC):
    name: str = "base"

    @abstractmethod
    def read(self) -> EEGSample | None:
        """Non-blocking; return latest sample or None if not ready."""

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SyntheticEEGSource(EEGSource):
    """Known oscillating band shares for validation (no hardware)."""

    name = "synthetic"

    def __init__(self, hz: float = 25.0, seed: int = 0):
        self.hz = hz
        self.rng = np.random.default_rng(seed)
        self._t0 = time.perf_counter()
        self._last = self._t0
        self.period = 1.0 / max(hz, 1e-6)

    def read(self) -> EEGSample | None:
        now = time.perf_counter()
        if now - self._last < self.period:
            return None
        self._last = now
        t = now - self._t0
        # smooth cyclic mixture so each band dominates in turn
        phase = 2 * np.pi * (t / 8.0)
        raw = np.array(
            [
                0.12 + 0.30 * (0.5 + 0.5 * np.sin(phase)),
                0.12 + 0.30 * (0.5 + 0.5 * np.sin(phase + 1.0)),
                0.12 + 0.30 * (0.5 + 0.5 * np.sin(phase + 2.0)),
                0.12 + 0.30 * (0.5 + 0.5 * np.sin(phase + 3.0)),
                0.10 + 0.28 * (0.5 + 0.5 * np.sin(phase + 4.0)),
            ],
            dtype=np.float32,
        )
        rel = raw / (raw.sum() + 1e-12)
        abs_p = rel * (50.0 + 20.0 * np.sin(phase * 0.5))
        # Vary per-channel RMS so electrode regions light up differently (Muse order).
        channel_rms = np.array(
            [
                25.0 + 35.0 * (0.5 + 0.5 * np.sin(phase + 0.3)),  # TP9
                30.0 + 40.0 * (0.5 + 0.5 * np.sin(phase + 1.1)),  # AF7
                28.0 + 38.0 * (0.5 + 0.5 * np.sin(phase + 2.0)),  # AF8
                22.0 + 32.0 * (0.5 + 0.5 * np.sin(phase + 2.9)),  # TP10
            ],
            dtype=np.float32,
        )
        return EEGSample(
            t=t,
            e=rel.astype(np.float32),
            abs_power=abs_p.astype(np.float32),
            calm=calm_from_rel(rel),
            quality=1.0,
            channel_rms=channel_rms,
        )


class ReplayEEGSource(EEGSource):
    name = "replay"

    def __init__(self, path: str | Path, realtime: bool = True):
        self.path = Path(path)
        self.realtime = realtime
        data = np.load(self.path, allow_pickle=False)
        self.t = np.asarray(data["t"], dtype=np.float64)
        self.e = np.asarray(data["e"], dtype=np.float32)
        if self.e.ndim == 1:
            self.e = self.e.reshape(-1, 1)
        if self.e.shape[1] != N_BANDS:
            padded = np.zeros((len(self.e), N_BANDS), dtype=np.float32)
            take = min(self.e.shape[1], N_BANDS)
            padded[:, :take] = self.e[:, :take]
            self.e = padded
        self.abs_power = np.asarray(data.get("abs_power", self.e), dtype=np.float32)
        if self.abs_power.ndim == 1:
            self.abs_power = self.abs_power.reshape(-1, N_BANDS)
        if self.abs_power.shape[1] != N_BANDS:
            padded = np.zeros((len(self.abs_power), N_BANDS), dtype=np.float32)
            take = min(self.abs_power.shape[1], N_BANDS)
            padded[:, :take] = self.abs_power[:, :take]
            self.abs_power = padded
        self.calm = np.asarray(data.get("calm", np.zeros(len(self.t))), dtype=np.float32)
        self.quality = np.asarray(data.get("quality", np.ones(len(self.t))), dtype=np.float32)
        self.channel_rms = np.asarray(
            data.get("channel_rms", np.zeros((len(self.t), 4), dtype=np.float32)),
            dtype=np.float32,
        )
        markers = data["markers"] if "markers" in data.files else None
        if markers is not None:
            # allow_pickle only for string marker column we wrote ourselves
            raw = np.load(self.path, allow_pickle=True)
            markers = raw["markers"]
            self.markers = [str(x) for x in markers.tolist()]
        else:
            self.markers = [""] * len(self.t)
        self.i = 0
        self._wall0 = time.perf_counter()
        self._t0 = float(self.t[0]) if len(self.t) else 0.0

    def read(self) -> EEGSample | None:
        if self.i >= len(self.t):
            return None
        if self.realtime:
            elapsed = time.perf_counter() - self._wall0
            target = float(self.t[self.i] - self._t0)
            if elapsed < target:
                return None
        i = self.i
        self.i += 1
        return EEGSample(
            t=float(self.t[i]),
            e=coerce_bands(self.e[i]),
            abs_power=coerce_bands(self.abs_power[i]),
            calm=float(self.calm[i]),
            quality=float(self.quality[i]),
            channel_rms=self.channel_rms[i],
            marker=self.markers[i],
        )


class LiveMuseEEGSource(EEGSource):
    """Headless Muse 2 via BrainFlow — same board path as neurofeedback-muse."""

    name = "live_muse"

    def __init__(self, mac: str | None = None, window_sec: float = 5.0):
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams, BrainFlowPresets

        self._BoardShim = BoardShim
        self._Presets = BrainFlowPresets
        params = BrainFlowInputParams()
        if mac:
            params.mac_address = mac
        params.timeout = 20  # seconds for BLE discovery/connect
        BoardShim.disable_board_logger()
        self.board = BoardShim(BoardIds.MUSE_2_BOARD, params)
        self.board.prepare_session()
        self.board.start_stream()
        self.board_id = BoardIds.MUSE_2_BOARD.value
        self.fs = float(BoardShim.get_sampling_rate(self.board_id))
        self.eeg_rows = BoardShim.get_eeg_channels(self.board_id)
        self.n_points = int(self.fs * window_sec)
        self._ema_rel = np.zeros(N_BANDS, dtype=np.float32)
        self._ema_abs = np.zeros(N_BANDS, dtype=np.float32)
        self._calm = 0.4
        self._t0 = time.perf_counter()
        self._period = 1.0 / 25.0
        self._last = 0.0

    def read(self) -> EEGSample | None:
        from .features import EMA_K

        now = time.perf_counter()
        if now - self._last < self._period:
            return None
        self._last = now
        try:
            if self.board.get_board_data_count(self._Presets.DEFAULT_PRESET) < 1:
                return None
            data = self.board.get_current_board_data(self.n_points, self._Presets.DEFAULT_PRESET)
        except Exception:
            return None
        if data.shape[1] < 64:
            return None
        eeg = data[self.eeg_rows]
        abs_bp, rel_bp, rms = compute_band_powers_from_eeg(eeg, self.fs)
        if float(abs_bp.sum()) <= 0:
            return None
        self._ema_rel = self._ema_rel + EMA_K * (rel_bp - self._ema_rel)
        self._ema_abs = self._ema_abs + EMA_K * (abs_bp - self._ema_abs)
        c = calm_from_rel(self._ema_rel)
        self._calm = self._calm + EMA_K * (c - self._calm)
        return EEGSample(
            t=now - self._t0,
            e=self._ema_rel.copy(),
            abs_power=self._ema_abs.copy(),
            calm=float(self._calm),
            quality=quality_from_rms(rms),
            channel_rms=rms,
        )

    def close(self) -> None:
        try:
            self.board.stop_stream()
        except Exception:
            pass
        try:
            self.board.release_session()
        except Exception:
            pass


def open_source(kind: str, **kwargs) -> EEGSource:
    kind = kind.lower().replace("-", "_")
    if kind in ("synthetic", "synth"):
        return SyntheticEEGSource(**{k: v for k, v in kwargs.items() if k in ("hz", "seed")})
    if kind == "replay":
        path = kwargs.get("recording") or kwargs.get("path")
        if not path:
            raise ValueError("replay requires recording= path")
        return ReplayEEGSource(path, realtime=bool(kwargs.get("realtime", True)))
    if kind in ("live_muse", "live", "muse"):
        return LiveMuseEEGSource(mac=kwargs.get("mac"), window_sec=float(kwargs.get("window_sec", 5.0)))
    raise ValueError(f"unknown EEG source: {kind}")


def save_recording(path: str | Path, samples: list[EEGSample], meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=np.array([s.t for s in samples], dtype=np.float64),
        e=np.stack([s.e for s in samples], axis=0).astype(np.float32),
        abs_power=np.stack([s.abs_power for s in samples], axis=0).astype(np.float32),
        calm=np.array([s.calm for s in samples], dtype=np.float32),
        quality=np.array([s.quality for s in samples], dtype=np.float32),
        channel_rms=np.stack([s.channel_rms for s in samples], axis=0).astype(np.float32),
        markers=np.array([s.marker for s in samples], dtype="U64"),
        feature_names=np.array(FEATURE_NAMES),
        meta_json=json.dumps(meta or {}),
    )
    return path


def iter_recording(path: str | Path) -> Iterator[EEGSample]:
    src = ReplayEEGSource(path, realtime=False)
    while True:
        s = src.read()
        if s is None:
            break
        yield s
