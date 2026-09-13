"""Mind-meld session: EEG -> normalize -> Win -> ONE fly reservoir."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..eeg.features import N_BANDS, coerce_bands
from ..eeg.source import EEGSample, EEGSource
from ..living.engine import LivingEngine
from ..reservoir import step
from .eeg_to_fly import EEGToFly
from .events import EventLog
from .fly_spectrum import FlySpectrum
from .normalize import RollingNormalizer


def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


@dataclass
class MeldMetrics:
    mean_abs: float
    active_frac: float
    input_norm: float
    e_norm: float
    step: int
    sps: float


class MeldSession:
    def __init__(
        self,
        source: EEGSource,
        *,
        graph: str = "oruk499",
        weighting: str = "unsigned",
        device: str = "cuda",
        seed: int = 0,
        win_seed: int = 0,
        rest: str = "tonic",
        tonic_amp: float = 0.05,
        eeg_gain: float = 1.0,
        reservoir_hz: float = 15.0,
        max_steps_per_tick: int = 1,
        log_every: int = 30,
    ):
        self.source = source
        self.engine = LivingEngine(graph=graph, weighting=weighting, device=device, seed=seed)
        self.norm = RollingNormalizer(dim=N_BANDS, window=250)
        self.mapper = EEGToFly(
            n=self.engine.n,
            k=N_BANDS,
            seed=win_seed,
            scale=0.25,
            device=str(self.engine.device),
        )
        self.mapper.gain = eeg_gain
        self.rest = rest if rest in ("silent", "tonic") else "tonic"
        self.tonic_amp = tonic_amp
        self.reservoir_hz = float(reservoir_hz)
        self.max_steps_per_tick = max(1, int(max_steps_per_tick))
        self.log_every = max(1, int(log_every))
        self.eeg_enabled = True
        self.logs = EventLog()
        self.last_e = np.zeros(N_BANDS, dtype=np.float32)
        self.last_e_raw = np.zeros(N_BANDS, dtype=np.float32)
        self.last_sample: EEGSample | None = None
        self.fly_spectrum = FlySpectrum(fs=self.reservoir_hz, window_sec=3.0)
        self.last_fly_abs = np.zeros(N_BANDS, dtype=np.float32)
        self.last_fly_rel = np.zeros(N_BANDS, dtype=np.float32)
        self._tick_i = 0
        self._last_step_wall = time.perf_counter()
        self._step_accum = 0.0
        try:
            vals = self.engine.w.values()
            n = int(vals.numel())
            stride = max(1, n // 4096)
            sample = vals[::stride][:4096].detach().float().cpu().numpy().tobytes()
            self.w_hash = hashlib.sha256(sample + str(n).encode()).hexdigest()[:16]
        except Exception:
            self.w_hash = "unknown"
        self.win_hash = self.mapper.hash()
        self.logs.emit(
            "SESSION_START",
            graph=graph,
            rest=self.rest,
            source=source.name,
            seed=seed,
            win_seed=win_seed,
            w_hash=self.w_hash,
            win_hash=self.win_hash,
            win_sparse=bool(self.mapper.sparse),
            win_targets=int(self.mapper.idx.numel()) if self.mapper.idx is not None else self.engine.n,
            e_dim=N_BANDS,
            bands=("delta", "theta", "alpha", "beta", "gamma"),
        )
        self.x_trace: list[np.ndarray] = []
        self.metric_trace: list[dict] = []

    def set_rest(self, mode: str):
        self.rest = mode if mode in ("silent", "tonic") else self.rest
        self.logs.emit("REST_MODE_CHANGE", rest=self.rest)

    def set_eeg_enabled(self, on: bool):
        self.eeg_enabled = bool(on)
        self.mapper.enabled = self.eeg_enabled
        self.logs.emit("EEG_ENABLE_CHANGE", enabled=self.eeg_enabled)

    def mark(self, label: str = "mark"):
        self.logs.emit("USER_MARKER", label=label)

    def reset_fly(self):
        self.engine.reset()
        self.logs.emit("RESET")

    def _tonic(self) -> torch.Tensor:
        """Background drive. On large graphs, only jitter Win targets (not all n)."""
        if self.rest != "tonic":
            return torch.zeros(self.engine.n, device=self.engine.device)
        if self.mapper.sparse and self.mapper.idx is not None:
            drive = torch.zeros(self.engine.n, device=self.engine.device, dtype=torch.float32)
            noise = torch.randn(
                self.mapper.idx.numel(),
                device=self.engine.device,
                dtype=torch.float32,
            ) * float(self.tonic_amp)
            drive.index_copy_(0, self.mapper.idx, noise)
            return drive
        return torch.randn(self.engine.n, device=self.engine.device, dtype=torch.float32) * float(
            self.tonic_amp
        )

    def poll_eeg(self) -> EEGSample | None:
        sample = self.source.read()
        if sample is None:
            return None
        self.last_sample = sample
        self.last_e_raw = coerce_bands(sample.e)
        self.last_e = self.norm.update(self.last_e_raw)
        if self._tick_i % self.log_every == 0:
            self.logs.emit(
                "EEG_FEATURE",
                source=self.source.name,
                e=self.last_e.tolist(),
                e_raw=self.last_e_raw.tolist(),
                calm=sample.calm,
                quality=sample.quality,
                marker=sample.marker,
            )
        return sample

    def step_reservoir(self, n_steps: int = 1) -> MeldMetrics:
        drive_eeg = self.mapper.project(self.last_e)
        tonic = self._tonic()
        t0 = time.perf_counter()
        n_steps = max(1, min(int(n_steps), self.max_steps_per_tick))
        for _ in range(n_steps):
            self.engine.x = step(self.engine.w, self.engine.x, drive_eeg + tonic)
            self.engine.step_i += 1
        if self.engine.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        sps = (n_steps / dt) if dt > 0 else 0.0
        self.engine._sps = sps
        act = self.engine.x.index_select(0, self.engine.viz_idx).detach().float()
        mean_abs = float(act.abs().mean().item())
        active_frac = float((act.abs() > 0.05).float().mean().item())

        self.last_fly_abs, self.last_fly_rel = self.fly_spectrum.push(mean_abs)

        m = MeldMetrics(
            mean_abs=mean_abs,
            active_frac=active_frac,
            input_norm=float(torch.linalg.vector_norm(drive_eeg).item()),
            e_norm=float(np.linalg.norm(self.last_e)),
            step=self.engine.step_i,
            sps=sps,
        )
        if self._tick_i % self.log_every == 0:
            self.logs.emit(
                "FLY_INPUT",
                input_norm=m.input_norm,
                e_norm=m.e_norm,
                mean_abs=m.mean_abs,
                active_frac=m.active_frac,
                step=m.step,
                fly_bands=self.last_fly_rel.tolist(),
            )
            self.metric_trace.append(
                {
                    "step": m.step,
                    "mean_abs": m.mean_abs,
                    "active_frac": m.active_frac,
                    "input_norm": m.input_norm,
                    "e_norm": m.e_norm,
                    "e": self.last_e.copy(),
                    "e_raw": self.last_e_raw.copy(),
                    "fly_rel": self.last_fly_rel.copy(),
                    "fly_abs": self.last_fly_abs.copy(),
                }
            )
        if self.engine.n <= 600:
            self.x_trace.append(act.detach().cpu().numpy().astype(np.float32))
        return m

    def tick(self) -> MeldMetrics:
        """One display-paced reservoir step (no catch-up backlog)."""
        self._tick_i += 1
        self.poll_eeg()
        now = time.perf_counter()
        dt = now - self._last_step_wall
        self._last_step_wall = now
        self._step_accum += dt * self.reservoir_hz
        n = int(self._step_accum)
        self._step_accum -= n
        n = 1 if n <= 0 else min(n, self.max_steps_per_tick)
        return self.step_reservoir(n)

    def frame_activity(self) -> np.ndarray:
        return self.engine.x.index_select(0, self.engine.viz_idx).detach().float().cpu().numpy()

    def save_run(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.logs.save(out / "events.jsonl")
        meta = {
            "graph": self.engine.graph,
            "rest": self.rest,
            "source": self.source.name,
            "w_hash": self.w_hash,
            "win_hash": self.win_hash,
            "win_seed": self.mapper.seed,
            "win_sparse": bool(self.mapper.sparse),
            "n": self.engine.n,
            "e_dim": N_BANDS,
            "bands": ["delta", "theta", "alpha", "beta", "gamma"],
        }
        (out / "config.json").write_text(json.dumps(meta, indent=2))
        if self.metric_trace:
            np.savez_compressed(
                out / "metrics.npz",
                step=np.array([m["step"] for m in self.metric_trace]),
                mean_abs=np.array([m["mean_abs"] for m in self.metric_trace], dtype=np.float32),
                active_frac=np.array([m["active_frac"] for m in self.metric_trace], dtype=np.float32),
                input_norm=np.array([m["input_norm"] for m in self.metric_trace], dtype=np.float32),
                e_norm=np.array([m["e_norm"] for m in self.metric_trace], dtype=np.float32),
                e=np.stack([m["e"] for m in self.metric_trace], axis=0),
                e_raw=np.stack([m["e_raw"] for m in self.metric_trace], axis=0),
                fly_rel=np.stack([m["fly_rel"] for m in self.metric_trace], axis=0),
                fly_abs=np.stack([m["fly_abs"] for m in self.metric_trace], axis=0),
                feature_names=np.array(["delta", "theta", "alpha", "beta", "gamma"]),
            )
        if self.x_trace:
            np.savez_compressed(out / "x_trace.npz", x=np.stack(self.x_trace, axis=0))
        return out
