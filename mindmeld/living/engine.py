"""GPU reservoir engine for living visualization (decoupled from rendering)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..oruk import load_oruk499
from ..preprocess import load_processed
from ..reservoir import step, to_torch
from .coords import load_viz_neurons, sample_display_edges
from .stim import Stimulator


@dataclass
class FrameState:
    step: int
    activity: np.ndarray  # |viz|
    mean: float
    active: int
    steps_per_sec: float
    vram_bytes: int | None
    mode: str
    view: str
    intensity: float
    paused: bool
    camera: str = "triad"
    yaw: float = 0.0
    pitch: float = 0.0


class LivingEngine:
    VIEWS = ("whole", "region", "heatmap", "storm")
    CAMERAS = ("triad", "orbit")
    YAW_STEP = 0.12
    PITCH_STEP = 0.08
    PITCH_MAX = 1.2

    def __init__(self, graph: str = "full", weighting: str = "unsigned", device: str = "cuda", seed: int = 0):
        self.graph = graph
        self.weighting = weighting
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        if graph == "oruk499":
            scipy_w, nodes, meta = load_oruk499()
            self.weighting = "oruk_signed"
        else:
            scipy_w, nodes, meta = load_processed(weighting)
        self.meta = meta
        self.n = scipy_w.shape[0]
        self.w = to_torch(scipy_w, str(self.device))
        self.x = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.stim = Stimulator(self.n, str(self.device), seed=seed)
        self.viz = load_viz_neurons(nodes, max_points=6000 if graph == "full" else min(506, 6000), seed=seed)
        self.edges = sample_display_edges(scipy_w, self.viz["index"], max_edges=3500, seed=seed + 1)
        self.viz_idx = torch.as_tensor(self.viz["index"], device=self.device, dtype=torch.long)
        self.view = "whole"
        self.camera = "triad"  # triad = XY|XZ|YZ+orb peek; orbit = single rotatable view
        self.yaw = 0.35
        self.pitch = 0.25
        self.paused = False
        self.speed = 1  # sim steps per frame
        self.step_i = 0
        self._sps = 0.0
        self.prev_activity = np.zeros(self.viz["n"], dtype=np.float32)
        del scipy_w  # free host CSR copy after edge sample + torch upload

    def cycle_view(self, delta: int = 1):
        i = self.VIEWS.index(self.view)
        self.view = self.VIEWS[(i + delta) % len(self.VIEWS)]

    def toggle_camera(self):
        i = self.CAMERAS.index(self.camera)
        self.camera = self.CAMERAS[(i + 1) % len(self.CAMERAS)]

    def nudge_yaw(self, delta: float):
        self.yaw = (self.yaw + delta) % (2 * math.pi)

    def nudge_pitch(self, delta: float):
        self.pitch = float(np.clip(self.pitch + delta, -self.PITCH_MAX, self.PITCH_MAX))

    def reset(self):
        self.x.zero_()
        self.step_i = 0
        self.prev_activity[:] = 0
        self.stim.t = 0

    def tick(self) -> FrameState:
        if not self.paused:
            t0 = time.perf_counter()
            for _ in range(max(1, self.speed)):
                drive = self.stim.drive()
                self.x = step(self.w, self.x, drive)
                self.step_i += 1
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self._sps = (max(1, self.speed) / dt) if dt > 0 else 0.0
        act = self.x.index_select(0, self.viz_idx).detach().float().cpu().numpy()
        active = int((np.abs(act) > 0.05).sum())
        vram = int(torch.cuda.max_memory_allocated()) if self.device.type == "cuda" else None
        return FrameState(
            step=self.step_i,
            activity=act,
            mean=float(act.mean()) if len(act) else 0.0,
            active=active,
            steps_per_sec=self._sps,
            vram_bytes=vram,
            mode=self.stim.mode,
            view=self.view,
            intensity=float(self.stim.intensity),
            paused=self.paused,
            camera=self.camera,
            yaw=float(self.yaw),
            pitch=float(self.pitch),
        )
