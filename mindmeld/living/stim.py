"""Configurable stimulation patterns for the living connectome."""
from __future__ import annotations

import numpy as np
import torch


class Stimulator:
    MODES = ("noise", "impulse", "oscillation", "sensory", "wave", "off")

    DESCRIPTIONS = {
        "noise": (
            "Gaussian white noise on 4 virtual sensory channels, "
            "projected sparsely onto every neuron (random ±gain)"
        ),
        "impulse": (
            "Brief pulsed spike rotating across the 4 channels "
            "(one channel lit per step; press i for an extra burst)"
        ),
        "oscillation": (
            "Steady sinusoidal drive on 4 channels with fixed phase offsets "
            "(continuous rhythmic input)"
        ),
        "sensory": (
            "Mixed sine+cosine 'sensory-like' drive on 4 channels "
            "(default stand-in for patterned sensory afferents)"
        ),
        "wave": (
            "Slow traveling/modulated wave across the 4 channels "
            "(amplitude envelope × phase-shifted sines)"
        ),
        "off": (
            "No continuous drive — network evolves from residual state only "
            "(i still fires a short impulse burst)"
        ),
    }

    def __init__(self, n: int, device: str, seed: int = 0, channels: int = 4):
        self.n = n
        self.device = device
        self.mode = "sensory"
        self.intensity = 0.35
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self.channels = channels
        # fixed sparse projection: each neuron one channel + gain
        self.chan = torch.as_tensor(self.rng.integers(channels, size=n), device=device)
        self.gain = torch.as_tensor(
            self.rng.uniform(-1.0, 1.0, size=n).astype(np.float32) * 0.3, device=device
        )
        self._impulse_left = 0

    def cycle_mode(self, delta: int = 1):
        i = self.MODES.index(self.mode)
        self.mode = self.MODES[(i + delta) % len(self.MODES)]

    def trigger_impulse(self, duration: int = 8):
        self._impulse_left = duration
        if self.mode == "off":
            self.mode = "impulse"

    def describe(self) -> str:
        """One-line human explanation of the active stimulation."""
        body = self.DESCRIPTIONS.get(self.mode, self.mode)
        burst = ""
        if self._impulse_left > 0 and self.mode != "impulse":
            burst = f" + impulse burst ({self._impulse_left} steps left)"
        return f"STIM [{self.mode}] @{self.intensity:.2f}: {body}{burst}"

    def drive(self) -> torch.Tensor:
        self.t += 1
        t = self.t
        intens = float(self.intensity)
        if self.mode == "off" and self._impulse_left <= 0:
            return torch.zeros(self.n, device=self.device)
        if self.mode == "noise":
            u = torch.randn(self.channels, device=self.device) * intens
        elif self.mode == "impulse" or self._impulse_left > 0:
            self._impulse_left = max(0, self._impulse_left - 1)
            u = torch.zeros(self.channels, device=self.device)
            u[t % self.channels] = 3.0 * intens
        elif self.mode == "oscillation":
            phase = torch.arange(self.channels, device=self.device, dtype=torch.float32)
            u = intens * torch.sin(0.07 * t + phase)
        elif self.mode == "wave":
            phase = torch.arange(self.channels, device=self.device, dtype=torch.float32)
            u = intens * torch.sin(0.03 * t + 0.6 * phase) * float(np.cos(0.011 * t))
        else:  # sensory
            phase = torch.arange(1, self.channels + 1, device=self.device, dtype=torch.float32)
            u = intens * (
                torch.sin(t * phase * 0.013) + 0.3 * torch.cos(t * phase * 0.037)
            )
        return self.gain * u[self.chan]
