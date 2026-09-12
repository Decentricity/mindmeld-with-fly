"""Non-GPU spoken status for living-viz toggles via say-alert (RHVoice / speech-dispatcher).

Never use Piper here — the living sim already owns the GPU.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

SAY_ALERT = Path("/home/decentricity/bin/say-alert")

# Short spoken lines (RHVoice); keep under ~2 sentences.
STIM_SAY = {
    "noise": "Stimulation: noise. Gaussian white noise on four channels.",
    "impulse": "Stimulation: impulse. Rotating pulse spikes on four channels.",
    "oscillation": "Stimulation: oscillation. Steady rhythmic sine drive.",
    "sensory": "Stimulation: sensory. Mixed sine-cosine sensory-like drive.",
    "wave": "Stimulation: wave. Slow modulated traveling wave.",
    "off": "Stimulation: off. No continuous drive.",
}

VIEW_SAY = {
    "whole": "View: whole connectome with edges.",
    "region": "View: region coloring by superclass.",
    "heatmap": "View: activity heatmap.",
    "storm": "View: storm — rising activity only.",
}

CAMERA_SAY = {
    "triad": "Camera: triad. X Y, X Z, and Y Z panels.",
    "orbit": "Camera: orbit. Use comma and period to yaw, J and K to pitch.",
}


def announce(text: str, *, enabled: bool = True) -> None:
    """Fire-and-forget say-alert so the viz loop never blocks on speech."""
    if not enabled:
        return
    text = " ".join((text or "").split())
    if not text:
        return
    try:
        subprocess.Popen(
            [str(SAY_ALERT), text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        pass


def announce_stim(mode: str, intensity: float, *, enabled: bool = True) -> None:
    base = STIM_SAY.get(mode, f"Stimulation: {mode}.")
    announce(f"{base} Intensity {intensity:.2f}.", enabled=enabled)


def announce_view(view: str, *, enabled: bool = True) -> None:
    announce(VIEW_SAY.get(view, f"View: {view}."), enabled=enabled)


def announce_camera(camera: str, *, enabled: bool = True) -> None:
    announce(CAMERA_SAY.get(camera, f"Camera: {camera}."), enabled=enabled)
