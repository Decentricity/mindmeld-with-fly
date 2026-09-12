"""Deterministic recording / replay of living-viz activity frames."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.frames: list[np.ndarray] = []
        self.meta: list[dict] = []
        self.active = False

    def start(self):
        self.frames.clear()
        self.meta.clear()
        self.active = True

    def stop_and_save(
        self,
        xy: np.ndarray,
        edges: np.ndarray,
        extra: dict | None = None,
        xyz: np.ndarray | None = None,
    ):
        self.active = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            return
        arr = np.stack(self.frames, axis=0).astype(np.float16)
        payload = {
            "activity": arr,
            "xy": xy.astype(np.float32),
            "edges": edges.astype(np.int32),
            "meta_json": np.array([str(self.meta)], dtype=object),
            "extra_json": np.array([str(extra or {})], dtype=object),
        }
        if xyz is not None:
            payload["xyz"] = np.asarray(xyz, dtype=np.float32)
        np.savez_compressed(self.path, **payload)

    def push(self, activity: np.ndarray, info: dict):
        if self.active:
            self.frames.append(np.asarray(activity, dtype=np.float32))
            self.meta.append(info)


def load_recording(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
