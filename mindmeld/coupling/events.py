"""Lightweight sequenced event log for mind-meld sessions."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Event:
    seq: int
    t_mono: float
    kind: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)


class EventLog:
    def __init__(self):
        self.seq = 0
        self.events: list[Event] = []
        self._t0 = time.perf_counter()

    def emit(self, kind: str, source: str = "session", **payload) -> Event:
        ev = Event(
            seq=self.seq,
            t_mono=time.perf_counter() - self._t0,
            kind=kind,
            source=source,
            payload=payload,
        )
        self.seq += 1
        self.events.append(ev)
        return ev

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(e), default=float) for e in self.events]
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
        return path
