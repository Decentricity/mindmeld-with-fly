"""HUMAN EEG superpane drawn into the living libcaca canvas (illustrative silhouette)."""
from __future__ import annotations

import numpy as np

from .render import CACA_CHARS, _caca_ansi_pair


def _brain_silhouette(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h * 0.48, w * 0.5
    ry, rx = h * 0.38, w * 0.40
    mask = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0
    stem = ((yy - (cy + ry * 0.85)) / max(h * 0.12, 1)) ** 2 + ((xx - cx) / max(w * 0.08, 1)) ** 2 <= 1.0
    return (mask | stem).astype(np.float32)


def paint_human_panel(lib, cv, session, x0: int, y0: int, width: int, height: int) -> None:
    """Paint HUMAN brain strip + EEG meters into an existing libcaca canvas region."""
    if height < 4 or width < 8:
        return
    sil_h = max(3, height - 1)
    sil = _brain_silhouette(sil_h, width)
    e = session.last_e
    sample = session.last_sample
    base = 0.12 * sil
    if e is not None and len(e) >= 3:
        base = sil * (0.15 + 0.55 * float(np.clip(e[2], 0, 1)))

    for y in range(sil_h):
        for x in range(width):
            v = float(base[y, x])
            ch = ord(CACA_CHARS[min(len(CACA_CHARS) - 1, int(v * (len(CACA_CHARS) - 1) + 1e-6))])
            pair = _caca_ansi_pair(v)
            fg, bg = pair & 0x0F, (pair >> 4) & 0x0F
            if v <= 0.015:
                fg, bg = 0x00, 0x00
            lib.caca_set_color_ansi(cv, fg, bg)
            lib.caca_put_char(cv, x0 + x, y0 + y, ch)

    lib.caca_set_color_ansi(cv, 0x0F, 0x00)
    lib.caca_put_str(cv, x0 + 1, y0, b"HUMAN")
    names = ("D", "T", "A", "B")
    for i, name in enumerate(names):
        val = float(e[i]) if e is not None and i < len(e) else 0.0
        bar_w = int(np.clip(val, 0, 1) * max(4, width // 5))
        yy = y0 + 1 + i
        if yy >= y0 + sil_h:
            break
        for x in range(bar_w):
            lib.caca_set_color_ansi(cv, 0x03, 0x00)
            lib.caca_put_char(cv, x0 + 2 + x, yy, ord("#"))
        lib.caca_set_color_ansi(cv, 0x0F, 0x00)
        lib.caca_put_str(cv, x0 + 3 + bar_w, yy, name.encode())

    if sample is not None:
        tag = f"Q{int(round(sample.quality * 100))} C{int(round(sample.calm * 100))}"
        lib.caca_set_color_ansi(cv, 0x0F, 0x00)
        lib.caca_put_str(cv, x0 + max(1, width - len(tag) - 1), y0, tag.encode())

    # conduit row under silhouette — pulse from EEG magnitude
    link_y = y0 + sil_h
    if link_y >= y0 + height:
        return
    pulse = float(
        np.clip(float(np.linalg.norm(e)) * 0.2 + (float(e.std()) * 4.0 if e is not None else 0.0), 0, 1)
    )
    lib.caca_set_color_ansi(cv, 0x02, 0x00)
    lib.caca_put_str(cv, x0, link_y, (("-" * width)[:width]).encode())
    cx = width // 2
    half = int(2 + pulse * (width // 4))
    for x in range(max(0, cx - half), min(width, cx + half)):
        lib.caca_set_color_ansi(cv, 0x05, 0x00)
        lib.caca_put_char(cv, x0 + x, link_y, ord("=" if pulse > 0.35 else "-"))
