"""ASCII housefly pane — FlyBrains art with MaleCNS XY view on the head.

Art + fit-to-box scaling from dealer1943/FlyBrains. Connectome overlay uses the
same projection as the triad top-left pane (XY), shrunk into the head/eye band.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .engine import FrameState, LivingEngine
from .project import activity_amp, project_plane, viz_xyz
from .render import CACA_CHARS, _caca_ansi_pair
from .views import _raster_into

ART_PATH = Path(__file__).resolve().parent / "art" / "fly_side.txt"
SHADES = set("░▒▓█")
FLY_MARGIN = 0
_ART_CACHE: list[str] | None = None

# FlyBrains head/thorax crop (atlas.fly_head_cells) — where the brain sits.
_HEAD_Y0, _HEAD_Y1 = 0.08, 0.40
_HEAD_X0, _HEAD_X1 = 0.30, 0.74
# Shrink CNS map inside that head box so it sits on head/eyes, not belly.
_CNS_FILL = 0.82


def _load_fly_art() -> list[str]:
    global _ART_CACHE
    if _ART_CACHE is not None:
        return _ART_CACHE
    if not ART_PATH.is_file():
        _ART_CACHE = ["  (fly)  "]
        return _ART_CACHE
    rows = ART_PATH.read_text().splitlines()
    width = max((len(row) for row in rows), default=1)
    _ART_CACHE = [row.ljust(width) for row in rows]
    return _ART_CACHE


def _fit_geometry(width: int, height: int) -> tuple[list[str], int, int, int, int, int, int]:
    """Return (art, art_w, art_h, dw, dh, x0, y0) — FlyBrains fly_lines fit."""
    width = max(8, int(width))
    height = max(4, int(height))
    art = _load_fly_art()
    art_h = max(1, len(art))
    art_w = max(1, len(art[0]) if art else 1)
    margin = max(0, int(FLY_MARGIN))
    inner_w = max(1, width - 2 * margin)
    inner_h = max(1, height - 2 * margin)
    scale = min(inner_w / art_w, inner_h / art_h)
    dw = max(1, int(round(art_w * scale)))
    dh = max(1, int(round(art_h * scale)))
    if dw < inner_w and dh * inner_w <= inner_h * art_w:
        dw = inner_w
        dh = max(1, int(round(dw * art_h / art_w)))
        dh = min(dh, inner_h)
    elif dh < inner_h:
        dh = inner_h
        dw = max(1, int(round(dh * art_w / art_h)))
        dw = min(dw, inner_w)
    x0 = margin + (inner_w - dw) // 2
    y0 = margin + (inner_h - dh) // 2
    return art, art_w, art_h, dw, dh, x0, y0


def _sample_art_cell(art: list[str], art_w: int, art_h: int, dw: int, dh: int, sx: int, sy: int) -> str:
    rank = {" ": 0, "░": 1, "▒": 2, "▓": 3, "█": 4}
    xa = int(sx * art_w / dw)
    xb = max(xa + 1, int((sx + 1) * art_w / dw))
    ya = int(sy * art_h / dh)
    yb = max(ya + 1, int((sy + 1) * art_h / dh))
    best, best_r = " ", -1
    for ay in range(ya, min(art_h, yb)):
        row = art[ay]
        for ax in range(xa, min(art_w, xb)):
            ch = row[ax] if ax < len(row) else " "
            r = rank.get(ch, 1 if ch not in " " else 0)
            if r > best_r:
                best, best_r = ch, r
    return best


def _head_box_panel(width: int, height: int) -> tuple[int, int, int, int]:
    """Pixel rect for CNS overlay: shrunk box inside the fly head band."""
    _art, _aw, _ah, dw, dh, ox, oy = _fit_geometry(width, height)
    hx0 = ox + int(_HEAD_X0 * dw)
    hx1 = ox + int(_HEAD_X1 * dw)
    hy0 = oy + int(_HEAD_Y0 * dh)
    hy1 = oy + int(_HEAD_Y1 * dh)
    bw = max(6, hx1 - hx0)
    bh = max(4, hy1 - hy0)
    pw = max(6, int(bw * _CNS_FILL))
    ph = max(4, int(bh * _CNS_FILL))
    x0 = hx0 + (bw - pw) // 2
    # bias upward within head box (eyes / top of head, not belly)
    y0 = hy0 + max(0, int((bh - ph) * 0.12))
    x0 = int(np.clip(x0, 0, max(0, width - pw)))
    y0 = int(np.clip(y0, 0, max(0, height - ph)))
    return x0, y0, pw, ph


def fly_body_and_head_masks(
    width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """body mask, edge mask, head-band mask (for stencil)."""
    art, art_w, art_h, dw, dh, x0, y0 = _fit_geometry(width, height)
    body = np.zeros((height, width), dtype=bool)
    head = np.zeros((height, width), dtype=bool)
    for cy in range(height):
        for cx in range(width):
            sx = cx - x0
            sy = cy - y0
            if sx < 0 or sy < 0 or sx >= dw or sy >= dh:
                continue
            ch = _sample_art_cell(art, art_w, art_h, dw, dh, sx, sy)
            if ch not in SHADES:
                continue
            body[cy, cx] = True
            # art-space coords for head band
            ax = sx * art_w / dw
            ay = sy * art_h / dh
            if (
                _HEAD_X0 * art_w <= ax <= _HEAD_X1 * art_w
                and _HEAD_Y0 * art_h <= ay <= _HEAD_Y1 * art_h
            ):
                head[cy, cx] = True
    edge = np.zeros_like(body)
    if body.any():
        up = np.zeros_like(body)
        down = np.zeros_like(body)
        left = np.zeros_like(body)
        right = np.zeros_like(body)
        up[1:, :] = body[:-1, :]
        down[:-1, :] = body[1:, :]
        left[:, 1:] = body[:, :-1]
        right[:, :-1] = body[:, 1:]
        edge = body & ~(up & down & left & right)
    return body, edge, head


def paint_ascii_fly(
    lib,
    cv,
    x0: int,
    y0: int,
    width: int,
    height: int,
    engine: LivingEngine | None = None,
    frame: FrameState | None = None,
) -> None:
    """Fly silhouette; triad top-left XY connectome shrunk onto the head."""
    if width < 8 or height < 4:
        return

    title_h = 1
    body_h = max(3, height - title_h)
    body_y = y0 + title_h

    lib.caca_set_color_ansi(cv, 0x0F, 0x00)
    title = b"ASCII FLY + XY"
    lib.caca_put_str(cv, x0 + max(0, (width - len(title)) // 2), y0, title)

    body, edge, head = fly_body_and_head_masks(width, body_h)

    # Dim body silhouette
    for y in range(body_h):
        for x in range(width):
            if edge[y, x]:
                lib.caca_set_color_ansi(cv, 0x0F, 0x00)
                lib.caca_put_char(cv, x0 + x, body_y + y, ord("@"))
            elif body[y, x]:
                lib.caca_set_color_ansi(cv, 0x02, 0x00)
                lib.caca_put_char(cv, x0 + x, body_y + y, ord(":"))

    if engine is None or frame is None:
        return

    # Same projection as triad top-left pane: XY
    _, amp = activity_amp(engine, frame)
    if amp.size == 0:
        return
    xyz = viz_xyz(engine)
    uv = project_plane(xyz, "xy")

    px, py, pw, ph = _head_box_panel(width, body_h)
    cns = np.zeros((body_h, width), dtype=np.float32)
    _raster_into(cns, uv, amp, engine, frame, px, py, pw, ph)

    # Paint CNS only where head silhouette is (and slightly inside head box)
    for y in range(py, min(body_h, py + ph)):
        for x in range(px, min(width, px + pw)):
            if not head[y, x] and not body[y, x]:
                continue
            # Prefer head cells; allow a little spill in the shrunk box on body
            if not head[y, x] and not (
                px + pw * 0.15 <= x <= px + pw * 0.85 and py + ph * 0.1 <= y <= py + ph * 0.85
            ):
                continue
            v = float(cns[y, x])
            if v <= 0.02:
                continue
            ch = ord(CACA_CHARS[min(len(CACA_CHARS) - 1, int(v * (len(CACA_CHARS) - 1) + 1e-6))])
            pair = _caca_ansi_pair(v)
            fg = pair & 0x0F
            if fg == 0x00:
                fg = 0x03
            lib.caca_set_color_ansi(cv, fg, 0x00)
            lib.caca_put_char(cv, x0 + x, body_y + y, ch)
