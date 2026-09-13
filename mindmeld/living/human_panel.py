"""HUMAN EEG superpane for the living libcaca canvas (illustrative, not spatial EEG)."""
from __future__ import annotations

import numpy as np

from ..eeg.features import FEATURE_NAMES, N_BANDS

# Muse 2 BrainFlow channel order (same as neurofeedback-muse): TP9, AF7, AF8, TP10
CHANNEL_NAMES = ("TP9", "AF7", "AF8", "TP10")
CHANNEL_COLORS = (0x02, 0x04, 0x03, 0x01)  # green, cyan, yellow, red
CHANNEL_LABELS = (
    "behind L ear",
    "L forehead",
    "R forehead",
    "behind R ear",
)
BAND_NAMES = FEATURE_NAMES  # delta..gamma
BAND_SHORT = ("D", "T", "A", "B", "G")
BAND_COLORS = (0x05, 0x02, 0x03, 0x01, 0x04)  # mag, green, yellow, red, cyan

_TRACE: list[np.ndarray] = []
_FLY_TRACE: list[np.ndarray] = []
_TRACE_MAX = 120
_RMS_TRACE: list[np.ndarray] = []
_MASK_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _brain_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h * 0.42, w * 0.50
    ry, rx = max(2.0, h * 0.40), max(3.0, w * 0.36)
    head = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    cleft = (np.abs(xx - cx) < max(1.0, w * 0.015)) & (yy < cy + ry * 0.1)
    stem = ((yy - (cy + ry * 0.95)) / max(1.5, h * 0.16)) ** 2 + (
        (xx - cx) / max(1.5, w * 0.07)
    ) ** 2 <= 1.0
    return (head | stem) & (~cleft)


def _brain_edge(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    up = np.zeros_like(m)
    down = np.zeros_like(m)
    left = np.zeros_like(m)
    right = np.zeros_like(m)
    up[1:, :] = m[:-1, :]
    down[:-1, :] = m[1:, :]
    left[:, 1:] = m[:, :-1]
    right[:, :-1] = m[:, 1:]
    return m & ~(up & down & left & right)


def _electrode_regions(h: int, w: int, mask: np.ndarray) -> np.ndarray:
    """Per-pixel electrode id 0..3 (TP9,AF7,AF8,TP10) or -1 outside mask.

    Illustrative top-ish view: forehead toward top of oval, ears toward sides/bottom.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h * 0.42, w * 0.50
    ry, rx = max(2.0, h * 0.40), max(3.0, w * 0.36)
    anchors = np.array(
        [
            [cy + 0.25 * ry, cx - 0.78 * rx],  # TP9 left temporal
            [cy - 0.55 * ry, cx - 0.42 * rx],  # AF7 left forehead
            [cy - 0.55 * ry, cx + 0.42 * rx],  # AF8 right forehead
            [cy + 0.25 * ry, cx + 0.78 * rx],  # TP10 right temporal
        ],
        dtype=np.float32,
    )
    regions = np.full((h, w), -1, dtype=np.int8)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return regions
    pts = np.stack([ys.astype(np.float32), xs.astype(np.float32)], axis=1)
    d = (pts[:, None, 0] - anchors[None, :, 0]) ** 2 + (pts[:, None, 1] - anchors[None, :, 1]) ** 2
    regions[ys, xs] = np.argmin(d, axis=1).astype(np.int8)
    return regions


def _cached_brain(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (h, w)
    hit = _MASK_CACHE.get(key)
    if hit is not None:
        return hit
    mask = _brain_mask(h, w)
    edge = _brain_edge(mask)
    regions = _electrode_regions(h, w, mask)
    _MASK_CACHE[key] = (mask, edge, regions)
    return mask, edge, regions


def _norm_rms(rms: np.ndarray) -> np.ndarray:
    rms = np.asarray(rms, dtype=np.float32).reshape(-1)
    if rms.size < 4:
        rms = np.pad(rms, (0, 4 - rms.size))
    rms = rms[:4]
    return np.clip((rms - 5.0) / 80.0, 0.05, 1.0).astype(np.float32)


def _coerce_show(vec: np.ndarray | None, n: int = N_BANDS) -> np.ndarray:
    if vec is None:
        return np.zeros(n, dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size < n:
        v = np.pad(v, (0, n - v.size))
    v = v[:n]
    s = float(np.abs(v).sum())
    if s > 1e-6:
        return (np.abs(v) / s).astype(np.float32)
    return np.clip(np.abs(v), 0.0, 1.0).astype(np.float32)


def _paint_meter_col(
    lib,
    cv,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: bytes,
    show: np.ndarray,
    short: bool = True,
) -> None:
    lib.caca_set_color_ansi(cv, 0x0F, 0x00)
    lib.caca_put_str(cv, x0, y0, title[: max(4, width)])
    for i in range(min(N_BANDS, height - 1)):
        yy = y0 + 1 + i
        name = BAND_SHORT[i] if short else BAND_NAMES[i][:5]
        val = float(show[i]) if i < len(show) else 0.0
        label_w = 2 if short else 5
        bar_w = max(1, int(np.clip(val, 0.05, 1.0) * max(3, width - label_w - 5)))
        lib.caca_set_color_ansi(cv, 0x0F, 0x00)
        lib.caca_put_str(cv, x0, yy, f"{name:<{label_w}}".encode())
        for x in range(bar_w):
            lib.caca_set_color_ansi(cv, BAND_COLORS[i], 0x00)
            lib.caca_put_char(cv, x0 + label_w + 1 + x, yy, ord("#"))
        lib.caca_set_color_ansi(cv, 0x07, 0x00)
        pct = f"{int(round(val * 100)):2d}"
        lib.caca_put_str(cv, x0 + label_w + 1 + bar_w + 1, yy, pct.encode())


def _paint_band_chart(
    lib,
    cv,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: bytes,
    traces: list[np.ndarray],
) -> None:
    if height < 3 or width < 8 or not traces:
        return
    lib.caca_set_color_ansi(cv, 0x0F, 0x00)
    lib.caca_put_str(cv, x0, y0, title[:width])
    chart_y0 = y0 + 1
    chart_h = height - 1
    for row in range(chart_h):
        lib.caca_set_color_ansi(cv, 0x00, 0x00)
        for x in range(width):
            lib.caca_put_char(cv, x0 + x, chart_y0 + row, ord(" "))
        lib.caca_set_color_ansi(cv, 0x07, 0x00)
        lib.caca_put_char(cv, x0, chart_y0 + row, ord("|"))
    lib.caca_set_color_ansi(cv, 0x07, 0x00)
    lib.caca_put_str(cv, x0, chart_y0 + chart_h - 1, b"+" + b"-" * (width - 1))

    trace = np.stack(traces[-width:], axis=0)
    tlen = trace.shape[0]
    n_series = min(N_BANDS, trace.shape[1])
    for bi in range(n_series):
        col = BAND_COLORS[bi]
        series = trace[:, bi]
        for ti in range(tlen):
            v = float(np.clip(series[ti], 0.0, 1.0))
            py = chart_h - 1 - int(v * (chart_h - 2) + 1e-6)
            py = int(np.clip(py, 0, chart_h - 2))
            px = width - tlen + ti
            if px < 1 or px >= width:
                continue
            px_draw = min(width - 1, px + (bi % 2))
            ch = ord(".*+@"[min(3, int(v * 3 + 1e-6))])
            lib.caca_set_color_ansi(cv, col, 0x00)
            lib.caca_put_char(cv, x0 + px_draw, chart_y0 + py, ch)


def paint_human_panel(lib, cv, session, x0: int, y0: int, width: int, height: int) -> None:
    """Brain | human+fly meters | dual band charts | conduit."""
    if height < 8 or width < 24:
        return

    e = np.asarray(
        session.last_e if session.last_e is not None else np.zeros(N_BANDS),
        dtype=np.float32,
    )
    e_raw = np.asarray(
        session.last_e_raw if getattr(session, "last_e_raw", None) is not None else e,
        dtype=np.float32,
    )
    sample = session.last_sample
    show = _coerce_show(e_raw)
    if float(show.sum()) <= 1e-6 and sample is not None:
        show = _coerce_show(sample.e)
    if float(show.sum()) <= 1e-6:
        show = _coerce_show(e)

    fly_show = _coerce_show(getattr(session, "last_fly_rel", None))

    rms = (
        sample.channel_rms
        if sample is not None and sample.channel_rms is not None
        else np.full(4, 30.0, dtype=np.float32)
    )
    rms_n = _norm_rms(rms)

    global _TRACE, _FLY_TRACE, _RMS_TRACE
    _TRACE.append(show.copy())
    _FLY_TRACE.append(fly_show.copy())
    _RMS_TRACE.append(rms_n.copy())
    if len(_TRACE) > _TRACE_MAX:
        _TRACE = _TRACE[-_TRACE_MAX:]
        _FLY_TRACE = _FLY_TRACE[-_TRACE_MAX:]
        _RMS_TRACE = _RMS_TRACE[-_TRACE_MAX:]

    split = max(14, int(width * 0.34))
    brain_w = split - 1
    right_x = split + 1
    right_w = max(16, width - right_x)
    sil_h = max(6, height - 1)

    # Right column: dual meter columns on top, then dual charts
    meter_h = N_BANDS + 1  # title + 5 bands
    mid = max(8, right_w // 2)
    hum_w = mid - 1
    fly_w = right_w - mid

    mask, edge, regions = _cached_brain(sil_h, brain_w)
    fill_chars = " .:-=+*#%@"
    energy = float(np.clip(np.linalg.norm(show) / 2.0, 0.0, 1.0))

    # --- electrode-colored brain ---
    for y in range(sil_h):
        for x in range(brain_w):
            if edge[y, x]:
                lib.caca_set_color_ansi(cv, 0x0F, 0x00)
                lib.caca_put_char(cv, x0 + x, y0 + y, ord("@"))
                continue
            rid = int(regions[y, x])
            if rid < 0:
                continue
            v = float(rms_n[rid])
            fi = min(len(fill_chars) - 1, int(0.25 + v * (len(fill_chars) - 1)))
            lib.caca_set_color_ansi(cv, CHANNEL_COLORS[rid], 0x00)
            lib.caca_put_char(cv, x0 + x, y0 + y, ord(fill_chars[fi]))

    lib.caca_set_color_ansi(cv, 0x0F, 0x00)
    legend = " ".join(
        f"{n}={int(round(float(rms_n[i])*100)):02d}" for i, n in enumerate(CHANNEL_NAMES)
    )
    lib.caca_put_str(cv, x0 + 1, y0, legend.encode()[: max(8, brain_w - 2)])

    # --- HUMAN + FLY meters side by side ---
    _paint_meter_col(
        lib,
        cv,
        x0=x0 + right_x,
        y0=y0,
        width=hum_w,
        height=meter_h,
        title=b"HUM D/T/A/B/G",
        show=show,
    )
    _paint_meter_col(
        lib,
        cv,
        x0=x0 + right_x + mid,
        y0=y0,
        width=fly_w,
        height=meter_h,
        title=b"FLY (reservoir)",
        show=fly_show,
    )

    # --- charts under meters ---
    chart_y0 = y0 + meter_h
    chart_avail = max(0, (y0 + sil_h) - chart_y0)
    if chart_avail >= 4:
        half_h = max(3, chart_avail // 2)
        _paint_band_chart(
            lib,
            cv,
            x0=x0 + right_x,
            y0=chart_y0,
            width=right_w,
            height=half_h,
            title=b"human bands  D/T/A/B/G",
            traces=_TRACE,
        )
        fly_chart_h = max(3, chart_avail - half_h)
        _paint_band_chart(
            lib,
            cv,
            x0=x0 + right_x,
            y0=chart_y0 + half_h,
            width=right_w,
            height=fly_chart_h,
            title=b"fly response (not bio EEG)",
            traces=_FLY_TRACE,
        )

    # conduit
    link_y = y0 + sil_h
    if link_y >= y0 + height:
        link_y = y0 + height - 1
    pulse = float(np.clip(energy * 1.2 + float(show.std()) * 2.0, 0.0, 1.0))
    mid_s = " EEG ~~~> FLY "
    pad = max(0, (width - len(mid_s)) // 2)
    lib.caca_set_color_ansi(cv, 0x02, 0x00)
    lib.caca_put_str(cv, x0, link_y, (("~" * pad + mid_s + "~" * width)[:width]).encode())
    half = int(3 + pulse * (width // 5))
    cx = width // 2
    for x in range(max(0, cx - half), min(width, cx + half)):
        lib.caca_set_color_ansi(cv, 0x05, 0x00)
        lib.caca_put_char(cv, x0 + x, link_y, ord("="))
