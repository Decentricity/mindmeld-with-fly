"""Compose visualization buffers from reservoir activity."""
from __future__ import annotations

import math

import numpy as np

from .engine import FrameState, LivingEngine
from .project import (
    activity_amp,
    commit_trail,
    orbit_frame_for_cloud,
    project_orbit_ortho,
    project_plane,
    project_wire_segments,
    triad_panels,
    viz_xyz,
)


def activity_grid(
    engine: LivingEngine,
    frame: FrameState,
    width: int,
    height: int,
    *,
    commit: bool = True,
) -> np.ndarray:
    """Rasterize neuron activity into a [H,W] float grid in [0,1]."""
    act, amp = activity_amp(engine, frame)
    if act.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    if getattr(engine, "camera", "triad") == "orbit":
        grid = _orbit_grid(engine, frame, amp, width, height)
    else:
        grid = _triad_grid(engine, frame, amp, width, height)

    if commit:
        commit_trail(engine, frame)
    return np.clip(grid, 0, 1)


def _triad_grid(engine: LivingEngine, frame: FrameState, amp: np.ndarray, width: int, height: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.float32)
    xyz = viz_xyz(engine)
    for panel in triad_panels(width, height):
        if panel.w < 8 or panel.h < 4:
            continue
        if panel.kind == "orbit":
            _draw_orbit_panel(grid, engine, frame, amp, xyz, panel.x0, panel.y0, panel.w, panel.h)
        else:
            uv = project_plane(xyz, panel.kind)
            _raster_into(grid, uv, amp, engine, frame, panel.x0, panel.y0, panel.w, panel.h)
        _label(grid, panel.x0 + 1, panel.y0, panel.name)

    mid_x = width // 2
    mid_y = height // 2
    if 0 <= mid_x < width:
        grid[:, mid_x] = np.maximum(grid[:, mid_x], 0.28)
    if 0 <= mid_y < height:
        grid[mid_y, :] = np.maximum(grid[mid_y, :], 0.28)
    return grid


def _orbit_grid(engine: LivingEngine, frame: FrameState, amp: np.ndarray, width: int, height: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.float32)
    xyz = viz_xyz(engine)
    _draw_orbit_panel(grid, engine, frame, amp, xyz, 0, 0, width, height)
    deg_y = int(round(math.degrees(float(engine.yaw)))) % 360
    deg_p = int(round(math.degrees(float(engine.pitch))))
    _label(grid, 1, 0, f"Y{deg_y} P{deg_p}")
    return grid


def _draw_orbit_panel(
    grid: np.ndarray,
    engine: LivingEngine,
    frame: FrameState,
    amp: np.ndarray,
    xyz: np.ndarray,
    x0: int,
    y0: int,
    pw: int,
    ph: int,
) -> None:
    yaw = float(engine.yaw)
    pitch = float(engine.pitch)
    center, radius, segs = orbit_frame_for_cloud(xyz, yaw, pitch)
    # Outer bounding box only (12 edges); letterbox so aspect stays rigid.
    for ua, ub in project_wire_segments(segs, yaw, pitch, center=center, radius=radius):
        xa, ya = _uv_to_square_panel(ua[0], ua[1], x0, y0, pw, ph)
        xb, yb = _uv_to_square_panel(ub[0], ub[1], x0, y0, pw, ph)
        _draw_line(grid, xa, ya, xb, yb, 0.32)
    uv, depth = project_orbit_ortho(xyz, yaw, pitch, center=center, radius=radius)
    _raster_into_square(grid, uv, amp, engine, frame, x0, y0, pw, ph, depth=depth)


def _uv_to_square_panel(u: float, v: float, x0: int, y0: int, pw: int, ph: int) -> tuple[int, int]:
    """Map square UV content into a panel with letterboxing (isotropic pixels)."""
    side = max(1, min(pw, ph))
    ox = x0 + (pw - side) // 2
    oy = y0 + (ph - side) // 2
    x = int(round(float(np.clip(u, 0, 1) * (side - 1)))) + ox
    y = int(round(float(np.clip(v, 0, 1) * (side - 1)))) + oy
    return x, y


def _uv_array_to_square_panel(uv: np.ndarray, x0: int, y0: int, pw: int, ph: int) -> tuple[np.ndarray, np.ndarray]:
    side = max(1, min(pw, ph))
    ox = x0 + (pw - side) // 2
    oy = y0 + (ph - side) // 2
    xs = np.clip((uv[:, 0] * (side - 1)).astype(np.int32), 0, side - 1) + ox
    ys = np.clip((uv[:, 1] * (side - 1)).astype(np.int32), 0, side - 1) + oy
    return xs, ys


def _raster_into_square(
    grid: np.ndarray,
    uv: np.ndarray,
    amp: np.ndarray,
    engine: LivingEngine,
    frame: FrameState,
    x0: int,
    y0: int,
    pw: int,
    ph: int,
    depth: np.ndarray | None = None,
):
    xs, ys = _uv_array_to_square_panel(uv, x0, y0, pw, ph)
    _raster_xy(grid, xs, ys, amp, engine, frame, x0, y0, pw, ph, depth=depth)


def _raster_into(
    grid: np.ndarray,
    uv: np.ndarray,
    amp: np.ndarray,
    engine: LivingEngine,
    frame: FrameState,
    x0: int,
    y0: int,
    pw: int,
    ph: int,
    depth: np.ndarray | None = None,
):
    xs = np.clip((uv[:, 0] * (pw - 1)).astype(np.int32), 0, pw - 1) + x0
    ys = np.clip((uv[:, 1] * (ph - 1)).astype(np.int32), 0, ph - 1) + y0
    _raster_xy(grid, xs, ys, amp, engine, frame, x0, y0, pw, ph, depth=depth)


def _raster_xy(
    grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    amp: np.ndarray,
    engine: LivingEngine,
    frame: FrameState,
    x0: int,
    y0: int,
    pw: int,
    ph: int,
    depth: np.ndarray | None = None,
):
    order = np.arange(len(amp))
    if depth is not None:
        order = np.argsort(depth)

    if frame.view == "region":
        rid = engine.viz["region_id"].astype(np.float32)
        nreg = max(1, len(engine.viz.get("region_names", [""])))
        val = (rid / nreg) * 0.55 + 0.45 * amp
        if depth is None:
            np.maximum.at(grid, (ys, xs), val)
        else:
            for i in order:
                y, x = int(ys[i]), int(xs[i])
                if grid[y, x] < val[i]:
                    grid[y, x] = val[i]
    elif frame.view == "heatmap":
        sub = np.zeros((ph, pw), dtype=np.float32)
        np.add.at(sub, (ys - y0, xs - x0), amp)
        sub = _box_blur(sub, 1)
        m = sub.max() + 1e-6
        sub /= m
        np.maximum(grid[y0 : y0 + ph, x0 : x0 + pw], sub, out=grid[y0 : y0 + ph, x0 : x0 + pw])
    elif frame.view == "storm":
        prev = np.abs(engine.prev_activity)
        act = np.abs(frame.activity)
        if prev.shape != act.shape:
            prev = np.zeros_like(act)
        delta = np.clip(act - prev, 0, None)
        if delta.max() > 0:
            d = np.clip(delta / (np.percentile(delta, 90) + 1e-6), 0, 1)
        else:
            d = amp
        if depth is None:
            np.maximum.at(grid, (ys, xs), d)
        else:
            for i in order:
                y, x = int(ys[i]), int(xs[i])
                if grid[y, x] < d[i]:
                    grid[y, x] = d[i]
        step_e = max(1, max(1, len(engine.edges)) // 800) if len(engine.edges) else 1
        for a, b in engine.edges[::step_e]:
            if amp[a] > 0.25 or amp[b] > 0.25:
                _draw_line(grid, xs[a], ys[a], xs[b], ys[b], 0.35 * max(amp[a], amp[b]))
    else:  # whole
        if depth is None:
            np.maximum.at(grid, (ys, xs), amp)
        else:
            for i in order:
                y, x = int(ys[i]), int(xs[i])
                if grid[y, x] < amp[i]:
                    grid[y, x] = amp[i]
        step_e = max(1, len(engine.edges) // 500) if len(engine.edges) else 1
        for a, b in engine.edges[::step_e]:
            strength = 0.15 * min(amp[a], amp[b])
            if strength > 0.02:
                _draw_line(grid, xs[a], ys[a], xs[b], ys[b], strength)


def _label(grid: np.ndarray, x: int, y: int, text: str):
    for i, _ch in enumerate(text[:8]):
        xx = x + i
        if 0 <= y < grid.shape[0] and 0 <= xx < grid.shape[1]:
            grid[y, xx] = max(grid[y, xx], 0.92)


def _box_blur(g: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return g
    out = g.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            out += np.roll(np.roll(g, dy, axis=0), dx, axis=1)
    return out / float((2 * r + 1) ** 2)


def _draw_line(grid, x0, y0, x1, y1, val):
    n = int(max(abs(x1 - x0), abs(y1 - y0), 1))
    for i in range(n + 1):
        t = i / n
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            if grid[y, x] < val:
                grid[y, x] = val
