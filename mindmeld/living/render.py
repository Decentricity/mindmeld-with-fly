"""Terminal renderers: ANSI + real libcaca (tmux-safe via mplay-caca env)."""
from __future__ import annotations

import ctypes
import io
import os
import shutil
import sys
from pathlib import Path
from typing import TextIO

import numpy as np

from .caca_env import apply_mplay_caca_env
from .engine import FrameState, LivingEngine
from .stim import Stimulator
from .views import activity_grid

BLOCKS = " ·░▒▓█"
CACA_CHARS = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
TOP_HUD = 3  # connectome + stim + status (drawn above the image)
BOTTOM_HUD = 2  # keys + disclaimer


def _color256(v: float) -> int:
    """Cyberpunk 256-color ramp: green → pink → yellow → red → white."""
    v = float(np.clip(v, 0, 1))
    if v < 0.2:
        t = v / 0.2
        r, g, b = 0, int(3 + 2 * t), int(1 + t)  # neon green
    elif v < 0.45:
        t = (v - 0.2) / 0.25
        r, g, b = int(3 + 2 * t), int(2 * (1 - t)), int(3 + 2 * t)  # hot pink/magenta
    elif v < 0.7:
        t = (v - 0.45) / 0.25
        r, g, b = 5, int(4 + t), int(2 * (1 - t))  # yellow
    else:
        t = (v - 0.7) / 0.3
        r, g, b = 5, int(5 * (1 - 0.4 * t)), int(5 * t)  # red → white-ish
    return 16 + 36 * min(5, max(0, r)) + 6 * min(5, max(0, g)) + min(5, max(0, b))


# libcaca ANSI: green / magenta / yellow / red / white (no dark blue)
_CACA_COLORS = (
    0x00,  # black
    0x02,  # green
    0x05,  # magenta
    0x03,  # yellow/brown
    0x01,  # red
    0x07,  # white
    0x0F,  # bright white
)


def _caca_ansi_pair(v: float) -> int:
    """Pack fg/bg for caca_set_color_ansi: (bg<<4)|fg — dark bg, bright fg."""
    idx = min(len(_CACA_COLORS) - 1, int(float(np.clip(v, 0, 1)) * (len(_CACA_COLORS) - 1) + 1e-6))
    fg = _CACA_COLORS[idx]
    bg = 0x00
    return (bg << 4) | fg


def connectome_banner(engine: LivingEngine) -> str:
    """Exact identity of the connectome being simulated."""
    graph = getattr(engine, "graph", "?")
    weighting = getattr(engine, "weighting", "?")
    n = getattr(engine, "n", None)
    n_viz = engine.viz.get("n") if getattr(engine, "viz", None) else None
    if graph == "oruk499":
        detail = (
            "Oruk-like ~499-neuron strongly-connected approx built from MaleCNS "
            "(local reconstruction; published Oruk body IDs are not public)"
        )
    elif graph == "replay":
        detail = "replay of a saved living-viz recording (no live GPU reservoir)"
    else:
        detail = f"full MaleCNS central-nervous-system connectome, weighting={weighting}"
    bits = [
        "CONNECTOME: Drosophila melanogaster male (\u2642) — FlyEM / Janelia MaleCNS v1.0",
        detail,
    ]
    if isinstance(n, int) and n > 0:
        bits.append(f"{n:,} graph neurons")
    if isinstance(n_viz, int) and n_viz > 0:
        bits.append(f"{n_viz:,} somata on screen")
    return " | ".join(bits)


def stim_banner(engine: LivingEngine, frame: FrameState) -> str:
    stim = getattr(engine, "stim", None)
    if isinstance(stim, Stimulator):
        return stim.describe()
    desc = Stimulator.DESCRIPTIONS.get(frame.mode)
    if desc:
        return f"STIM [{frame.mode}] @{frame.intensity:.2f}: {desc}"
    return f"STIM [{frame.mode}] @{frame.intensity:.2f}"


def status_banner(engine: LivingEngine, frame: FrameState, fps: float, renderer_name: str) -> str:
    vram = f" | VRAM {frame.vram_bytes / 1024**3:.2f}G" if frame.vram_bytes else ""
    cam = getattr(frame, "camera", getattr(engine, "camera", "triad"))
    cam_note = {
        "triad": "triad panels XY|XZ|YZ + orbit peek",
        "orbit": "single rotatable camera (,/. yaw, j/k pitch)",
    }.get(cam, cam)
    return (
        f"{renderer_name} | cam={cam} ({cam_note}) | view={frame.view} | "
        f"{'PAUSE' if frame.paused else 'LIVE'} | step={frame.step} | "
        f"sim {frame.steps_per_sec:.0f}/s | fps={fps:.1f} | "
        f"active={frame.active} | mean={frame.mean:+.3f}{vram}"
    )


def keys_line() -> str:
    return (
        "keys: [tab]cam  [,/./j/k]orbit  [space]pause  [n]stim  [v]view  "
        "[[/]]intens  [+/-]speed  [i]impulse  [r]reset  [R]record=NPZ+MP4  [q]quit"
    )


def disclaimer_line() -> str:
    return (
        "assumed sparse reservoir dynamics — not biophysics | "
        "R records NPZ+cinema MP4 by default (--no-cinema to opt out)"
    )


def top_hud_lines(engine: LivingEngine, frame: FrameState, fps: float, renderer_name: str) -> list[str]:
    return [
        connectome_banner(engine),
        stim_banner(engine, frame),
        status_banner(engine, frame, fps, renderer_name),
    ]


def _bind_caca(lib) -> None:
    lib.caca_create_canvas.restype = ctypes.c_void_p
    lib.caca_clear_canvas.argtypes = [ctypes.c_void_p]
    lib.caca_put_char.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
    lib.caca_put_str.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    lib.caca_set_color_ansi.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8]
    lib.caca_get_canvas_width.restype = ctypes.c_int
    lib.caca_get_canvas_width.argtypes = [ctypes.c_void_p]
    lib.caca_get_canvas_height.restype = ctypes.c_int
    lib.caca_get_canvas_height.argtypes = [ctypes.c_void_p]
    lib.caca_export_canvas_to_memory.restype = ctypes.c_void_p
    lib.caca_export_canvas_to_memory.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.caca_free_canvas.argtypes = [ctypes.c_void_p]
    if hasattr(lib, "caca_free"):
        lib.caca_free.argtypes = [ctypes.c_void_p]


def paint_caca_frame(
    lib,
    cv,
    engine: LivingEngine,
    frame: FrameState,
    fps: float,
    renderer_name: str = "living/caca",
    *,
    commit: bool = True,
) -> None:
    """Paint one living frame onto an existing libcaca canvas (black background)."""
    w = lib.caca_get_canvas_width(cv)
    h = lib.caca_get_canvas_height(cv)
    grid_h = max(8, h - TOP_HUD - BOTTOM_HUD)
    # True black backdrop — never leave stale glyphs / light cells.
    lib.caca_set_color_ansi(cv, 0x00, 0x00)
    lib.caca_clear_canvas(cv)

    grid = activity_grid(engine, frame, w, grid_h, commit=commit)

    lib.caca_set_color_ansi(cv, 0x0F, 0x00)  # bright white on black HUD
    for i, ln in enumerate(top_hud_lines(engine, frame, fps, renderer_name)):
        padded = (ln[:w] + " " * w)[:w]
        lib.caca_put_str(cv, 0, i, padded.encode())

    for y in range(grid_h):
        for x in range(w):
            v = float(grid[y, x])
            ch = ord(CACA_CHARS[min(len(CACA_CHARS) - 1, int(v * (len(CACA_CHARS) - 1) + 1e-6))])
            pair = _caca_ansi_pair(v)
            fg, bg = pair & 0x0F, (pair >> 4) & 0x0F  # bg nibble is 0 → black
            if v <= 0.015:
                fg, bg = 0x00, 0x00  # empty cells stay pure black
            lib.caca_set_color_ansi(cv, fg, bg)
            lib.caca_put_char(cv, x, TOP_HUD + y, ch)

    lib.caca_set_color_ansi(cv, 0x0F, 0x00)
    y0 = TOP_HUD + grid_h
    lib.caca_put_str(cv, 0, y0, (keys_line()[:w] + " " * w)[:w].encode())
    lib.caca_put_str(cv, 0, y0 + 1, (disclaimer_line()[:w] + " " * w)[:w].encode())


def export_canvas_png(lib, cv, path: Path) -> Path:
    """Export libcaca canvas via TGA → PNG, composited onto pure black."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbytes = ctypes.c_size_t()
    ptr = lib.caca_export_canvas_to_memory(cv, b"tga", ctypes.byref(nbytes))
    if not ptr or nbytes.value <= 0:
        raise RuntimeError("caca_export_canvas_to_memory(tga) failed")
    try:
        buf = ctypes.string_at(ptr, nbytes.value)
    finally:
        if hasattr(lib, "caca_free"):
            lib.caca_free(ptr)
    im = Image.open(io.BytesIO(buf))
    # Force black background (TGA may be RGBA with transparent empty cells).
    rgb = Image.new("RGB", im.size, (0, 0, 0))
    if im.mode == "RGBA":
        rgb.paste(im, mask=im.split()[3])
    else:
        rgb.paste(im.convert("RGB"))
    rgb.save(path, format="PNG")
    return path


def save_living_screenshot(
    engine: LivingEngine,
    frame: FrameState,
    path: Path,
    *,
    cols: int = 120,
    rows: int = 36,
    fps: float = 12.0,
) -> Path:
    """Headless screenshot of the real libcaca living frame (no X server / display)."""
    lib = ctypes.CDLL("libcaca.so.0")
    _bind_caca(lib)
    total_h = rows + TOP_HUD + BOTTOM_HUD
    cv = lib.caca_create_canvas(cols, total_h)
    if not cv:
        raise RuntimeError("caca_create_canvas failed")
    try:
        paint_caca_frame(lib, cv, engine, frame, fps, renderer_name="living/caca(export)", commit=False)
        return export_canvas_png(lib, cv, path)
    finally:
        lib.caca_free_canvas(cv)


class AnsiRenderer:
    name = "ansi"

    def __init__(self, stream: TextIO | None = None):
        self.stream = stream or sys.stdout
        self.color = not os.environ.get("NO_COLOR")
        self._hide = False
        self.env_info = {"renderer": "ansi"}

    def size(self) -> tuple[int, int]:
        size = shutil.get_terminal_size(fallback=(100, 35))
        rows = max(12, size.lines - TOP_HUD - BOTTOM_HUD)
        return max(40, size.columns), rows

    def begin(self):
        # Avoid tmux alternate-screen fights when possible (mplay-caca tip).
        if os.environ.get("TMUX") and shutil.which("tmux"):
            os.system("tmux set-option -p alternate-screen off >/dev/null 2>&1")
        self.stream.write("\033[?25l\033[2J")
        self._hide = True
        self.stream.flush()

    def end(self):
        if self._hide:
            self.stream.write("\033[?25h\033[0m\n")
            self.stream.flush()

    def draw(self, engine: LivingEngine, frame: FrameState, fps: float, *, commit: bool = True):
        cols, rows = self.size()
        grid = activity_grid(engine, frame, cols, rows, commit=commit)
        self.stream.write("\033[H")
        lines = [ln[:cols] for ln in top_hud_lines(engine, frame, fps, f"living/{self.name}")]
        for y in range(rows):
            row = []
            for x in range(cols):
                v = float(grid[y, x])
                ch = BLOCKS[min(len(BLOCKS) - 1, int(v * (len(BLOCKS) - 1) + 1e-6))]
                if self.color and v > 0.02:
                    row.append(f"\033[38;5;{_color256(v)}m{ch}")
                else:
                    row.append(ch)
            if self.color:
                row.append("\033[0m")
            lines.append("".join(row))
        lines.append(keys_line()[:cols])
        lines.append(disclaimer_line()[:cols])
        self.stream.write("\n".join(lines))
        self.stream.flush()


class CacaRenderer:
    """Real libcaca display using mplay-caca's tmux driver policy (slang)."""

    name = "caca"

    def __init__(self, stream: TextIO | None = None):
        self.stream = stream or sys.stdout
        self.env_info = apply_mplay_caca_env()
        self._lib = None
        self._cv = None
        self._dp = None
        self._fallback: AnsiRenderer | None = None
        self._w = 80
        self._h = 24
        try:
            lib = ctypes.CDLL("libcaca.so.0")
            _bind_caca(lib)
            lib.caca_create_display_with_driver.restype = ctypes.c_void_p
            lib.caca_create_display_with_driver.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.caca_create_display.restype = ctypes.c_void_p
            lib.caca_create_display.argtypes = [ctypes.c_void_p]
            lib.caca_refresh_display.argtypes = [ctypes.c_void_p]
            lib.caca_get_canvas.restype = ctypes.c_void_p
            lib.caca_get_canvas.argtypes = [ctypes.c_void_p]
            lib.caca_free_display.argtypes = [ctypes.c_void_p]
            self._lib = lib
        except OSError as e:
            self._fallback = AnsiRenderer(stream)
            self._fallback.name = "caca-fallback-ansi"
            self.env_info["fallback"] = f"libcaca load failed: {e}"

    def size(self) -> tuple[int, int]:
        if self._fallback:
            return self._fallback.size()
        if self._cv:
            w = max(40, self._lib.caca_get_canvas_width(self._cv))
            h = max(12, self._lib.caca_get_canvas_height(self._cv) - TOP_HUD - BOTTOM_HUD)
            return w, h
        size = shutil.get_terminal_size(fallback=(100, 35))
        return max(40, size.columns), max(12, size.lines - TOP_HUD - BOTTOM_HUD)

    def begin(self):
        if self._fallback:
            self._fallback.begin()
            return
        # mplay-caca assumes a real terminal. Never let slang abort a dumb/pipe session.
        term = os.environ.get("TERM", "")
        if (not sys.stdout.isatty()) or term in ("", "dumb"):
            self._fallback = AnsiRenderer(self.stream)
            self._fallback.name = "caca-fallback-ansi"
            self.env_info["fallback"] = f"non-interactive/dumb TERM={term!r}; ANSI fallback"
            print(
                f"[living/caca] {self.env_info['fallback']} "
                f"(in a real tmux pane use: living-caca — CACA_DRIVER=slang like mplay-caca)",
                file=sys.stderr,
            )
            self._fallback.begin()
            return
        if os.environ.get("TMUX") and shutil.which("tmux"):
            os.system("tmux set-option -p alternate-screen off >/dev/null 2>&1")
        lib = self._lib
        cols, rows = self.size()
        self._cv = lib.caca_create_canvas(cols, rows + TOP_HUD + BOTTOM_HUD)
        driver = os.environ.get("CACA_DRIVER", "slang").encode()
        self._dp = lib.caca_create_display_with_driver(self._cv, driver)
        if not self._dp:
            self._dp = lib.caca_create_display(self._cv)
        if not self._dp:
            self._fallback = AnsiRenderer(self.stream)
            self._fallback.name = "caca-fallback-ansi"
            self.env_info["fallback"] = "caca_create_display failed; using ANSI"
            try:
                lib.caca_free_canvas(self._cv)
            except Exception:
                pass
            self._cv = None
            print(f"[living/caca] {self.env_info['fallback']}", file=sys.stderr)
            self._fallback.begin()
            return
        self._cv = lib.caca_get_canvas(self._dp) or self._cv
        self._w = lib.caca_get_canvas_width(self._cv)
        self._h = lib.caca_get_canvas_height(self._cv)
        print(
            f"[living/caca] using libcaca driver={self.env_info['CACA_DRIVER']} "
            f"(policy from {self.env_info['source']}; TMUX={self.env_info['TMUX']})",
            file=sys.stderr,
        )

    def end(self):
        if self._fallback:
            self._fallback.end()
            return
        lib = self._lib
        if self._dp:
            lib.caca_free_display(self._dp)
            self._dp = None
        # canvas owned by display when created with it; free_display handles it
        self._cv = None

    def draw(self, engine: LivingEngine, frame: FrameState, fps: float, *, commit: bool = True):
        if self._fallback:
            self._fallback.draw(engine, frame, fps, commit=commit)
            return
        rname = f"living/caca({self.env_info.get('CACA_DRIVER', '?')})"
        paint_caca_frame(
            self._lib, self._cv, engine, frame, fps, renderer_name=rname, commit=commit
        )
        self._lib.caca_refresh_display(self._dp)

    def save_screenshot(self, path: Path, engine: LivingEngine, frame: FrameState, fps: float) -> Path:
        """Export current (or freshly painted) libcaca canvas to PNG on black."""
        if self._fallback or not self._cv:
            return save_living_screenshot(engine, frame, path, fps=fps)
        rname = f"living/caca({self.env_info.get('CACA_DRIVER', '?')})"
        paint_caca_frame(self._lib, self._cv, engine, frame, fps, renderer_name=rname, commit=False)
        return export_canvas_png(self._lib, self._cv, path)


def make_renderer(name: str) -> AnsiRenderer | CacaRenderer:
    name = (name or "caca").lower()
    if name in ("caca", "libcaca"):
        return CacaRenderer()
    if name in ("ansi", "unicode", "term"):
        return AnsiRenderer()
    raise ValueError(f"Unknown renderer: {name}")
