"""Headless GPU cinema renderer — black-background PNG/MP4 twin of living-caca."""
from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..download import ROOT
from .engine import FrameState, LivingEngine
from .project import commit_trail
from .render import connectome_banner, stim_banner, status_banner
from .views import activity_grid


def _parse_size(spec: str) -> tuple[int, int]:
    w, h = spec.lower().split("x")
    return max(320, int(w)), max(180, int(h))


def _colorize(grid: np.ndarray) -> np.ndarray:
    """Map [0,1] activity → high-contrast cyberpunk RGB on pure black."""
    g = np.clip(grid, 0, 1).astype(np.float32)
    # stops: black → neon green → hot pink → yellow → red → white
    stops_t = np.array([0.0, 0.10, 0.28, 0.50, 0.72, 1.0], dtype=np.float32)
    stops_c = np.array(
        [
            [0, 0, 0],
            [0, 255, 110],
            [255, 45, 200],
            [255, 235, 50],
            [255, 55, 70],
            [255, 255, 255],
        ],
        dtype=np.float32,
    )
    rgb = np.zeros(g.shape + (3,), dtype=np.float32)
    for c in range(3):
        rgb[..., c] = np.interp(g, stops_t, stops_c[:, c])
    # lift midtones so dark blues never dominate; boost bloom in pink/yellow
    bloom = _blur2d(g, 1)
    rgb[..., 0] = np.clip(rgb[..., 0] + 40.0 * bloom * bloom, 0, 255)
    rgb[..., 1] = np.clip(rgb[..., 1] + 25.0 * bloom, 0, 255)
    rgb[..., 2] = np.clip(rgb[..., 2] + 35.0 * bloom * (1.0 - g), 0, 255)
    # hard black for empty cells (keep faint grid ~0.20 visible as green/pink)
    rgb *= (g > 0.04)[..., None]
    return rgb.astype(np.uint8)


def _blur2d(g: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return g
    out = g.astype(np.float32).copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            out += np.roll(np.roll(g, dy, axis=0), dx, axis=1)
    return out / float((2 * r + 1) ** 2)


class CinemaRenderer:
    """Offscreen twin of the living viz (no X server)."""

    def __init__(self, width: int = 1280, height: int = 720, device: str = "cuda"):
        self.width = int(width)
        self.height = int(height)
        self.top = 56
        self.bottom = 40
        self.device = torch.device(
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self._font = ImageFont.load_default()
        self.last_rgb: np.ndarray | None = None

    @property
    def grid_size(self) -> tuple[int, int]:
        return self.width, max(64, self.height - self.top - self.bottom)

    def draw(self, engine: LivingEngine, frame: FrameState, fps: float, *, commit: bool = False) -> np.ndarray:
        from .project import orbit_frame_for_cloud, project_wire_segments, viz_xyz

        gw, gh = self.grid_size
        grid = activity_grid(engine, frame, gw, gh, commit=commit)
        # optional light GPU path: blur on device when available
        if self.device.type == "cuda":
            t = torch.as_tensor(grid, device=self.device)
            t4 = t[None, None]
            t4 = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(t4, (1, 1, 1, 1), mode="replicate"),
                kernel_size=3,
                stride=1,
            )
            glow = t4[0, 0].detach().float().cpu().numpy()
            grid = np.clip(0.72 * grid + 0.28 * glow, 0, 1)
        body = _colorize(grid)

        # Extra crisp cyberpunk wireframe on orbit (AABB-aligned with neurons)
        if getattr(engine, "camera", "triad") == "orbit" and engine.viz.get("n", 0) > 0:
            xyz = viz_xyz(engine)
            yaw, pitch = float(engine.yaw), float(engine.pitch)
            center, radius, segs = orbit_frame_for_cloud(xyz, yaw, pitch)
            body_img = Image.fromarray(body, mode="RGB")
            draw_b = ImageDraw.Draw(body_img)
            side = max(1, min(gw, gh))
            ox = (gw - side) // 2
            oy = (gh - side) // 2
            for ua, ub in project_wire_segments(segs, yaw, pitch, center=center, radius=radius):
                xa = int(round(float(np.clip(ua[0], 0, 1) * (side - 1)))) + ox
                ya = int(round(float(np.clip(ua[1], 0, 1) * (side - 1)))) + oy
                xb = int(round(float(np.clip(ub[0], 0, 1) * (side - 1)))) + ox
                yb = int(round(float(np.clip(ub[1], 0, 1) * (side - 1)))) + oy
                draw_b.line([(xa, ya), (xb, yb)], fill=(255, 60, 220), width=2)
            body = np.asarray(body_img, dtype=np.uint8)

        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # pure black
        y0 = self.top
        canvas[y0 : y0 + gh, :gw] = body[:gh, :gw]

        img = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(img)
        lines = [
            connectome_banner(engine)[:180],
            stim_banner(engine, frame)[:180],
            status_banner(engine, frame, fps, "living/cinema")[:180],
        ]
        y = 4
        for ln in lines:
            draw.text((8, y), ln, fill=(255, 120, 220), font=self._font)  # hot pink HUD
            y += 16
        foot = "R record = NPZ+MP4 | tab cam | n stim | cyberpunk cinema / assumed reservoir"
        draw.text((8, self.height - 28), foot[:180], fill=(80, 255, 160), font=self._font)
        cam = getattr(engine, "camera", "triad")
        draw.text(
            (8, self.height - 14),
            f"cam={cam} view={frame.view} step={frame.step}",
            fill=(255, 230, 80),
            font=self._font,
        )
        self.last_rgb = np.asarray(img, dtype=np.uint8)
        return self.last_rgb

    def save_png(self, path: Path) -> Path:
        if self.last_rgb is None:
            raise RuntimeError("CinemaRenderer.draw() first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.last_rgb, mode="RGB").save(path)
        return path


class CinemaSession:
    """Tied to Recorder: capture frames while recording; encode MP4 on stop."""

    def __init__(
        self,
        cinema: CinemaRenderer,
        *,
        enabled: bool = True,
        mp4_path: Path | None = None,
        frames_dir: Path | None = None,
        every: int = 1,
        keep_pngs: bool = False,
        fps: float = 15.0,
    ):
        self.cinema = cinema
        self.enabled = enabled
        self.every = max(1, int(every))
        self.keep_pngs = keep_pngs
        self.fps = float(fps)
        self.mp4_path = mp4_path
        self.frames_dir = frames_dir
        self.active = False
        self._i = 0
        self._written = 0
        self._stamp = ""

    def start(self) -> None:
        if not self.enabled:
            return
        self._stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_root = ROOT / "recordings"
        rec_root.mkdir(parents=True, exist_ok=True)
        if self.mp4_path is None:
            self.mp4_path = rec_root / f"living_cinema_{self._stamp}.mp4"
        if self.frames_dir is None:
            self.frames_dir = rec_root / f"cinema_frames_{self._stamp}"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        # clear prior frames in dir
        for p in self.frames_dir.glob("frame_*.png"):
            p.unlink(missing_ok=True)
        self.active = True
        self._i = 0
        self._written = 0

    def push(self, engine: LivingEngine, frame: FrameState, fps: float) -> None:
        if not self.active or not self.enabled:
            return
        self._i += 1
        if (self._i - 1) % self.every != 0:
            return
        self.cinema.draw(engine, frame, fps, commit=False)
        out = self.frames_dir / f"frame_{self._written:06d}.png"
        self.cinema.save_png(out)
        self._written += 1

    def stop_and_encode(self) -> Path | None:
        if not self.active:
            return None
        self.active = False
        if not self.enabled or self._written <= 0 or self.mp4_path is None:
            return None
        mp4 = Path(self.mp4_path)
        mp4.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            print("[cinema] ffmpeg not found; left PNG frames only", flush=True)
            return None
        pattern = str(self.frames_dir / "frame_%06d.png")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(self.fps),
            "-i",
            pattern,
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-movflags",
            "+faststart",
            str(mp4),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[cinema] ffmpeg failed: {e.stderr[-500:] if e.stderr else e}", flush=True)
            return None
        if not self.keep_pngs:
            for p in self.frames_dir.glob("frame_*.png"):
                p.unlink(missing_ok=True)
            try:
                self.frames_dir.rmdir()
            except OSError:
                pass
        print(f"[cinema] wrote {mp4} ({self._written} frames)", flush=True)
        return mp4


def make_cinema_from_args(args) -> CinemaSession | None:
    if getattr(args, "no_cinema", False):
        return None
    w, h = _parse_size(getattr(args, "cinema_size", "1280x720"))
    device = getattr(args, "device", "cuda")
    cine = CinemaRenderer(width=w, height=h, device=device)
    return CinemaSession(
        cine,
        enabled=True,
        mp4_path=Path(args.cinema_mp4) if getattr(args, "cinema_mp4", None) else None,
        frames_dir=Path(args.cinema_dir) if getattr(args, "cinema_dir", None) else None,
        every=int(getattr(args, "cinema_every", 1) or 1),
        keep_pngs=bool(getattr(args, "cinema_keep_pngs", False)),
        fps=float(getattr(args, "fps", 15.0)),
    )
