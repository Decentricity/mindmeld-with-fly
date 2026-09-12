"""CLI: python -m mindmeld.living --renderer caca --graph full"""
from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

from ..download import ROOT
from .cinema import make_cinema_from_args
from .engine import FrameState, LivingEngine
from .project import commit_trail
from .record import Recorder, load_recording
from .render import connectome_banner, make_renderer
from .voice import announce, announce_camera, announce_stim, announce_view


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Living MaleCNS terminal visualization (Phase 1.5B)")
    p.add_argument("--renderer", default="caca", choices=["caca", "ansi"])
    p.add_argument("--graph", default="full", choices=["full", "oruk499"])
    p.add_argument("--weighting", default="unsigned", choices=["unsigned", "fast_nt_signed"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--seconds", type=float, default=0.0, help="Auto-quit after N seconds (0=interactive)")
    p.add_argument("--record", type=Path, default=None, help="Write recording to this npz path")
    p.add_argument("--replay", type=Path, default=None, help="Replay a recording (no GPU sim)")
    p.add_argument("--demo", action="store_true", help="Non-interactive demo cycling modes/views")
    p.add_argument(
        "--mute",
        action="store_true",
        help="Disable say-alert toggle announcements (non-GPU RHVoice)",
    )
    p.add_argument("--no-cinema", action="store_true", help="Record NPZ only (disable default MP4 cinema)")
    p.add_argument("--cinema-mp4", type=Path, default=None, help="Override cinema MP4 output path")
    p.add_argument("--cinema-dir", type=Path, default=None, help="Override cinema PNG staging directory")
    p.add_argument("--cinema-size", default="1280x720", help="Cinema resolution WxH (default 1280x720)")
    p.add_argument("--cinema-every", type=int, default=1, help="Write every Nth frame while recording")
    p.add_argument("--cinema-keep-pngs", action="store_true", help="Keep staging PNGs after MP4 encode")
    return p.parse_args(argv)


class RawTTY:
    def __init__(self, fd):
        self.fd = fd
        self.old = None

    def __enter__(self):
        if not sys.stdin.isatty():
            return self
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll_key(self) -> str | None:
        if not sys.stdin.isatty():
            return None
        r, _, _ = select.select([self.fd], [], [], 0)
        if not r:
            return None
        return sys.stdin.read(1)


def _start_record(recorder: Recorder, cinema, voice_on: bool) -> None:
    recorder.start()
    if cinema is not None:
        cinema.start()
    announce("Recording started.", enabled=voice_on)


def _stop_record(recorder: Recorder, cinema, engine: LivingEngine, voice_on: bool, meta: dict) -> None:
    recorder.stop_and_save(
        engine.viz["xy"],
        engine.edges,
        meta,
        xyz=engine.viz.get("xyz"),
    )
    mp4 = None
    if cinema is not None:
        mp4 = cinema.stop_and_encode()
    msg = "Recording saved."
    if mp4 is not None:
        msg = f"Recording saved, cinema {mp4.name}."
        print(f"Saved recording → {recorder.path}", file=sys.stderr)
        print(f"Saved cinema   → {mp4}", file=sys.stderr)
    else:
        print(f"\nSaved recording → {recorder.path}", file=sys.stderr)
    announce(msg, enabled=voice_on)


def run_replay(path: Path, renderer_name: str, fps: float, seconds: float):
    data = load_recording(path)
    acts = data["activity"]
    renderer = make_renderer(renderer_name)
    renderer.begin()

    class Stub:
        graph = "replay"
        weighting = "replay"

    stub = Stub()
    xy = np.asarray(data["xy"], dtype=np.float32)
    if "xyz" in data:
        xyz = np.asarray(data["xyz"], dtype=np.float32)
    else:
        xyz = np.concatenate([xy, np.full((len(xy), 1), 0.5, dtype=np.float32)], axis=1)
    stub.viz = {
        "xy": xy,
        "xyz": xyz,
        "region_id": np.zeros(len(xy), dtype=np.int16),
        "region_names": ["replay"],
        "n": len(xy),
    }
    stub.edges = np.asarray(data["edges"], dtype=np.int32)
    stub.prev_activity = np.zeros(len(xy), dtype=np.float32)
    stub.camera = "triad"
    stub.yaw = 0.35
    stub.pitch = 0.25
    stub.view = "whole"

    t0 = time.time()
    frame_i = 0
    try:
        while True:
            if seconds and (time.time() - t0) >= seconds:
                break
            i = frame_i % len(acts)
            act = np.asarray(acts[i], dtype=np.float32)
            frame = FrameState(
                step=i,
                activity=act,
                mean=float(act.mean()),
                active=int((np.abs(act) > 0.05).sum()),
                steps_per_sec=0.0,
                vram_bytes=None,
                mode="replay",
                view="whole",
                intensity=0.0,
                paused=False,
            )
            loop_t0 = time.time()
            renderer.draw(stub, frame, fps)
            frame_i += 1
            time.sleep(max(0.0, 1.0 / fps - (time.time() - loop_t0)))
    finally:
        renderer.end()


def main(argv=None):
    args = _parse_args(argv)
    if args.replay:
        run_replay(args.replay, args.renderer, args.fps, args.seconds or 8.0)
        return

    engine = LivingEngine(graph=args.graph, weighting=args.weighting, device=args.device)
    renderer = make_renderer(args.renderer)
    rec_path = args.record or (ROOT / "recordings" / "living_demo.npz")
    recorder = Recorder(rec_path)
    cinema = make_cinema_from_args(args)
    auto_record = bool(args.demo or args.record)
    voice_on = not args.mute and not args.demo

    renderer.begin()
    t_start = time.time()
    fps_ema = float(args.fps)
    demo_phase = 0
    demo_t0 = time.time()
    fps_samples: list[float] = []
    if voice_on:
        announce(
            "Living MaleCNS visualization. "
            + connectome_banner(engine).replace("CONNECTOME:", "Connectome:").replace("♂", "male"),
            enabled=True,
        )
        announce_stim(engine.stim.mode, engine.stim.intensity, enabled=True)
        announce_camera(engine.camera, enabled=True)
    try:
        with RawTTY(sys.stdin.fileno()) as tty_in:
            if auto_record:
                _start_record(recorder, cinema, voice_on=False)
            while True:
                now = time.time()
                if args.seconds and (now - t_start) >= args.seconds:
                    break
                if args.demo and (now - demo_t0) > 2.0:
                    demo_t0 = now
                    demo_phase += 1
                    if demo_phase % 2 == 0:
                        engine.stim.cycle_mode(1)
                    else:
                        engine.cycle_view(1)
                    if demo_phase % 3 == 0:
                        engine.toggle_camera()
                    if demo_phase % 4 == 0:
                        engine.stim.trigger_impulse()
                    if engine.camera == "orbit":
                        engine.nudge_yaw(engine.YAW_STEP * 2)

                key = tty_in.poll_key()
                if key in ("q", "Q", "\x1b"):
                    if voice_on:
                        announce("Quitting living visualization.", enabled=True)
                    break
                if key == " ":
                    engine.paused = not engine.paused
                    announce("Paused." if engine.paused else "Live.", enabled=voice_on)
                elif key == "\t":
                    engine.toggle_camera()
                    announce_camera(engine.camera, enabled=voice_on)
                elif key == "n":
                    engine.stim.cycle_mode(1)
                    announce_stim(engine.stim.mode, engine.stim.intensity, enabled=voice_on)
                elif key == "v":
                    engine.cycle_view(1)
                    announce_view(engine.view, enabled=voice_on)
                elif key == "[":
                    engine.stim.intensity = max(0.0, engine.stim.intensity - 0.05)
                    announce(f"Intensity {engine.stim.intensity:.2f}.", enabled=voice_on)
                elif key == "]":
                    engine.stim.intensity = min(2.0, engine.stim.intensity + 0.05)
                    announce(f"Intensity {engine.stim.intensity:.2f}.", enabled=voice_on)
                elif key in ("+", "="):
                    engine.speed = min(16, engine.speed + 1)
                    announce(f"Sim speed {engine.speed}.", enabled=voice_on)
                elif key == "-":
                    engine.speed = max(1, engine.speed - 1)
                    announce(f"Sim speed {engine.speed}.", enabled=voice_on)
                elif key == ",":
                    engine.nudge_yaw(-engine.YAW_STEP)
                elif key == ".":
                    engine.nudge_yaw(engine.YAW_STEP)
                elif key == "j":
                    engine.nudge_pitch(-engine.PITCH_STEP)
                elif key == "k":
                    engine.nudge_pitch(engine.PITCH_STEP)
                elif key == "i":
                    engine.stim.trigger_impulse()
                    announce("Impulse burst.", enabled=voice_on)
                elif key == "r":
                    engine.reset()
                    announce("State reset.", enabled=voice_on)
                elif key == "R":
                    if not recorder.active:
                        _start_record(recorder, cinema, voice_on)
                    else:
                        _stop_record(
                            recorder,
                            cinema,
                            engine,
                            voice_on,
                            {"graph": engine.graph, "weighting": engine.weighting},
                        )

                loop_t0 = time.time()
                frame = engine.tick()
                if cinema is not None and cinema.active:
                    cinema.push(engine, frame, fps_ema)
                renderer.draw(engine, frame, fps_ema, commit=False)
                commit_trail(engine, frame)
                if recorder.active:
                    recorder.push(frame.activity, {"step": frame.step, "view": frame.view, "mode": frame.mode})
                dt = time.time() - loop_t0
                inst_fps = 1.0 / dt if dt > 0 else args.fps
                fps_ema = 0.85 * fps_ema + 0.15 * inst_fps
                fps_samples.append(inst_fps)
                sleep = (1.0 / args.fps) - (time.time() - loop_t0)
                if sleep > 0:
                    time.sleep(sleep)
    finally:
        if recorder.active and recorder.frames:
            _stop_record(
                recorder,
                cinema,
                engine,
                voice_on=False,
                meta={"graph": args.graph, "weighting": args.weighting},
            )
        renderer.end()
        if fps_samples:
            arr = np.asarray(fps_samples, dtype=np.float64)
            print(
                f"FPS stats: mean={arr.mean():.1f}  p10={np.percentile(arr,10):.1f}  "
                f"p50={np.percentile(arr,50):.1f}  n={len(arr)}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
