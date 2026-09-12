"""Phase 1 acceptance gate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .download import ROOT

REPORTS = ROOT / "reports"


def check() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    items = []

    def add(name, ok, detail=""):
        items.append({"name": name, "ok": bool(ok), "detail": detail})

    raw = ROOT / "data/raw"
    add("Official MaleCNS data cached", all((raw / f).exists() for f in [
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "body-neurotransmitters-male-cns-v1.0.feather",
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    ]))
    add("Schemas recorded", (REPORTS / "schemas.json").exists())
    add("Oruk-like reservoir ran", (REPORTS / "simulate_oruk499_signed_10000.json").exists())
    full_ok = (REPORTS / "simulate_full_unsigned_10000.json").exists() and (
        REPORTS / "simulate_full_fast_nt_signed_10000.json"
    ).exists()
    add("Full MaleCNS sparse graph ran (both weightings)", full_ok)
    # densify guard: no .npy that is N*N float for full N~1e5
    dense = False
    for p in (ROOT / "data/processed").rglob("*"):
        if p.suffix == ".npy" and "dense" in p.name.lower():
            dense = True
    add("No dense full adjacency artifact", not dense)
    # numerical health from reports
    boom = False
    for name in [
        "simulate_oruk499_signed_10000.json",
        "simulate_full_unsigned_10000.json",
        "simulate_full_fast_nt_signed_10000.json",
    ]:
        path = REPORTS / name
        if path.exists():
            r = json.loads(path.read_text())
            if r.get("nan_inf", 1) != 0 or abs(r.get("state_max", 99)) > 5:
                boom = True
    add("10k steps without numerical explosion", not boom and full_ok)
    vram_ok = False
    if (REPORTS / "simulate_full_unsigned_10000.json").exists():
        r = json.loads((REPORTS / "simulate_full_unsigned_10000.json").read_text())
        peak = r.get("peak_vram_bytes") or 0
        vram_ok = 0 < peak < 12 * 1024**3
        add("GPU memory measured and <12GB", vram_ok, f"peak={peak}")
    else:
        add("GPU memory measured and <12GB", False)
    add("steps/sec benchmarks present", (REPORTS / "benchmarks.json").exists())
    add("Idempotent processed cache", (ROOT / "data/processed/unsigned/metadata.json").exists())
    # unit tests
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    add("Unit tests pass", proc.returncode == 0, proc.stdout.strip()[-200:])
    add("README exists", (ROOT / "README.md").exists())
    add("reports/phase1.md exists", (REPORTS / "phase1.md").exists())

    passed = all(i["ok"] for i in items)
    text = ["# Phase 1 acceptance", ""]
    for i in items:
        mark = "x" if i["ok"] else " "
        text.append(f"- [{mark}] {i['name']}" + (f" — {i['detail']}" if i["detail"] else ""))
    text.append("")
    banner = "PHASE 1: PASS" if passed else "PHASE 1: FAIL"
    text.append(banner)
    (REPORTS / "acceptance.json").write_text(json.dumps({"pass": passed, "items": items}, indent=2))
    print("\n".join(text))
    print()
    print("=" * 20)
    print(banner)
    print("=" * 20)
    return 0 if passed else 1


def main():
    raise SystemExit(check())


if __name__ == "__main__":
    main()
