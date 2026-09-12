"""Discover local Muse tooling paths inside the mindmeld-with-fly monorepo."""
from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # mindmeld-with-fly/


def muse_app_dir() -> Path:
    return REPO_ROOT / "muse" / "neurofeedback-muse"


def start_muse_analyzer() -> Path:
    return muse_app_dir() / "start-muse-analyzer"


def muselsl_bin() -> str | None:
    return shutil.which("muselsl")


def summarize() -> dict:
    return {
        "repo_root": str(REPO_ROOT),
        "muse_app": str(muse_app_dir()),
        "start_muse_analyzer": str(start_muse_analyzer()),
        "start_muse_analyzer_exists": start_muse_analyzer().is_file(),
        "muselsl": muselsl_bin(),
        "muse_viewer": str(muse_app_dir() / "muse_viewer.py"),
    }
