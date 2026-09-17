"""Demo backend configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path

from audio2map.utils.paths import chart_meta_dir, formal_checkpoint_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = REPO_ROOT / "demo"
BACKEND_ROOT = DEMO_ROOT / "backend"

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoint" / "step_200000.pt"
DEFAULT_MANIFEST = REPO_ROOT / "chart_meta" / "manifest.jsonl"
FORMAL_CHECKPOINT_FILE = "step_200000.pt"
OUTPUT_DIR = BACKEND_ROOT / "outputs"
UPLOAD_DIR = BACKEND_ROOT / "uploads"

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav"}

_STEP_PT = re.compile(r"^step_(\d+)\.pt$")


def get_device() -> str:
    return os.environ.get("AUDIO2MAP_DEVICE", "cuda").strip() or "cuda"


def resolve_device_name() -> str:
    """Honor ``AUDIO2MAP_DEVICE=cpu`` even if CUDA is installed."""
    requested = get_device().lower()
    if requested == "cpu":
        return "cpu"
    if requested in {"cuda", "gpu"}:
        return "cuda"
    return requested


def _latest_step_pt(directory: Path) -> Path | None:
    best: Path | None = None
    best_n = -1
    if not directory.is_dir():
        return None
    for path in directory.glob("step_*.pt"):
        match = _STEP_PT.match(path.name)
        if match is None:
            continue
        n = int(match.group(1))
        if n > best_n:
            best_n, best = n, path
    return best


def get_checkpoint_path() -> Path:
    env = os.environ.get("AUDIO2MAP_CHECKPOINT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    formal_dir = formal_checkpoint_dir()
    exact = formal_dir / FORMAL_CHECKPOINT_FILE
    if exact.is_file():
        return exact
    latest = _latest_step_pt(formal_dir)
    if latest is not None:
        return latest
    if DEFAULT_CHECKPOINT.is_file():
        return DEFAULT_CHECKPOINT
    return exact


def get_manifest_path() -> Path:
    env = os.environ.get("AUDIO2MAP_CHART_META_MANIFEST", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    data = chart_meta_dir() / "manifest.jsonl"
    if data.is_file():
        return data
    return DEFAULT_MANIFEST
