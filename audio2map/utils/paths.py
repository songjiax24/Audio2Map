"""Filesystem paths for code vs. data separation."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/audio2map_data")


def get_data_root() -> Path:
    return Path(os.environ.get("AUDIO2MAP_DATA_ROOT", DEFAULT_DATA_ROOT))


def raw_dir() -> Path:
    """Downloaded beatmap sets: ``{sid}/audio.mp3 + *.osu``."""
    return get_data_root() / "raw"


def collector_dir() -> Path:
    """Collector runtime files (state, logs, temp downloads)."""
    return get_data_root() / ".collector"


def processed_dir() -> Path:
    """Archived v1 mel+events cache on the data disk (safe to delete)."""
    return get_data_root() / "processed"


def processed_v2_dir() -> Path:
    """REMI / tick-grid preprocessed samples (v2 pipeline)."""
    return get_data_root() / "processed_v2"


def audio_grid_dir() -> Path:
    """Tick-grid audio feature caches."""
    return processed_v2_dir() / "audio_grid"


def chart_meta_dir() -> Path:
    """Per-chart difficulty metadata (SR, MSD, hold ratio)."""
    return get_data_root() / "chart_meta"


def default_train_checkpoint_dir() -> Path:
    """Default output for ``scripts/train_v2.py`` formal multi-chart runs."""
    return processed_v2_dir() / "checkpoints" / "train_v1"
