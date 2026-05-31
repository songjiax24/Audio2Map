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
    """Preprocessed training samples (mel + events + metadata)."""
    return get_data_root() / "processed"


def processed_v2_dir() -> Path:
    """REMI / tick-grid preprocessed samples (v2 pipeline)."""
    return get_data_root() / "processed_v2"


def audio_grid_dir() -> Path:
    """Tick-grid audio feature caches."""
    return processed_v2_dir() / "audio_grid"


def v2_charts_dir() -> Path:
    return processed_v2_dir() / "charts"


def chart_meta_dir() -> Path:
    """Per-chart difficulty metadata (SR, MSD, hold ratio)."""
    return get_data_root() / "chart_meta"
