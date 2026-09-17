"""Filesystem paths for code vs. data separation.

Set ``AUDIO2MAP_DATA_ROOT`` to the data disk. If unset, uses repo-local
``.local-data/`` (gitignored) and emits a warning.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_LOCAL_ROOT = Path(__file__).resolve().parents[2] / ".local-data"
_WARNED_UNSET = False

FORMAL_CHECKPOINT_NAME = "formal_enc_dec_v3"


def _path_is_dir(path: Path) -> bool:
    """``Path.is_dir()`` that treats permission errors as missing."""
    try:
        return path.is_dir()
    except OSError:
        return False


def get_data_root() -> Path:
    """Return the data root, preferring ``AUDIO2MAP_DATA_ROOT``."""
    global _WARNED_UNSET
    env = os.environ.get("AUDIO2MAP_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if not _WARNED_UNSET:
        _WARNED_UNSET = True
        logger.warning(
            "AUDIO2MAP_DATA_ROOT is unset; using %s. "
            "Set AUDIO2MAP_DATA_ROOT to your data disk (raw/, processed/, chart_meta/).",
            _REPO_LOCAL_ROOT,
        )
    return _REPO_LOCAL_ROOT


def raw_dir() -> Path:
    """Downloaded beatmap sets: ``{sid}/*.osu`` plus referenced audio."""
    return get_data_root() / "raw"


def collector_dir() -> Path:
    """Collector runtime files (state, logs, temp downloads)."""
    return get_data_root() / ".collector"


def processed_dir() -> Path:
    """Tick-grid caches, checkpoints, and generated outputs."""
    return get_data_root() / "processed"


def audio_grid_dir() -> Path:
    """Tick-grid audio feature caches."""
    return processed_dir() / "audio_grid"


def chart_meta_dir() -> Path:
    """Per-chart difficulty metadata (SR, MSD, pattern stats)."""
    return get_data_root() / "chart_meta"


def formal_checkpoint_dir() -> Path:
    """Default formal training checkpoint directory."""
    return processed_dir() / "checkpoints" / FORMAL_CHECKPOINT_NAME


def raw_data_available() -> bool:
    """True if ``raw/`` exists and is readable (safe under permission errors)."""
    return _path_is_dir(raw_dir())
