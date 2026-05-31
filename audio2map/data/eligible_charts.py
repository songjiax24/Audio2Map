"""List eligible Phase-1 charts under raw data."""

from __future__ import annotations

from pathlib import Path

from audio2map.data.chart_filter import check_osu_path
from audio2map.utils.paths import raw_dir


def list_eligible_osu_paths(root: Path | None = None) -> list[Path]:
    root = raw_dir() if root is None else Path(root)
    out: list[Path] = []
    for path in sorted(root.rglob("*.osu")):
        if check_osu_path(path).eligible:
            out.append(path)
    return out
