"""Choose which .osu chart to use per beatmap set."""

from __future__ import annotations

from pathlib import Path

from audio2map.osu.parser import parse_beatmap


def select_highest_od(osu_paths: list[Path]) -> Path:
    """Return the mania 4K ``.osu`` with the highest ``OverallDifficulty``."""
    if not osu_paths:
        raise FileNotFoundError("no .osu files to select from")

    best_path = osu_paths[0]
    best_od = -1.0
    for path in osu_paths:
        bm = parse_beatmap(path)
        od = bm.metadata.overall_difficulty
        if od > best_od:
            best_od = od
            best_path = path
    return best_path
