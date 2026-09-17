"""Official osu! star rating via rosu-pp (lazer difficulty calculator)."""

from __future__ import annotations

from pathlib import Path

import rosu_pp_py as rosu


def official_star_rating(path: Path | str, *, lazer: bool = True) -> float:
    """Return official star rating for a beatmap file."""
    beatmap = rosu.Beatmap(path=str(path))
    attrs = rosu.Difficulty(lazer=lazer).calculate(beatmap)
    return float(attrs.stars)
