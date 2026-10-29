"""Official osu! star rating via rosu-pp (lazer difficulty calculator)."""

from __future__ import annotations

import math
from pathlib import Path

import rosu_pp_py as rosu


def official_star_rating(
    path: Path | str,
    *,
    lazer: bool = True,
    clock_rate: float = 1.0,
) -> float:
    """Official star rating. ``clock_rate`` is rosu's DT/HT clock, not ``stars * rate``."""
    if not math.isfinite(clock_rate) or clock_rate <= 0.0:
        raise ValueError(f"clock_rate must be positive and finite, got {clock_rate}")
    beatmap = rosu.Beatmap(path=str(path))
    difficulty = rosu.Difficulty(lazer=lazer)
    if clock_rate != 1.0:
        difficulty.set_clock_rate(clock_rate)
    attrs = difficulty.calculate(beatmap)
    return float(attrs.stars)
