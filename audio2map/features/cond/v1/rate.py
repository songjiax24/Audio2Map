"""Difficulty scalars at a playback clock. Pattern amounts stay at rate 1.

``official_sr`` uses rosu ``set_clock_rate``. MSD uses MinaCalc ``music_rate``.
Rework uses ``speed_rate``. None of these multiply a star rating by ``rate``.
The 18-d condition vector does not gain a rate dimension.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from audio2map.features.cond.chart import ChartMeta
from audio2map.features.cond.minacalc import compute_msd
from audio2map.features.cond.official_sr import official_star_rating
from audio2map.features.cond.v1.rework import rework_star_rating
from audio2map.osu.parser import parse_beatmap


def scalars_at_rate(path: Path | str, base: ChartMeta, rate: float) -> ChartMeta:
    """Copy ``base`` and replace SR, Rework, and MSD when ``rate != 1``."""
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"rate must be positive and finite, got {rate}")
    if rate == 1.0:
        return base
    path = Path(path)
    beatmap = parse_beatmap(path)
    stars = official_star_rating(path, clock_rate=rate)
    rework = rework_star_rating(path, speed_rate=rate)
    msd = compute_msd(beatmap, music_rate=rate)
    return replace(
        base,
        official_sr=stars,
        rework_sr=rework,
        msd_overall=msd.overall,
        msd_stream=msd.stream,
        msd_jumpstream=msd.jumpstream,
        msd_handstream=msd.handstream,
        msd_stamina=msd.stamina,
        msd_jack_speed=msd.jack_speed,
        msd_chordjack=msd.chordjack,
        msd_technical=msd.technical,
    )
