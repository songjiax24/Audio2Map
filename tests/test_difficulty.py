"""Tests for chart difficulty computation."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio2map.difficulty.chart_meta import compute_chart_meta
from audio2map.difficulty.hold_ratio import hold_ratio
from audio2map.difficulty.minacalc import compute_msd
from audio2map.difficulty.official_sr import official_star_rating
from audio2map.osu.parser import parse_beatmap
from audio2map.utils.paths import raw_dir


def _sample_osu() -> Path | None:
    root = raw_dir()
    if not root.is_dir():
        return None
    for path in root.rglob("*.osu"):
        return path
    return None


@pytest.fixture
def sample_osu() -> Path:
    path = _sample_osu()
    if path is None:
        pytest.skip("no raw .osu files available")
    return path


def test_official_sr_positive(sample_osu: Path) -> None:
    sr = official_star_rating(sample_osu)
    assert sr >= 0.0


def test_hold_ratio_in_unit_interval(sample_osu: Path) -> None:
    bm = parse_beatmap(sample_osu)
    ratio = hold_ratio(bm)
    assert 0.0 <= ratio <= 1.0


def test_msd_eight_skillsets(sample_osu: Path) -> None:
    bm = parse_beatmap(sample_osu)
    msd = compute_msd(bm)
    assert msd.overall >= 0.0
    assert msd.stream >= 0.0


def test_compute_chart_meta_no_error(sample_osu: Path) -> None:
    meta = compute_chart_meta(sample_osu)
    assert meta.error is None
    assert meta.official_sr is not None
    assert meta.msd_overall is not None
    assert meta.note_count > 0
