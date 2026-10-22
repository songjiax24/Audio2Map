"""BeatGrid coincidence, residual beats, and near-duplicate red lines."""

from __future__ import annotations

import pytest

from audio2map.grid.v1.beatgrid import build_beatgrid
from audio2map.utils.errors import DUPLICATE_REDLINE, ChartRejected
from audio2map.osu.v1.timing import Redline, parse_redlines
from audio2map.utils.tolerance import EPS


def _rl(offset: float, length: float, index: int = 0) -> Redline:
    return Redline(offset, length, 4, index)


def test_forward_extension_and_range() -> None:
    grid = build_beatgrid([_rl(500.0, 200.0)], t_audio_ms=900.0)
    assert grid.coverage_start_ms == -100.0
    assert grid.coverage_end_ms == 900.0
    assert [beat.start_ms for beat in grid.beats] == [-100.0, 100.0, 300.0, 500.0, 700.0]
    assert [beat.global_b for beat in grid.beats] == [-3, -2, -1, 0, 1]
    assert grid.beats[3].beat_k == 0


def test_exact_duplicate_redline_rejects() -> None:
    with pytest.raises(ChartRejected) as caught:
        parse_redlines(["0,500,4,2,0,50,1,0", "0,400,4,2,0,50,1,0"])
    assert caught.value.reason == DUPLICATE_REDLINE


def test_near_duplicate_keeps_later_redline() -> None:
    text = ["0,500,4,2,0,50,1,0", f"{EPS / 2},480,4,2,0,50,1,0"]
    parsed = parse_redlines(text)
    assert len(parsed.near_duplicate_pairs) == 1
    grid = build_beatgrid(parsed.redlines, t_audio_ms=1000.0)
    assert len(grid.near_duplicate_pairs) == 1
    assert all(beat.section == 1 for beat in grid.beats)
    assert grid.beats[0].global_b == 0 or grid.beats[0].start_ms <= 0.0
    origins = [beat for beat in grid.beats if beat.beat_k == 0]
    assert len(origins) == 1
    assert origins[0].global_b == 0
    assert origins[0].start_ms == pytest.approx(EPS / 2)


def test_coincidence_drops_residual() -> None:
    # Distance to the nearest beat is 1e-7 ms, inside EPS.
    length = 500.0
    t_red = 2 * length + 1e-7
    grid = build_beatgrid([_rl(0.0, length), _rl(t_red, length, 1)], t_audio_ms=2000.0)
    residual = [beat for beat in grid.beats if beat.is_residual]
    assert residual == []
    assert [beat.start_ms for beat in grid.beats if beat.section == 0] == [0.0, 500.0]


def test_nokia_scale_residual_is_kept() -> None:
    # Residual about 4.25e-5 ms, greater than EPS, so the short beat stays.
    length = 500.0
    t_red = 1000.0 + 4.25e-5
    grid = build_beatgrid([_rl(0.0, length), _rl(t_red, length, 1)], t_audio_ms=2000.0)
    residual = [beat for beat in grid.beats if beat.is_residual]
    assert len(residual) == 1
    assert residual[0].start_ms == pytest.approx(1000.0)
    assert residual[0].actual_length_ms == pytest.approx(4.25e-5)
    index = residual[0].global_b if False else grid.beats.index(residual[0])
    times = [point.time_ms for point in grid.points[index]]
    assert times == pytest.approx([1000.0])


def test_seychelles_beat_length_is_not_forced_coincidence() -> None:
    length = 165.9999
    # Ten nominal beats of 166 ms land 0.001 ms off this beat length.
    t_red = 10 * 166.0
    grid = build_beatgrid([_rl(0.0, length), _rl(t_red, 200.0, 1)], t_audio_ms=3000.0)
    assert any(beat.is_residual for beat in grid.beats)


def test_residual_grid_points_inside_eps_are_removed() -> None:
    length = 400.0
    t_red = 400.0 + 0.05
    grid = build_beatgrid([_rl(0.0, length), _rl(t_red, length, 1)], t_audio_ms=2000.0)
    residual = next(beat for beat in grid.beats if beat.is_residual)
    times = [point.time_ms for point in grid.points[grid.beats.index(residual)]]
    assert all(time < t_red - EPS for time in times)
    assert 0.0 in [round(time - residual.start_ms, 6) for time in times]
