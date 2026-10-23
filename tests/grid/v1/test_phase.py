"""phase_at on ordinary beats, residual beats, and a red line."""

from __future__ import annotations

import pytest

from audio2map.grid.v1.beatgrid import build_beatgrid
from audio2map.osu.v1.timing import Redline


def _rl(offset: float, length: float, index: int = 0) -> Redline:
    return Redline(offset, length, 4, index)


def test_phase_on_an_ordinary_beat_and_before_the_first_redline() -> None:
    grid = build_beatgrid([_rl(500.0, 200.0)], t_audio_ms=900.0)
    assert grid.phase_at(-100.0) == (0.0, -3)
    assert grid.phase_at(0.0) == pytest.approx((0.5, -3))
    assert grid.phase_at(500.0) == (0.0, 0)
    phi, beat = grid.phase_at(700.0)
    assert beat == 1
    assert phi == pytest.approx(0.0)


def test_residual_phase_uses_nominal_length_until_the_redline() -> None:
    length = 500.0
    t_red = 1000.0 + 4.25e-5
    grid = build_beatgrid([_rl(0.0, length), _rl(t_red, 400.0, 1)], t_audio_ms=2000.0)
    residual = next(beat for beat in grid.beats if beat.is_residual)
    inside = residual.start_ms + 2.0e-5
    phi, beat = grid.phase_at(inside)
    assert beat == residual.global_b
    assert phi == pytest.approx(2.0e-5 / length)
    assert phi < 1.0
    just_before = t_red - 1e-9
    phi_before, beat_before = grid.phase_at(just_before)
    assert beat_before == residual.global_b
    assert phi_before == pytest.approx((just_before - residual.start_ms) / length)
    phi_on, beat_on = grid.phase_at(t_red)
    nxt = grid.beats[grid.beats.index(residual) + 1]
    assert beat_on == nxt.global_b
    assert phi_on == 0.0
    phi_after, beat_after = grid.phase_at(t_red + 10.0)
    assert beat_after == nxt.global_b
    assert phi_after == pytest.approx(10.0 / 400.0)
