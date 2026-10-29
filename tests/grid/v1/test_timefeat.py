"""log BPM and u stay on the chart clock. u is not clamped."""

from __future__ import annotations

import math

import pytest

from audio2map.features.audio.v1.constants import DT_MS
from audio2map.grid.v1.timefeat import (
    LOG_BPM_CENTER,
    LOG_BPM_SCALE,
    bpm_at,
    bpm_from_segments,
    log_bpm_norm,
    relative_u,
    segment_durations,
)
from audio2map.osu.v1.timing import Redline


def test_log_bpm_affine_is_centered_at_180_with_scale_one_half() -> None:
    assert LOG_BPM_CENTER == pytest.approx(math.log(180.0))
    assert LOG_BPM_SCALE == 0.5
    assert log_bpm_norm(180.0) == pytest.approx(0.0)
    assert log_bpm_norm(180.0 * math.exp(0.5)) == pytest.approx(1.0)
    assert log_bpm_norm(360.0) - log_bpm_norm(180.0) == pytest.approx(math.log(2.0) / 0.5)


def test_extreme_bpm_is_kept() -> None:
    slow = log_bpm_norm(20.0)
    fast = log_bpm_norm(1000.0)
    assert math.isfinite(slow) and math.isfinite(fast)
    assert slow < -1.0
    assert fast > 1.0


def test_log_bpm_changes_across_a_redline() -> None:
    offsets = [0.0, 1000.0]
    beats = [500.0, 250.0]
    before = log_bpm_norm(bpm_from_segments(offsets, beats, 999.0))
    at_boundary = log_bpm_norm(bpm_from_segments(offsets, beats, 1000.0))
    assert at_boundary - before == pytest.approx(math.log(2.0) / LOG_BPM_SCALE)
    assert at_boundary != pytest.approx(before)


def test_u_is_outside_on_either_side_of_coverage() -> None:
    t0, t1 = 0.0, 100.0
    assert relative_u(t0 - DT_MS, t0, t1) < 0.0
    assert relative_u(t1 + DT_MS, t0, t1) > 1.0
    assert relative_u(50.0, t0, t1) == pytest.approx(0.5)


def test_segment_weight_starts_at_zero_and_stops_at_audio_end() -> None:
    weights = segment_durations([500.0, 1500.0, 4000.0], 3000.0)
    assert weights == pytest.approx([1500.0, 1500.0, 0.0])


def test_bpm_before_the_first_offset_uses_that_redline() -> None:
    lines = (Redline(100.0, 500.0, 4, 0),)
    assert bpm_at(lines, 0.0) == pytest.approx(120.0)


def test_bpm_at_reuses_redline_columns() -> None:
    from audio2map.grid.v1.timefeat import redline_columns

    lines = (Redline(0.0, 500.0, 4, 0), Redline(1000.0, 250.0, 4, 1))
    bpm_at(lines, 0.0)
    offsets, lengths = redline_columns(lines)
    assert bpm_at(lines, 1500.0) == pytest.approx(240.0)
    assert redline_columns(lines)[0] is offsets
    assert redline_columns(lines)[1] is lengths


def test_log_bpm_and_u_matches_the_scalar_loop() -> None:
    import numpy as np

    from audio2map.grid.v1.timefeat import bpm_from_segments, log_bpm_and_u

    offsets = np.array([0.0, 1000.0])
    beats = np.array([500.0, 250.0])
    times = np.array([-20.0, 0.0, 999.0, 1000.0, 2500.0])
    log_bpm, rel_u = log_bpm_and_u(offsets, beats, times, 0.0, 2000.0)
    for index, t_ms in enumerate(times):
        expect = log_bpm_norm(bpm_from_segments(offsets.tolist(), beats.tolist(), float(t_ms)))
        assert log_bpm[index] == expect
        assert rel_u[index] == relative_u(float(t_ms), 0.0, 2000.0)
