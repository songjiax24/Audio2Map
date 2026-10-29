"""Batch phase rows match phase_at / bpm_at on the frame lattice."""

from __future__ import annotations

import math

import numpy as np
import pytest

from audio2map.features.audio.v1.constants import DT_MS, HOP, SAMPLE_RATE
from audio2map.features.audio.v1.mel import frame_index
from audio2map.grid.v1.beatgrid import Beat, BeatGrid, build_beatgrid
from audio2map.grid.v1.timefeat import bpm_at, log_bpm_norm, phase_rows, relative_u
from audio2map.utils.time import DT_MS as SHARED_DT_MS
from audio2map.osu.v1.timing import Redline


def test_frame_step_has_one_source() -> None:
    assert DT_MS is SHARED_DT_MS
    assert DT_MS == (2 * HOP) / SAMPLE_RATE * 1000.0


def test_rows_match_the_scalar_apis_across_a_residual() -> None:
    length = 500.0
    t_red = 1000.0 + 4.25e-5
    redlines = (Redline(0.0, length, 4, 0), Redline(t_red, 400.0, 4, 1))
    grid = build_beatgrid(redlines, 2000.0)
    frame_lo = frame_index(grid.coverage_start_ms) - 3
    n_frames = frame_index(grid.coverage_end_ms) - frame_lo + 4
    got = phase_rows(grid, redlines, frame_lo, n_frames)
    expect = _scalar_rows(grid, redlines, frame_lo, n_frames)
    for left, right in zip(got, expect, strict=True):
        assert np.array_equal(left, right)
    assert np.any(got[3] < 0.0)
    assert np.any(got[3] > 1.0)
    stacked = np.stack(got, axis=0).astype(np.float32)
    cast = np.stack([row.astype(np.float32) for row in expect], axis=0)
    assert np.array_equal(stacked, cast)


def test_phi_at_least_one_uses_nextafter_without_calling_phase_at(monkeypatch) -> None:
    beats = (
        Beat(0, 0, -2, 0.0, 0.0, 100.0, 100.0, False, None),
        Beat(0, 1, -1, 250.0, 0.0, 100.0, 100.0, False, None),
    )
    grid = BeatGrid(
        beats=beats,
        points=(),
        coverage_start_ms=0.0,
        coverage_end_ms=400.0,
        t_audio_ms=400.0,
        cells=(),
        near_duplicate_pairs=(),
    )
    redlines = (Redline(0.0, 500.0, 4, 0), Redline(200.0, 20.0, 4, 1))
    frame_lo = -2
    n_frames = 40
    expect = _scalar_rows(grid, redlines, frame_lo, n_frames)

    def _refuse(*_args, **_kwargs):
        raise AssertionError("batch path must not query one frame at a time")

    monkeypatch.setattr(BeatGrid, "phase_at", _refuse)
    monkeypatch.setattr("audio2map.grid.v1.timefeat.bpm_at", _refuse)
    got = phase_rows(grid, redlines, frame_lo, n_frames)
    for left, right in zip(got, expect, strict=True):
        assert np.array_equal(left, right)
    assert math.nextafter(1.0, 0.0) in got[0]
    slow = float(got[2][np.flatnonzero(got[0] == 0.0)[0]])
    assert math.isfinite(slow)


def test_empty_frame_count_is_four_empty_rows() -> None:
    grid = build_beatgrid((Redline(0.0, 500.0, 4, 0),), 1000.0)
    rows = phase_rows(grid, grid_redlines(), 0, 0)
    assert [row.shape for row in rows] == [(0,), (0,), (0,), (0,)]
    assert all(row.dtype == np.float64 for row in rows)


def test_missing_beats_or_bad_beat_length_raises() -> None:
    empty = BeatGrid(
        beats=(),
        points=(),
        coverage_start_ms=0.0,
        coverage_end_ms=1.0,
        t_audio_ms=1.0,
        cells=(),
        near_duplicate_pairs=(),
    )
    with pytest.raises(ValueError, match="no beats"):
        phase_rows(empty, (Redline(0.0, 500.0, 4, 0),), 0, 1)
    grid = build_beatgrid((Redline(0.0, 500.0, 4, 0),), 1000.0)
    with pytest.raises(ValueError, match="positive"):
        phase_rows(grid, (Redline(0.0, 0.0, 4, 0),), 0, 2)


def _scalar_rows(grid, redlines, frame_lo: int, n_frames: int):
    phi = np.empty(n_frames, dtype=np.float64)
    beat_b = np.empty(n_frames, dtype=np.float64)
    log_bpm = np.empty(n_frames, dtype=np.float64)
    rel_u = np.empty(n_frames, dtype=np.float64)
    t0 = float(grid.coverage_start_ms)
    t1 = float(grid.coverage_end_ms)
    for index in range(n_frames):
        t_ms = (frame_lo + index) * DT_MS
        phase, beat = grid.phase_at(t_ms)
        phi[index] = phase
        beat_b[index] = beat
        log_bpm[index] = log_bpm_norm(bpm_at(redlines, t_ms))
        rel_u[index] = relative_u(t_ms, t0, t1)
    return phi, beat_b, log_bpm, rel_u


def grid_redlines():
    return (Redline(0.0, 500.0, 4, 0),)
