"""Tests for note matching metrics."""

from __future__ import annotations

from audio2map.eval.note_match import compare_note_lists, compare_note_lists_tick_tol
from audio2map.osu.row_tokens import CanonicalTiming
from audio2map.osu.schema import ManiaNote, NoteType


def test_exact_match_f1() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    notes = [
        ManiaNote(time_ms=0, col=0, note_type=NoteType.TAP),
        ManiaNote(time_ms=0, col=1, note_type=NoteType.HOLD, end_time_ms=100),
    ]
    stats = compare_note_lists(notes, notes, timing)
    assert stats.f1 == 1.0
    assert stats.tp == 2


def test_partial_match() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    exp = [ManiaNote(time_ms=0, col=0, note_type=NoteType.TAP)]
    pred = [
        ManiaNote(time_ms=0, col=0, note_type=NoteType.TAP),
        ManiaNote(time_ms=10, col=1, note_type=NoteType.TAP),
    ]
    stats = compare_note_lists(pred, exp, timing)
    assert stats.tp == 1 and stats.fp == 1 and stats.fn == 0
    assert stats.precision == 0.5 and stats.recall == 1.0


def test_tick_tolerance() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    tick_ms = timing.tick_ms
    exp = [ManiaNote(time_ms=0, col=0, note_type=NoteType.TAP)]
    pred = [ManiaNote(time_ms=int(round(tick_ms)), col=0, note_type=NoteType.TAP)]
    exact = compare_note_lists(pred, exp, timing)
    tol = compare_note_lists_tick_tol(pred, exp, timing, tick_tolerance=1)
    assert exact.tp == 0
    assert tol.tp == 1
