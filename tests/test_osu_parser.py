"""Tests for osu! parser and sparse event tokenisation."""

from pathlib import Path

import pytest

from audio2map.osu import beatmap_to_events, event_ar_tokens, events_to_frames, parse_beatmap
from audio2map.osu.events import EventType, notes_to_events
from audio2map.osu.frames import CellState
from audio2map.osu.mania import parse_hit_object, x_to_column
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.utils.paths import raw_dir

RAW = raw_dir()


@pytest.mark.parametrize(
    ("x", "col"),
    [(64, 0), (192, 1), (320, 2), (448, 3)],
)
def test_x_to_column(x: int, col: int) -> None:
    assert x_to_column(x) == col


def test_sparse_events_absolute_not_delta() -> None:
    notes = [
        ManiaNote(time_ms=1000, col=1, note_type=NoteType.TAP),
        ManiaNote(time_ms=5000, col=0, note_type=NoteType.HOLD, end_time_ms=5200),
    ]
    events = notes_to_events(notes, hop_ms=10)
    assert len(events) == 2
    assert events[0].frame == 100
    assert events[1].frame == 500
    assert events[1].end_frame == 520
    assert event_ar_tokens(events[0]) == (100, 1, 0)
    assert event_ar_tokens(events[1]) == (500, 0, 1, 520)


def test_events_shorter_than_dense_grid() -> None:
    notes = [ManiaNote(time_ms=100, col=0, note_type=NoteType.TAP)]
    events = notes_to_events(notes, hop_ms=10)
    grid = events_to_frames(events, duration_ms=10_000)
    assert len(events) == 1
    assert grid.num_frames * 4 > len(events)


def test_parse_real_beatmap() -> None:
    path = RAW / "871715" / "Amane - TWEEKER (FAMoss) [UNPREDICTABLE'S NORMAL].osu"
    if not path.exists():
        pytest.skip("dataset not present")
    bm = parse_beatmap(path)
    events = beatmap_to_events(bm)
    assert len(events) == bm.note_count
    grid = events_to_frames(events, duration_ms=bm.duration_ms + 500)
    assert grid.grid[events[0].frame, events[0].col] != CellState.EMPTY
