"""Tests for 5-state ROW tokenisation and round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio2map.dataset.filter import check_osu_path
from audio2map.osu.parser import parse_beatmap
from audio2map.grid import (
    CanonicalTiming,
    TICKS_PER_BAR,
    TickNote,
    chart_event_bar_range,
    split_tick,
)
from audio2map.tokens import (
    LaneState,
    build_vocab,
    encode_notes,
    is_event_row,
    is_legal_row,
    tokens_to_notes,
)
from audio2map.osu.schema import Beatmap, ChartMetadata, ManiaNote, NoteType, TimingPoint
from audio2map.utils.paths import raw_data_available, raw_dir


def test_vocab_size() -> None:
    assert len(build_vocab()) == 4 + 192 + 625  # specials + POS + ROW


def test_spec_hold_active_example() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    tick_ms = timing.tick_ms
    notes = [
        ManiaNote(
            time_ms=int(round(48 * tick_ms)),
            col=2,
            note_type=NoteType.HOLD,
            end_time_ms=int(round(144 * tick_ms)),
        ),
        ManiaNote(time_ms=int(round(96 * tick_ms)), col=3, note_type=NoteType.TAP),
    ]
    tokens = encode_notes(notes, timing, start_bar=0, end_bar=1)
    assert tokens[0:6] == [
        "<BAR>",
        "<POS_48>",
        "<ROW_0020>",
        "<POS_96>",
        "<ROW_0031>",
        "<POS_144>",
    ]


def test_event_vs_non_event_rows() -> None:
    empty, tap, start, active = (
        LaneState.EMPTY,
        LaneState.TAP,
        LaneState.HOLD_START,
        LaneState.HOLD_ACTIVE,
    )
    assert is_event_row((empty, empty, start, empty))
    assert not is_event_row((empty, active, empty, empty))
    assert is_event_row((tap, empty, empty, empty))


def test_is_legal_row_hold_lane_cannot_be_empty() -> None:
    active = [True, False, False, False]
    empty, tap, hold_active = LaneState.EMPTY, LaneState.TAP, LaneState.HOLD_ACTIVE
    assert is_legal_row((hold_active, tap, empty, empty), active)
    assert not is_legal_row((empty, tap, empty, empty), active)


def test_encode_notes_rejects_tap_during_hold() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    notes = [
        ManiaNote(time_ms=0, col=0, note_type=NoteType.HOLD, end_time_ms=1000),
        ManiaNote(time_ms=500, col=0, note_type=NoteType.TAP),
    ]
    with pytest.raises(ValueError, match="TAP"):
        encode_notes(notes, timing)


def test_split_negative_tick() -> None:
    assert split_tick(-30) == (-1, 162)
    assert split_tick(-1) == (-1, 191)
    assert split_tick(0) == (0, 0)


def test_pre_offset_events_round_trip() -> None:
    """offset is beat origin; notes before offset have negative ticks."""
    timing = CanonicalTiming(980, 180.0, 180.0, 0)
    tick_ms = timing.tick_ms
    notes = [
        ManiaNote(time_ms=950, col=0, note_type=NoteType.TAP),
        ManiaNote(
            time_ms=737,
            col=1,
            note_type=NoteType.HOLD,
            end_time_ms=1200,
        ),
        ManiaNote(time_ms=int(round(980 + 48 * tick_ms)), col=2, note_type=NoteType.TAP),
    ]
    bm = Beatmap(
        path=Path("x.osu"),
        metadata=ChartMetadata("", "", "", "", None, None),
        timing_points=[TimingPoint(980, 60000 / 180, 4, True)],
        notes=notes,
    )
    tokens = encode_notes(notes, timing)
    start_bar, _ = chart_event_bar_range(bm.notes, timing)
    assert TickNote.from_notes(notes, timing) == TickNote.from_notes(
        tokens_to_notes(tokens, timing, start_bar=start_bar).notes, timing
    )


def test_synthetic_round_trip() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    tick_ms = timing.tick_ms
    notes = [
        ManiaNote(time_ms=0, col=0, note_type=NoteType.TAP),
        ManiaNote(time_ms=0, col=3, note_type=NoteType.TAP),
        ManiaNote(time_ms=int(round(48 * tick_ms)), col=2, note_type=NoteType.HOLD, end_time_ms=int(round(144 * tick_ms))),
        ManiaNote(time_ms=int(round(96 * tick_ms)), col=3, note_type=NoteType.TAP),
    ]
    tokens = encode_notes(notes, timing)
    assert tokens and tokens[0] == "<BAR>"
    start_bar, _ = chart_event_bar_range(notes, timing)
    result = tokens_to_notes(tokens, timing, start_bar=start_bar)
    assert TickNote.from_notes(notes, timing) == TickNote.from_notes(result.notes, timing)
    assert result.issues == []


def test_encode_notes_window_range() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    tick_ms = timing.tick_ms
    notes = [
        ManiaNote(time_ms=0, col=0, note_type=NoteType.TAP),
        ManiaNote(time_ms=int(round(TICKS_PER_BAR * tick_ms)), col=1, note_type=NoteType.TAP),
    ]
    full = encode_notes(notes, timing)
    window = encode_notes(notes, timing, start_bar=0, end_bar=1)
    assert "<POS_0>" in window and "<ROW_1000>" in window
    assert window.count("<BAR>") == 1
    assert full.count("<BAR>") >= 2


def test_encode_notes_range_must_be_paired() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    notes = [ManiaNote(time_ms=0, col=0, note_type=NoteType.TAP)]
    with pytest.raises(ValueError, match="both be set"):
        encode_notes(notes, timing, start_bar=0)
    with pytest.raises(ValueError, match="both be set"):
        encode_notes(notes, timing, end_bar=1)


@pytest.mark.requires_data
@pytest.mark.skipif(not raw_data_available(), reason="no dataset")
def test_real_chart_round_trip() -> None:
    path = next(raw_dir().rglob("*.osu"))
    if not check_osu_path(path).eligible:
        pytest.skip("no eligible chart")
    bm = parse_beatmap(path)
    timing = CanonicalTiming.from_beatmap(bm)
    start_bar, _ = chart_event_bar_range(bm.notes, timing)
    assert TickNote.from_notes(bm.notes, timing) == TickNote.from_notes(
        tokens_to_notes(encode_notes(bm.notes, timing), timing, start_bar=start_bar).notes,
        timing,
    )
