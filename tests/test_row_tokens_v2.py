"""Tests for v2 5-state ROW tokenisation and round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio2map.data.chart_filter import check_parsed_beatmap
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.row_tokens import (
    TOKEN_BOS,
    TOKEN_EOS,
    CanonicalTiming,
    LaneState,
    beatmap_to_row_tokens,
    build_vocab,
    chart_event_bar_range,
    encode_window_tokens,
    is_event_row,
    is_initial_row,
    is_legal_row,
    row_state_to_token,
    split_tick,
    validate_token_sequence,
)
from audio2map.osu.round_trip import notes_equal, quantize_notes, round_trip_beatmap, tokens_to_notes
from audio2map.osu.schema import Beatmap, ChartMetadata, ManiaNote, NoteType, TimingPoint
from audio2map.utils.paths import raw_dir


def test_vocab_size() -> None:
    assert len(build_vocab()) == 4 + 192 + 625  # specials + POS + ROW


def test_spec_hold_active_example() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    tokens = encode_window_tokens(
        {
            0: {
                48: (0, 0, 2, 0),
                96: (0, 0, 3, 1),
                144: (0, 0, 4, 0),
            }
        },
        start_bar=0,
        end_bar=1,
        initial_row=(0, 0, 0, 0),
    )
    assert tokens[2:8] == [
        "<BAR>",
        "<POS_48>",
        "<ROW_0020>",
        "<POS_96>",
        "<ROW_0031>",
        "<POS_144>",
    ]


def test_initial_vs_event_rows() -> None:
    assert is_initial_row((0, 3, 0, 0))
    assert not is_initial_row((1, 0, 0, 0))
    assert is_event_row((0, 0, 2, 0))
    assert not is_event_row((0, 3, 0, 0))


def test_is_legal_row_hold_lane_cannot_be_empty() -> None:
    active = [True, False, False, False]
    assert is_legal_row((3, 1, 0, 0), active)
    assert not is_legal_row((0, 1, 0, 0), active)


def test_split_negative_tick() -> None:
    assert split_tick(-30) == (-1, 162)
    assert split_tick(-1) == (-1, 191)
    assert split_tick(0) == (0, 0)


def test_pre_offset_events_round_trip() -> None:
    """offset is beat origin; notes before offset have negative absolute_tick."""
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
        metadata=ChartMetadata("", "", "", "", None, None, 0, 4, 0, 0, 1, ""),
        timing_points=[TimingPoint(980, 60000 / 180, 4, True)],
        notes=notes,
    )
    tokens = beatmap_to_row_tokens(bm, timing)
    assert validate_token_sequence(tokens) == []
    start_bar, _ = chart_event_bar_range(bm.notes, timing)
    assert notes_equal(
        quantize_notes(notes, timing),
        tokens_to_notes(tokens, timing, start_bar=start_bar),
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
    bm = Beatmap(
        path=Path("x.osu"),
        metadata=ChartMetadata("", "", "", "", None, None, 0, 4, 0, 0, 1, ""),
        timing_points=[TimingPoint(0, 500.0, 4, True)],
        notes=notes,
    )
    tokens = beatmap_to_row_tokens(bm, timing)
    assert tokens[0] == TOKEN_BOS and tokens[-1] == TOKEN_EOS
    assert validate_token_sequence(tokens) == []
    start_bar, _ = chart_event_bar_range(bm.notes, timing)
    assert notes_equal(
        quantize_notes(notes, timing),
        tokens_to_notes(tokens, timing, start_bar=start_bar),
    )


@pytest.mark.skipif(not raw_dir().is_dir(), reason="no dataset")
def test_real_chart_round_trip() -> None:
    path = next(raw_dir().rglob("*.osu"))
    if not check_parsed_beatmap(path).eligible:
        pytest.skip("no eligible chart")
    bm = parse_beatmap(path)
    assert round_trip_beatmap(bm)


def test_pattern_analyser() -> None:
    from audio2map.pattern_analyser import analyze_osu

    path = next(raw_dir().rglob("*.osu"), None)
    if path is None:
        pytest.skip("no dataset")
    feat = analyze_osu(path)
    assert feat.analyzer_available
    assert abs(sum(feat.core_dist.values()) - 1.0) < 1e-6 or sum(feat.core_dist.values()) == 0
