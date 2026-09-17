"""Export / encode-decode behaviour locked for Phase-1 (synthetic, no data disk)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio2map.osu.export import (
    filter_notes_for_export,
    note_to_hit_object,
    osu_export_names,
    write_osu,
)
from audio2map.osu.parser import parse_beatmap
from audio2map.grid import CanonicalTiming, TICKS_PER_BAR, tick_to_ms
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.tokens import (
    TOKEN_BAR,
    DecodeIssue,
    encode_notes,
    initial_row_at_bar,
    initial_row_from_active_hold,
    tokens_to_notes,
)


def test_encode_raises_if_hold_collapses_to_same_tick() -> None:
    """Valid hold in ms can snap to end_tick == start_tick; encode does not drop it."""
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    tick_ms = timing.tick_ms
    notes = [
        ManiaNote(
            time_ms=0,
            col=0,
            note_type=NoteType.HOLD,
            end_time_ms=max(1, int(tick_ms * 0.4)),
        ),
        ManiaNote(time_ms=0, col=1, note_type=NoteType.TAP),
    ]
    with pytest.raises(ValueError, match="conflicting lane"):
        encode_notes(notes, timing)


def test_export_filter_drops_negative_and_invalid_holds() -> None:
    # Build a zero-length hold without ManiaNote validation (export sanitize input).
    zero = object.__new__(ManiaNote)
    object.__setattr__(zero, "time_ms", 100)
    object.__setattr__(zero, "col", 2)
    object.__setattr__(zero, "note_type", NoteType.HOLD)
    object.__setattr__(zero, "end_time_ms", 100)

    notes = [
        ManiaNote(time_ms=-10, col=0, note_type=NoteType.TAP),
        ManiaNote(time_ms=0, col=1, note_type=NoteType.TAP),
        zero,
        ManiaNote(time_ms=200, col=3, note_type=NoteType.HOLD, end_time_ms=250),
    ]
    out = filter_notes_for_export(notes)
    assert [(n.time_ms, n.col, n.note_type) for n in out] == [
        (0, 1, NoteType.TAP),
        (200, 3, NoteType.HOLD),
    ]


def test_tokens_to_notes_missing_hold_tail_clips_to_window_end() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    result = tokens_to_notes(
        [TOKEN_BAR, "<POS_0>", "<ROW_2000>"],
        timing,
        start_bar=0,
    )
    end_ms = tick_to_ms(TICKS_PER_BAR, timing)
    assert result.notes == [
        ManiaNote(time_ms=0, col=0, note_type=NoteType.HOLD, end_time_ms=end_ms),
    ]
    assert result.issues == [DecodeIssue("missing_hold_tail", 0, 0)]


def test_tokens_to_notes_missing_hold_head_clips_to_window_start() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    initial = initial_row_from_active_hold([True, False, False, False])
    result = tokens_to_notes(
        [TOKEN_BAR, "<POS_48>", "<ROW_4000>"],
        timing,
        start_bar=0,
        initial_row=initial,
    )
    end_ms = tick_to_ms(48, timing)
    assert result.notes == [
        ManiaNote(time_ms=0, col=0, note_type=NoteType.HOLD, end_time_ms=end_ms),
    ]
    assert result.issues == [DecodeIssue("missing_hold_head", 0, 0)]


def test_tokens_to_notes_hold_end_without_hold_raises() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    with pytest.raises(ValueError, match="illegal HOLD_END"):
        tokens_to_notes(
            [TOKEN_BAR, "<POS_0>", "<ROW_4000>"],
            timing,
            start_bar=0,
        )


def test_tokens_to_notes_hold_spanning_window() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    initial = initial_row_from_active_hold([True, False, False, False])
    result = tokens_to_notes(
        [TOKEN_BAR],
        timing,
        start_bar=0,
        initial_row=initial,
    )
    end_ms = tick_to_ms(TICKS_PER_BAR, timing)
    assert result.notes == [
        ManiaNote(time_ms=0, col=0, note_type=NoteType.HOLD, end_time_ms=end_ms),
    ]
    assert result.issues == [
        DecodeIssue("missing_hold_head", 0, 0),
        DecodeIssue("missing_hold_tail", 0, 0),
    ]


def test_tokens_to_notes_hold_active_without_hold_raises() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    with pytest.raises(ValueError, match="illegal HOLD_ACTIVE"):
        tokens_to_notes(
            [TOKEN_BAR, "<POS_0>", "<ROW_1300>"],
            timing,
            start_bar=0,
        )


def test_tokens_to_notes_empty_while_holding_raises() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    tokens = [
        TOKEN_BAR,
        "<POS_0>",
        "<ROW_2000>",
        "<POS_48>",
        "<ROW_0100>",
    ]
    with pytest.raises(ValueError, match="illegal EMPTY"):
        tokens_to_notes(tokens, timing, start_bar=0)


def test_tokens_to_notes_raises_on_tap_during_hold() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    tokens = [
        TOKEN_BAR,
        "<POS_0>",
        "<ROW_2000>",
        "<POS_48>",
        "<ROW_1000>",
    ]
    with pytest.raises(ValueError, match="illegal TAP"):
        tokens_to_notes(tokens, timing, start_bar=0)


def test_tokens_to_notes_pos_before_bar_raises() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    with pytest.raises(ValueError, match="POS before BAR"):
        tokens_to_notes(
            ["<POS_0>", "<ROW_1000>"],
            timing,
            start_bar=0,
        )


def test_tokens_to_notes_duplicate_pos_in_bar_raises() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    with pytest.raises(ValueError, match="duplicate POS"):
        tokens_to_notes(
            [TOKEN_BAR, "<POS_48>", "<ROW_1000>", "<POS_48>", "<ROW_0100>"],
            timing,
            start_bar=0,
        )


def test_tokens_to_notes_hold_crossing_window_start_clips() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    tick_ms = timing.tick_ms
    notes = [
        ManiaNote(
            time_ms=0,
            col=0,
            note_type=NoteType.HOLD,
            end_time_ms=int(round((TICKS_PER_BAR + 48) * tick_ms)),
        ),
    ]
    tokens = encode_notes(notes, timing, start_bar=1, end_bar=2)
    initial = initial_row_at_bar(notes, timing, start_bar=1)
    result = tokens_to_notes(tokens, timing, start_bar=1, initial_row=initial)
    start_ms = tick_to_ms(TICKS_PER_BAR, timing)
    end_ms = tick_to_ms(TICKS_PER_BAR + 48, timing)
    assert result.notes == [
        ManiaNote(time_ms=start_ms, col=0, note_type=NoteType.HOLD, end_time_ms=end_ms),
    ]
    assert result.issues == [DecodeIssue("missing_hold_head", TICKS_PER_BAR, 0)]


def test_tokens_to_notes_hold_crossing_window_end_clips() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    tick_ms = timing.tick_ms
    notes = [
        ManiaNote(
            time_ms=0,
            col=0,
            note_type=NoteType.HOLD,
            end_time_ms=int(round((TICKS_PER_BAR + 48) * tick_ms)),
        ),
    ]
    tokens = encode_notes(notes, timing, start_bar=0, end_bar=1)
    result = tokens_to_notes(tokens, timing, start_bar=0)
    end_ms = tick_to_ms(TICKS_PER_BAR, timing)
    assert result.notes == [
        ManiaNote(time_ms=0, col=0, note_type=NoteType.HOLD, end_time_ms=end_ms),
    ]
    assert result.issues == [DecodeIssue("missing_hold_tail", 0, 0)]


def test_note_to_hit_object_formats() -> None:
    tap = ManiaNote(time_ms=1000, col=0, note_type=NoteType.TAP)
    hold = ManiaNote(time_ms=1000, col=2, note_type=NoteType.HOLD, end_time_ms=1500)
    assert note_to_hit_object(tap) == "64,192,1000,1,0,0:0:0:0:"
    assert note_to_hit_object(hold) == "320,192,1000,128,0,1500:0:0:0:0:"


def test_write_osu_from_source_replaces_hitobjects(tmp_path: Path) -> None:
    source = tmp_path / "source.osu"
    source.write_text(
        "\n".join(
            [
                "osu file format v14",
                "[General]",
                "AudioFilename: audio.mp3",
                "Mode:0",
                "[Metadata]",
                "Title:Test",
                "Artist:A",
                "Creator:C",
                "Version:Normal",
                "[Difficulty]",
                "CircleSize:5",
                "[Events]",
                "// background",
                "0,0,\"bg.jpg\",0,0",
                "[TimingPoints]",
                "0,500,4,2,0,100,1,0",
                "[HitObjects]",
                "256,192,980,1,0,0:0:0:0:",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.osu"
    write_osu(
        out,
        [
            ManiaNote(time_ms=-5, col=0, note_type=NoteType.TAP),
            ManiaNote(time_ms=480, col=1, note_type=NoteType.TAP),
        ],
        source_osu=source,
        version_suffix=" (Audio2Map)",
    )
    text = out.read_text(encoding="utf-8")
    assert "Mode:3" in text
    assert "CircleSize:4" in text
    assert "Version:Normal (Audio2Map)" in text
    assert "bg.jpg" not in text
    assert "256,192,980" not in text
    assert text.count("[HitObjects]") == 1
    hit_section = text.split("[HitObjects]")[1]
    assert "480" in hit_section
    assert "-5" not in hit_section
    assert "192,192,480,1,0,0:0:0:0:" in text
    beatmap = parse_beatmap(out)
    assert len(beatmap.notes) == 1
    assert beatmap.notes[0].time_ms == 480


def test_write_osu_from_scratch(tmp_path: Path) -> None:
    out = write_osu(
        tmp_path / "gen.osu",
        [ManiaNote(time_ms=200, col=0, note_type=NoteType.TAP)],
        title="T",
        artist="A",
        creator="C",
        version="V",
        offset_ms=100,
        original_bpm=150.0,
    )
    text = out.read_text(encoding="utf-8")
    assert "Countdown: 0" in text
    assert "SampleSet: Soft" in text
    assert "HPDrainRate:8" in text
    assert "OverallDifficulty:8" in text
    assert "WidescreenStoryboard: 0" in text
    assert "BeatmapID:0" in text
    beatmap = parse_beatmap(out)
    assert len(beatmap.notes) == 1
    timing = CanonicalTiming.from_beatmap(beatmap)
    assert timing.offset_ms == 100
    assert timing.original_bpm == pytest.approx(150.0)
    assert beatmap.metadata.version == "V"


def test_write_osu_requires_source_or_bpm(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_osu or original_bpm"):
        write_osu(tmp_path / "x.osu", [])


def test_osu_export_names_follow_editor_convention() -> None:
    names = osu_export_names(
        artist="Camellia",
        title="Crystallized",
        creator="Audio2Map",
        version="Generated",
        audio_filename="drop.MP3",
    )
    assert names.osu == "Camellia - Crystallized (Audio2Map) [Generated].osu"
    assert names.osz == "Camellia - Crystallized.osz"
    assert names.audio == "drop.mp3"


def test_osu_export_names_strip_illegal_and_empty() -> None:
    names = osu_export_names(
        artist='A<B>/C:"D"',
        title="  ",
        creator="",
        version="Hard*",
        audio_filename="nested/foo:bar.WAV",
    )
    assert names.osu == "A B C D - Untitled (Audio2Map) [Hard].osu"
    assert names.audio == "foo bar.wav"
    assert "/" not in names.osu
    assert ":" not in names.osu
    assert ":" not in names.audio
