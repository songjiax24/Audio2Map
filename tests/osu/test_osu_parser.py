"""Tests for osu! parser."""

from pathlib import Path

import pytest

from audio2map.osu import parse_beatmap
from audio2map.osu.parser import (
    InvalidHitObjectError,
    InvalidTimingPointError,
    audio_filename,
    chart_audio_path,
    is_mania_4k,
    parse_hit_object,
    parse_notes,
    parse_timing_points,
    x_to_column,
)
from audio2map.utils.paths import raw_data_available, raw_dir

RAW = raw_dir()


def _raw_file(rel: str):
    """Return path if readable, else None (no PermissionError)."""
    if not raw_data_available():
        return None
    path = RAW / rel
    try:
        return path if path.is_file() else None
    except OSError:
        return None


@pytest.mark.parametrize(
    ("x", "col"),
    [
        (64, 0),
        (192, 1),
        (320, 2),
        (448, 3),
        (0, 0),
        (128, 1),
        (256, 2),
        (384, 3),
        (358, 2),
        (51, 0),
        (96, 0),
        (480, 3),
    ],
)
def test_x_to_column(x: int, col: int) -> None:
    assert x_to_column(x) == col


@pytest.mark.requires_data
def test_parse_real_beatmap() -> None:
    path = _raw_file("871715/Amane - TWEEKER (FAMoss) [UNPREDICTABLE'S NORMAL].osu")
    if path is None:
        pytest.skip("dataset not present")
    bm = parse_beatmap(path)
    assert bm.note_count > 0
    assert len(bm.notes) == bm.note_count


def test_parse_hit_object_tap() -> None:
    note = parse_hit_object("64,192,1000,1,0,0:0:0:0:")
    assert note.time_ms == 1000
    assert note.col == 0


def test_parse_hit_object_alt_x_layout() -> None:
    note = parse_hit_object("256,192,12055,1,0,1:0:1:20:")
    assert note is not None
    assert note.time_ms == 12055
    assert note.col == 2


@pytest.mark.requires_data
def test_parse_beatmap_alt_x_layout() -> None:
    path = _raw_file("2358613/hkmori - panic attack in bed (D_bobr) [normal].osu")
    if path is None:
        pytest.skip("dataset not present")
    bm = parse_beatmap(path)
    assert bm.note_count > 400
    assert len(bm.notes) == bm.note_count


def test_parse_hit_object_duplicate_same_lane_is_deduped() -> None:
    notes = parse_notes(
        [
            "64,192,116630,1,0,0:0:0:0:",
            "64,192,116630,1,0,0:0:0:0:",
        ]
    )
    assert len(notes) == 1
    assert notes[0].col == 0


def _write_4k(tmp_path, hit_objects: str, *, timing: str = "0,500,4,2,0,50,1,0"):
    path = tmp_path / "map.osu"
    path.write_text(
        "osu file format v14\n"
        "[General]\nMode:3\n"
        "[Difficulty]\nCircleSize:4\n"
        "[TimingPoints]\n"
        f"{timing}\n"
        "[HitObjects]\n"
        f"{hit_objects}\n",
        encoding="utf-8",
    )
    return path


def test_parse_hit_object_rejects_unsupported_type() -> None:
    with pytest.raises(InvalidHitObjectError, match="unsupported type"):
        parse_hit_object("64,192,1000,0,0,0:0:0:0:")
    with pytest.raises(InvalidHitObjectError, match="unsupported type"):
        parse_hit_object("64,192,1000,2,0,0:0:0:0:")


def test_parse_hit_object_rejects_invalid_hold() -> None:
    with pytest.raises(InvalidHitObjectError, match="invalid hold"):
        parse_hit_object("64,192,1000,128,0,1000:0:0:0:0:")


def test_parse_beatmap_rejects_unsupported_hit_object(tmp_path) -> None:
    path = _write_4k(tmp_path, "64,192,1000,8,0,0:0:0:0:")
    with pytest.raises(InvalidHitObjectError):
        parse_beatmap(path)


def test_filter_rejects_invalid_hit_objects(tmp_path) -> None:
    from audio2map.dataset.filter import FilterReason, check_osu_path

    path = _write_4k(
        tmp_path,
        "64,192,0,1,0,0:0:0:0:\n64,192,1000,8,0,0:0:0:0:",
    )
    result = check_osu_path(path)
    assert result.reason == FilterReason.INVALID_HIT_OBJECTS
    assert not result.eligible


def test_parse_timing_points_rejects_short_line() -> None:
    with pytest.raises(InvalidTimingPointError, match="too few fields"):
        parse_timing_points(["0,500,4"])


def test_filter_rejects_invalid_timing_points(tmp_path) -> None:
    from audio2map.dataset.filter import FilterReason, check_osu_path

    path = _write_4k(tmp_path, "64,192,0,1,0,0:0:0:0:", timing="0,500,4")
    result = check_osu_path(path)
    assert result.reason == FilterReason.INVALID_TIMING_POINTS
    assert not result.eligible


@pytest.mark.requires_data
def test_duplicate_hitobject_chart_round_trips() -> None:
    from audio2map.grid import CanonicalTiming, TickNote, chart_event_bar_range
    from audio2map.tokens import encode_notes, tokens_to_notes

    path = _raw_file(
        "2452954/EBIMAYO - NIGHTMARE INVITATION (Lleethenoob) [S K Y Y - ADVANCED].osu"
    )
    if path is None:
        pytest.skip("dataset not present")
    bm = parse_beatmap(path)
    assert bm.note_count == 1088
    timing = CanonicalTiming.from_beatmap(bm)
    start_bar, _ = chart_event_bar_range(bm.notes, timing)
    assert TickNote.from_notes(bm.notes, timing) == TickNote.from_notes(
        tokens_to_notes(encode_notes(bm.notes, timing), timing, start_bar=start_bar).notes,
        timing,
    )


def _authoritative_note_heads(beatmap) -> list[tuple[int, int]]:
    return sorted((note.time_ms, note.col) for note in beatmap.notes)


def _analyser_note_heads(path) -> list[tuple[int, int]]:
    from audio2map.features.cond.pattern_analyser.chart import NoteType as AnalyserNoteType
    from audio2map.features.cond.pattern_analyser.osu_parser import parse_osu_mania

    heads = (AnalyserNoteType.NORMAL, AnalyserNoteType.HOLDHEAD)
    chart = parse_osu_mania(str(path))
    out: list[tuple[int, int]] = []
    for item in chart.Notes:
        time_ms = round(item.Time)
        for col, note_type in enumerate(item.Data):
            if note_type in heads:
                out.append((time_ms, col))
    return sorted(out)


def test_note_heads_match_pattern_analyser_on_synthetic(tmp_path) -> None:
    osu = tmp_path / "heads.osu"
    osu.write_text(
        """osu file format v14

[General]
AudioFilename: song.mp3
Mode: 3

[Metadata]
Title: Heads
Artist: Test
Creator: Test
Version: 4K
BeatmapID: 1
BeatmapSetID: 1

[Difficulty]
HPDrainRate: 7
CircleSize: 4
OverallDifficulty: 8

[TimingPoints]
0,500,4,2,0,50,1,0

[HitObjects]
64,192,0,1,0,0:0:0:0:
192,192,500,1,0,0:0:0:0:
320,192,1000,128,0,1500:0:0:0:0:
448,192,2000,1,0,0:0:0:0:
""",
        encoding="utf-8",
    )
    bm = parse_beatmap(osu)
    assert is_mania_4k(osu)
    assert audio_filename(osu) == "song.mp3"
    assert _authoritative_note_heads(bm) == _analyser_note_heads(osu)
    assert _authoritative_note_heads(bm) == [
        (0, 0),
        (500, 1),
        (1000, 2),
        (2000, 3),
    ]


@pytest.mark.requires_data
def test_note_heads_match_pattern_analyser_on_offset_x_chart() -> None:
    path = _raw_file("936195/tatatat - Melody of Promise (tatatat) [4K Beginner].osu")
    if path is None:
        pytest.skip("dataset not present")
    bm = parse_beatmap(path)
    assert bm.note_count == 91
    assert _authoritative_note_heads(bm) == _analyser_note_heads(path)


def test_chart_audio_path_uses_audio_filename(tmp_path: Path) -> None:
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"x")
    osu = tmp_path / "chart.osu"
    osu.write_text("[General]\nAudioFilename: song.mp3\n", encoding="utf-8")
    assert chart_audio_path(osu) == audio


def test_chart_audio_path_missing_file(tmp_path: Path) -> None:
    osu = tmp_path / "chart.osu"
    osu.write_text("[General]\nAudioFilename: song.mp3\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="audio not found"):
        chart_audio_path(osu)


def test_chart_audio_path_missing_filename(tmp_path: Path) -> None:
    osu = tmp_path / "chart.osu"
    osu.write_text("[General]\nMode: 3\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing AudioFilename"):
        chart_audio_path(osu)
