"""Tests for osu! parser."""

from pathlib import Path

import pytest

from audio2map.osu import parse_beatmap
from audio2map.osu.mania import parse_hit_object, x_to_column
from audio2map.osu.parser import _parse_notes
from audio2map.utils.paths import raw_dir

RAW = raw_dir()


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


def test_parse_real_beatmap() -> None:
    path = RAW / "871715" / "Amane - TWEEKER (FAMoss) [UNPREDICTABLE'S NORMAL].osu"
    if not path.exists():
        pytest.skip("dataset not present")
    bm = parse_beatmap(path)
    assert len(bm.notes) == bm.note_count
    assert bm.duration_ms > 0


def test_parse_hit_object_tap() -> None:
    note = parse_hit_object("64,192,1000,1,0,0:0:0:0:")
    assert note.time_ms == 1000
    assert note.col == 0


def test_parse_hit_object_alt_x_layout() -> None:
    note = parse_hit_object("256,192,12055,1,0,1:0:1:20:")
    assert note is not None
    assert note.time_ms == 12055
    assert note.col == 2


def test_parse_beatmap_alt_x_layout() -> None:
    path = RAW / "2358613" / "hkmori - panic attack in bed (D_bobr) [normal].osu"
    if not path.exists():
        pytest.skip("dataset not present")
    bm = parse_beatmap(path)
    assert bm.note_count > 400
    assert len(bm.notes) == bm.note_count


def test_parse_hit_object_duplicate_same_lane_is_deduped() -> None:
    notes = _parse_notes(
        [
            "64,192,116630,1,0,0:0:0:0:",
            "64,192,116630,1,0,0:0:0:0:",
        ]
    )
    assert len(notes) == 1
    assert notes[0].col == 0


def test_duplicate_hitobject_chart_round_trips() -> None:
    from audio2map.osu.round_trip import round_trip_beatmap

    path = RAW / "2452954" / "EBIMAYO - NIGHTMARE INVITATION (Lleethenoob) [S K Y Y - ADVANCED].osu"
    if not path.exists():
        pytest.skip("dataset not present")
    bm = parse_beatmap(path)
    assert bm.note_count == 1088
    assert round_trip_beatmap(bm)


def test_hold_ratio_matches_analyzer_ln_on_offset_x_chart() -> None:
    from audio2map.data.chart_stats import hold_ratio
    from audio2map.pattern_analyser import analyze_osu

    path = RAW / "936195" / "tatatat - Melody of Promise (tatatat) [4K Beginner].osu"
    if not path.exists():
        pytest.skip("dataset not present")
    bm = parse_beatmap(path)
    pat = analyze_osu(path)
    assert pat.analyzer_available
    assert hold_ratio(bm) == pat.ln_percent
    assert bm.note_count == 91
