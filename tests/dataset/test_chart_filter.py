"""Chart eligibility filter tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio2map.dataset.filter import (
    FilterReason,
    check_beatmap_eligibility,
    check_osu_path,
    list_eligible_osu_paths,
)
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.schema import Beatmap, ChartMetadata, ManiaNote, NoteType, TimingPoint
from audio2map.utils.paths import raw_data_available, raw_dir


def _beatmap(notes: list[ManiaNote], *, bpm: float = 180.0, offset_ms: int = 0) -> Beatmap:
    return Beatmap(
        path=Path("x.osu"),
        metadata=ChartMetadata("", "", "", "", None, None),
        timing_points=[
            TimingPoint(
                offset_ms=offset_ms,
                beat_length_ms=60_000.0 / bpm,
                meter=4,
                uninherited=True,
            )
        ],
        notes=notes,
    )


def test_legal_notes_eligible() -> None:
    result = check_beatmap_eligibility(
        _beatmap(
            [
                ManiaNote(time_ms=0, col=0, note_type=NoteType.TAP),
                ManiaNote(time_ms=500, col=1, note_type=NoteType.TAP),
            ]
        )
    )
    assert result.eligible
    assert result.reason is None


def test_collapsed_hold_invalid_grid() -> None:
    result = check_beatmap_eligibility(
        _beatmap(
            [ManiaNote(time_ms=0, col=0, note_type=NoteType.HOLD, end_time_ms=1)]
        )
    )
    assert not result.eligible
    assert result.reason == FilterReason.INVALID_GRID


def test_same_tick_same_lane_invalid_grid() -> None:
    result = check_beatmap_eligibility(
        _beatmap(
            [
                ManiaNote(time_ms=0, col=0, note_type=NoteType.TAP),
                ManiaNote(time_ms=1, col=0, note_type=NoteType.TAP),
            ]
        )
    )
    assert not result.eligible
    assert result.reason == FilterReason.INVALID_GRID


def test_tap_during_hold_invalid_grid() -> None:
    result = check_beatmap_eligibility(
        _beatmap(
            [
                ManiaNote(time_ms=0, col=0, note_type=NoteType.HOLD, end_time_ms=1000),
                ManiaNote(time_ms=500, col=0, note_type=NoteType.TAP),
            ]
        )
    )
    assert not result.eligible
    assert result.reason == FilterReason.INVALID_GRID


EMPTY_CHARTS = (
    "1985016/LE SSERAFIM - UNFORGIVEN (feat. Nile Rodgers) (Ilham) [Normal].osu",
    "2107667/fhana - Eien to Iu Hikari (Game Ver.) (Syrion-) [Virtue's Eternity].osu",
)


@pytest.mark.requires_data
@pytest.mark.parametrize("rel_path", EMPTY_CHARTS)
def test_empty_hitobjects_not_eligible(rel_path: str) -> None:
    if not raw_data_available():
        pytest.skip("no dataset")
    path = raw_dir() / rel_path
    try:
        exists = path.is_file()
    except OSError:
        pytest.skip(f"inaccessible fixture: {path}")
    if not exists:
        pytest.skip(f"missing fixture: {path}")

    osu_result = check_osu_path(path)
    assert not osu_result.eligible
    assert osu_result.reason == FilterReason.NO_NOTES

    beatmap = parse_beatmap(path)
    assert beatmap.note_count == 0
    bm_result = check_beatmap_eligibility(beatmap)
    assert not bm_result.eligible
    assert bm_result.reason == FilterReason.NO_NOTES


@pytest.mark.requires_data
def test_nonempty_chart_still_eligible() -> None:
    if not raw_data_available():
        pytest.skip("no dataset")
    root = raw_dir()
    try:
        paths = sorted(root.rglob("*.osu"))
    except OSError:
        pytest.skip("raw/ inaccessible")
    for path in paths:
        beatmap = parse_beatmap(path)
        if beatmap.note_count == 0:
            continue
        result = check_beatmap_eligibility(beatmap)
        if result.eligible:
            return
    pytest.skip("no eligible chart with notes in raw data")


def test_list_eligible_osu_paths(tmp_path: Path) -> None:
    good = tmp_path / "good.osu"
    good.write_text(
        "osu file format v14\n"
        "[General]\nMode:3\n"
        "[Difficulty]\nCircleSize:4\n"
        "[TimingPoints]\n0,500,4,2,0,50,1,0\n"
        "[HitObjects]\n64,192,0,1,0,0:0:0:0:\n",
        encoding="utf-8",
    )
    (tmp_path / "empty.osu").write_text(
        "osu file format v14\n"
        "[General]\nMode:3\n"
        "[Difficulty]\nCircleSize:4\n"
        "[TimingPoints]\n0,500,4,2,0,50,1,0\n"
        "[HitObjects]\n",
        encoding="utf-8",
    )
    assert list_eligible_osu_paths(tmp_path) == [good]
