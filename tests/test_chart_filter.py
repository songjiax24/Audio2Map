"""Chart eligibility filter tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio2map.data.chart_filter import FilterReason, check_beatmap_eligibility, check_osu_path
from audio2map.osu.parser import parse_beatmap
from audio2map.utils.paths import raw_dir

EMPTY_CHARTS = (
    "1985016/LE SSERAFIM - UNFORGIVEN (feat. Nile Rodgers) (Ilham) [Normal].osu",
    "2107667/fhana - Eien to Iu Hikari (Game Ver.) (Syrion-) [Virtue's Eternity].osu",
)


@pytest.mark.parametrize("rel_path", EMPTY_CHARTS)
def test_empty_hitobjects_not_eligible(rel_path: str) -> None:
    path = raw_dir() / rel_path
    if not path.is_file():
        pytest.skip(f"missing fixture: {path}")

    osu_result = check_osu_path(path)
    assert not osu_result.eligible
    assert osu_result.reason == FilterReason.NO_NOTES

    beatmap = parse_beatmap(path)
    assert beatmap.note_count == 0
    bm_result = check_beatmap_eligibility(beatmap)
    assert not bm_result.eligible
    assert bm_result.reason == FilterReason.NO_NOTES


def test_nonempty_chart_still_eligible() -> None:
    root = raw_dir()
    for path in sorted(root.rglob("*.osu")):
        beatmap = parse_beatmap(path)
        if beatmap.note_count == 0:
            continue
        result = check_beatmap_eligibility(beatmap)
        if result.eligible:
            return
    pytest.skip("no eligible chart with notes in raw data")
