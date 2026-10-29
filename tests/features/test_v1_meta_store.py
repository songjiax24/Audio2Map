"""Rate-1 v1 ChartMeta disk table."""

from __future__ import annotations

from pathlib import Path

from audio2map.features.cond.chart import ChartMeta
from audio2map.features.cond.v1.store import (
    append_v1_chart_meta_records,
    clear_v1_chart_meta_cache,
    load_v1_chart_meta,
    measure_v1_chart_meta,
    rate1_meta_for,
)


def _record(relpath: str, stars: float) -> dict:
    payload = ChartMeta(osu_path=relpath, official_sr=stars).to_dict()
    payload["relpath"] = relpath
    return payload


def test_append_skips_a_relpath_already_stored(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    append_v1_chart_meta_records([_record("a/b.osu", 1.5)], manifest)
    append_v1_chart_meta_records([_record("a/b.osu", 9.0)], manifest)
    loaded = load_v1_chart_meta(manifest)
    assert list(loaded) == ["a/b.osu"]
    assert loaded["a/b.osu"].official_sr == 1.5


def test_rate1_meta_for_reads_the_disk_table(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    append_v1_chart_meta_records([_record("a/b.osu", 2.5)], manifest)
    clear_v1_chart_meta_cache()
    meta = rate1_meta_for(tmp_path, "a/b.osu", manifest)
    assert meta.official_sr == 2.5
    clear_v1_chart_meta_cache()


def test_measure_missing_chart_is_a_failed_row(tmp_path: Path) -> None:
    record = measure_v1_chart_meta((str(tmp_path), "nope.osu"))
    assert record["relpath"] == "nope.osu"
    assert record["error"]
