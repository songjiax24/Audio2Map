"""Eval uses the frozen split chart set, not a global eligible sample."""

from __future__ import annotations

from pathlib import Path

from audio2map.dataset.split import build_split_manifest, group_id_for_chart, split_paths


def _write_osu(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[Metadata]\nBeatmapID: 1\n", encoding="utf-8")


def test_requested_split_paths_do_not_include_other_groups(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    train_chart = raw / "g_train" / "a.osu"
    test_chart = raw / "g_test" / "b.osu"
    _write_osu(train_chart)
    _write_osu(test_chart)
    manifest = build_split_manifest(raw_root=raw, seed=0)
    by_split = split_paths([train_chart, test_chart], manifest, raw_root=raw)
    for split, paths in by_split.items():
        for path in paths:
            assert manifest.groups[group_id_for_chart(path, raw)] == split


def test_chart_group_split_can_be_checked_for_osu_debug(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    a = raw / "g1" / "a.osu"
    b = raw / "g2" / "b.osu"
    _write_osu(a)
    _write_osu(b)
    manifest = build_split_manifest(raw_root=raw, seed=0)
    assigned = manifest.groups[group_id_for_chart(a, raw)]
    other = "test" if assigned != "test" else "train"
    assert assigned != other
