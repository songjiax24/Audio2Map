"""Beatmapset split from raw/ layout."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from audio2map.dataset.split import (
    SplitError,
    SplitManifest,
    build_split_manifest,
    group_id_for_chart,
    load_split_manifest,
    manifest_sha256,
    save_split_manifest,
    split_group_chart_counts,
    split_paths,
    validate_chart_set,
)


def _write_osu(path: Path, *, set_id: int | None, beatmap_id: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    set_line = f"BeatmapSetID: {set_id}\n" if set_id is not None else ""
    path.write_text(
        "[Metadata]\n"
        f"BeatmapID: {beatmap_id}\n"
        f"{set_line}",
        encoding="utf-8",
    )


def test_group_id_is_raw_relative_parent(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    chart = raw / "12345" / "easy.osu"
    _write_osu(chart, set_id=12345)
    assert group_id_for_chart(chart, raw) == "12345"


def test_group_id_allows_non_numeric_and_nested(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    chart = raw / "custom_pack" / "song_a" / "hard.osu"
    _write_osu(chart, set_id=None)
    assert group_id_for_chart(chart, raw) == "custom_pack/song_a"


def test_group_id_rejects_chart_directly_in_raw(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    chart = raw / "loose.osu"
    _write_osu(chart, set_id=1)
    with pytest.raises(SplitError, match="group directory"):
        group_id_for_chart(chart, raw)


def test_same_directory_stays_in_one_split(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    a = raw / "pack_a" / "easy.osu"
    b = raw / "pack_a" / "hard.osu"
    c = raw / "pack_b" / "insane.osu"
    _write_osu(a, set_id=None)
    _write_osu(b, set_id=None)
    _write_osu(c, set_id=None)
    manifest = build_split_manifest(raw_root=raw, seed=0)
    assert manifest.groups[group_id_for_chart(a, raw)] == manifest.groups[group_id_for_chart(b, raw)]
    assigned = split_paths([a, b, c], manifest, raw_root=raw)
    a_split = next(s for s, ps in assigned.items() if a in ps)
    assert b in assigned[a_split]


def test_set_id_mismatch_fail_fast(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    chart = raw / "12345" / "a.osu"
    _write_osu(chart, set_id=67890)
    with pytest.raises(SplitError, match="BeatmapSetID/folder"):
        build_split_manifest(raw_root=raw, seed=0)


def test_set_id_mismatch_allowed_records_audit(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    chart = raw / "12345" / "a.osu"
    _write_osu(chart, set_id=67890)
    manifest = build_split_manifest(raw_root=raw, seed=0, allow_set_id_mismatch=True)
    assert manifest.audit["set_id_mismatches"]
    assert manifest.groups["12345"] in {"train", "val", "test"}


def test_new_chart_in_known_group_is_error(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    a = raw / "g1" / "a.osu"
    _write_osu(a, set_id=None)
    manifest = build_split_manifest(raw_root=raw, seed=0)
    c = raw / "g1" / "c.osu"
    _write_osu(c, set_id=None)
    with pytest.raises(SplitError, match="not in the split manifest"):
        split_paths([a, c], manifest, raw_root=raw)


def test_missing_snapshot_chart_is_error(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    a = raw / "g1" / "a.osu"
    b = raw / "g1" / "b.osu"
    _write_osu(a, set_id=None)
    _write_osu(b, set_id=None)
    manifest = build_split_manifest(raw_root=raw, seed=0)
    b.unlink()
    with pytest.raises(SplitError, match="missing from disk"):
        split_paths([a], manifest, raw_root=raw)


def test_unknown_group_is_error(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    a = raw / "g1" / "a.osu"
    _write_osu(a, set_id=None)
    manifest = build_split_manifest(raw_root=raw, seed=0)
    extra = raw / "g_new" / "x.osu"
    _write_osu(extra, set_id=None)
    with pytest.raises(SplitError, match="not in the split manifest"):
        split_paths([a, extra], manifest, raw_root=raw)


def test_groups_are_source_of_truth(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    a = raw / "g1" / "a.osu"
    _write_osu(a, set_id=None)
    manifest = build_split_manifest(raw_root=raw, seed=0)
    other = "test" if manifest.charts[0].split != "test" else "train"
    bad = SplitManifest(
        split_scheme=manifest.split_scheme,
        group_by=manifest.group_by,
        seed=manifest.seed,
        ratios=dict(manifest.ratios),
        groups=dict(manifest.groups),
        charts=[replace(manifest.charts[0], split=other)],
        audit=dict(manifest.audit),
    )
    with pytest.raises(SplitError, match="groups\\["):
        validate_chart_set(bad, [a], raw_root=raw)


def test_snapshot_group_id_must_match_layout(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    a = raw / "g1" / "a.osu"
    _write_osu(a, set_id=None)
    manifest = build_split_manifest(raw_root=raw, seed=0)
    bad = SplitManifest(
        split_scheme=manifest.split_scheme,
        group_by=manifest.group_by,
        seed=manifest.seed,
        ratios=dict(manifest.ratios),
        groups={"other": manifest.groups["g1"]},
        charts=[replace(manifest.charts[0], group_id="other")],
        audit=dict(manifest.audit),
    )
    with pytest.raises(SplitError, match="layout"):
        validate_chart_set(bad, [a], raw_root=raw)


def test_load_rejects_seed_mismatch(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_osu(raw / "g1" / "a.osu", set_id=None)
    manifest = build_split_manifest(raw_root=raw, seed=0)
    path = tmp_path / "beatmapset_v1.json"
    save_split_manifest(manifest, path)
    with pytest.raises(SplitError, match="seed"):
        load_split_manifest(path, expected_seed=99)


def test_save_refuses_overwrite(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_osu(raw / "g1" / "a.osu", set_id=None)
    manifest = build_split_manifest(raw_root=raw, seed=0)
    path = tmp_path / "beatmapset_v1.json"
    save_split_manifest(manifest, path)
    with pytest.raises(SplitError, match="already exists"):
        save_split_manifest(manifest, path)


def test_manifest_sha256_ignores_audit_order_and_indent(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_osu(raw / "g1" / "a.osu", set_id=None)
    _write_osu(raw / "g2" / "b.osu", set_id=None)
    manifest = build_split_manifest(raw_root=raw, seed=0)
    baseline = manifest_sha256(manifest)
    shuffled = SplitManifest(
        split_scheme=manifest.split_scheme,
        group_by=manifest.group_by,
        seed=manifest.seed,
        ratios=dict(manifest.ratios),
        groups=dict(manifest.groups),
        charts=list(reversed(manifest.charts)),
        audit={"n_charts": 99, "note": "ignored"},
    )
    assert manifest_sha256(shuffled) == baseline
    path = tmp_path / "beatmapset_v1.json"
    save_split_manifest(manifest, path)
    loaded = load_split_manifest(path)
    assert manifest_sha256(loaded) == baseline
    flipped = dict(manifest.groups)
    key = next(iter(flipped))
    flipped[key] = "test" if flipped[key] != "test" else "train"
    changed_groups = SplitManifest(
        split_scheme=manifest.split_scheme,
        group_by=manifest.group_by,
        seed=manifest.seed,
        ratios=dict(manifest.ratios),
        groups=flipped,
        charts=list(manifest.charts),
        audit=dict(manifest.audit),
    )
    assert manifest_sha256(changed_groups) != baseline


def test_split_group_chart_counts(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_osu(raw / "g1" / "a.osu", set_id=None)
    _write_osu(raw / "g1" / "b.osu", set_id=None)
    _write_osu(raw / "g2" / "c.osu", set_id=None)
    manifest = build_split_manifest(raw_root=raw, seed=0)
    counts = split_group_chart_counts(manifest)
    assert sum(g for g, _ in counts.values()) == 2
    assert sum(c for _, c in counts.values()) == 3
    g1_split = manifest.groups["g1"]
    assert counts[g1_split][1] >= 2
