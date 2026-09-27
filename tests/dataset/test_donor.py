"""Donor identity is the saved path map, not a seed index."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio2map.dataset.chart_subset import (
    load_or_create_chart_subset,
    sample_chart_subset,
)
from audio2map.dataset.donor import (
    DonorMapError,
    candidate_universe_id,
    create_donor_map,
    donor_pool_report,
    load_donor_map,
    load_or_create_donor_map,
    save_donor_map,
)


def test_donor_map_is_reused_and_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "random_full.json"
    first = create_donor_map(["b.osu", "a.osu", "c.osu"], seed=1)
    save_donor_map(first, path)
    again = load_or_create_donor_map(path, ["c.osu", "a.osu"], seed=99)
    assert again.pairs["a.osu"] == first.pairs["a.osu"]
    assert again.seed == 1
    assert load_donor_map(path).pairs == first.pairs
    with pytest.raises(DonorMapError, match="already exists"):
        save_donor_map(first, path)


def test_missing_target_fails_instead_of_resampling(tmp_path: Path) -> None:
    path = tmp_path / "random_full.json"
    save_donor_map(create_donor_map(["a.osu"], seed=0), path)
    with pytest.raises(DonorMapError, match="missing"):
        load_or_create_donor_map(path, ["a.osu", "new.osu"], seed=0)


def test_self_draw_is_allowed() -> None:
    mapping = create_donor_map(["only.osu"], seed=0)
    assert mapping.pairs == {"only.osu": "only.osu"}


def test_donor_pool_identity_is_shared_by_reload(tmp_path: Path) -> None:
    targets = ["b/x.osu", "a/y.osu", "c/z.osu"]
    path = tmp_path / "random_full.json"
    first = load_or_create_donor_map(path, list(reversed(targets)), seed=3)
    second = load_or_create_donor_map(path, targets, seed=99)
    report = donor_pool_report(second, targets)
    assert second.pairs == first.pairs
    assert report["candidate_count"] == 3
    assert report["universe_id"] == candidate_universe_id(targets)
    assert [(row["target"], row["donor"]) for row in report["pairs"]] == [
        (rel, second.pairs[rel]) for rel in sorted(targets)
    ]


def test_diversity_subset_ignores_input_order(tmp_path: Path) -> None:
    universe = ["c.osu", "a.osu", "b.osu", "d.osu"]
    forward = sample_chart_subset(universe, 2, seed=1)
    backward = sample_chart_subset(list(reversed(universe)), 2, seed=1)
    assert forward == backward
    assert forward != tuple(universe[:2])
    path = tmp_path / "diversity_subset.json"
    saved = load_or_create_chart_subset(path, universe, count=2, seed=1)
    again = load_or_create_chart_subset(path, list(reversed(universe)), count=2, seed=99)
    assert again.charts == saved.charts == forward
