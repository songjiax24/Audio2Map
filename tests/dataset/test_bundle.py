"""Tests for chart-bundle preload."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio2map.utils.paths import audio_grid_dir, raw_data_available, raw_dir


@pytest.mark.requires_data
@pytest.mark.skipif(not raw_data_available(), reason="no dataset")
def test_preload_chart_bundles_share_audio_grid() -> None:
    from audio2map.dataset.bundle import (
        SharedGridCache,
        filter_v0_paths,
        preload_chart_bundle,
    )
    from audio2map.features.audio.grid import resolve_grid_stem
    from audio2map.grid import CanonicalTiming
    from audio2map.osu.parser import chart_audio_path, parse_beatmap

    root = raw_dir()
    pair: list[Path] = []
    for set_dir in root.iterdir():
        if not set_dir.is_dir():
            continue
        groups = {}
        charts = filter_v0_paths(
            sorted(set_dir.glob("*.osu")), require_grid=True, grid_dir=audio_grid_dir(),
        )
        for path in charts:
            timing = CanonicalTiming.from_beatmap(parse_beatmap(path))
            stem = resolve_grid_stem(chart_audio_path(path), timing)
            same_grid = groups.setdefault(stem, [])
            same_grid.append(path)
            if len(same_grid) == 2:
                pair = same_grid
                break
        if pair:
            break
    if len(pair) < 2:
        pytest.skip("no pair of eligible charts sharing one training grid")

    cache = SharedGridCache(audio_grid_dir())
    b0 = preload_chart_bundle(pair[0], grid_cache=cache)
    b1 = preload_chart_bundle(pair[1], grid_cache=cache)
    assert b0 is not None and b1 is not None
    assert np.shares_memory(b0.grid, b1.grid)
    assert len(cache) == 1
