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
        has_audio_grid,
        preload_chart_bundle,
    )

    root = raw_dir()
    pair: list[Path] = []
    for set_dir in root.iterdir():
        if not set_dir.is_dir():
            continue
        charts = [p for p in set_dir.glob("*.osu") if has_audio_grid(p)]
        if len(charts) >= 2:
            pair = charts[:2]
            break
    if len(pair) < 2:
        pytest.skip("no set with two grid-backed charts")

    cache = SharedGridCache(audio_grid_dir())
    b0 = preload_chart_bundle(pair[0], grid_cache=cache)
    b1 = preload_chart_bundle(pair[1], grid_cache=cache)
    assert b0 is not None and b1 is not None
    assert np.shares_memory(b0.grid, b1.grid)
    assert len(cache) == 1
