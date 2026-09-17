"""Smoke test for pattern features via the features-layer wrapper."""

from __future__ import annotations

import pytest

from audio2map.utils.paths import raw_data_available, raw_dir


@pytest.mark.requires_data
def test_pattern_analyser() -> None:
    from audio2map.features.cond import analyze_chart_patterns

    if not raw_data_available():
        pytest.skip("no dataset")
    path = next(raw_dir().rglob("*.osu"), None)
    if path is None:
        pytest.skip("no dataset")
    feat = analyze_chart_patterns(path)
    assert abs(sum(feat.core_dist.values()) - 1.0) < 1e-6 or sum(feat.core_dist.values()) == 0
