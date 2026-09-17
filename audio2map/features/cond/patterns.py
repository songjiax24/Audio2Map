"""Pattern analyser wrapper for the chart condition table."""

from __future__ import annotations

from pathlib import Path

from audio2map.features.cond.pattern_analyser import PatternFeatures, analyze_osu

__all__ = ["PatternFeatures", "analyze_chart_patterns"]


def analyze_chart_patterns(path: Path | str) -> PatternFeatures:
    """Product name for the vendored ``analyze_osu`` pattern stats."""
    return analyze_osu(path)
