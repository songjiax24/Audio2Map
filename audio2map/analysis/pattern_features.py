"""Pattern analyser feature wrapper for chart metadata and training conditions."""

from __future__ import annotations

from pathlib import Path

from audio2map.pattern_analyser import PatternFeatures, analyze_osu

__all__ = ["PatternFeatures", "analyze_chart_patterns"]


def analyze_chart_patterns(path: Path | str) -> PatternFeatures:
    return analyze_osu(path)
