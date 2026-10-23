"""Scheme-A pattern shares, from one parse and one pattern search.

Each CorePattern share is the classic union of that class's ``[Start, End]``
intervals, divided by ``LastNote - FirstNote``. Overlap across classes is
kept, so the six shares need not sum to 1. Importance, clusters, RC
filtering, and BPM cuts are not applied here.

The same search feeds the v0 card features (cluster Importance).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio2map.features.cond.pattern_analyser.analyze import (
    PatternFeatures,
    features_from_report,
)
from audio2map.features.cond.pattern_analyser.find_patterns import find
from audio2map.features.cond.pattern_analyser.osu_parser import parse_osu_mania
from audio2map.features.cond.pattern_analyser.patterns_def import CorePattern
from audio2map.features.cond.pattern_analyser.summary import from_chart

PATTERN_ORDER: tuple[str, ...] = tuple(pattern.value for pattern in CorePattern)


def union_ms(intervals: list[tuple[float, float]]) -> float:
    """Length of the classic union. A later start opens a new segment."""
    usable = [
        (float(start), float(end))
        for start, end in intervals
        if np.isfinite(start) and np.isfinite(end) and end > start
    ]
    if not usable:
        return 0.0
    usable.sort(key=lambda item: (item[0], item[1]))
    total = 0.0
    current_start, current_end = usable[0]
    for start, end in usable[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start = start
            current_end = end
        else:
            current_end = max(current_end, end)
    total += current_end - current_start
    return total


def amounts_from_found(found, duration: float) -> np.ndarray | None:
    """Six Amount fractions in ``CorePattern`` order, each clipped to 1.

    Returns None when duration is not a positive finite number. Clipping is
    per pattern, so the six values may still sum to more than 1.
    """
    if not np.isfinite(duration) or duration <= 0.0:
        return None
    amounts = np.empty(len(PATTERN_ORDER), dtype=np.float64)
    for index, pattern in enumerate(CorePattern):
        intervals = [
            (item.Start, item.End) for item in found if item.Pattern == pattern
        ]
        amounts[index] = min(union_ms(intervals) / duration, 1.0)
    if not np.isfinite(amounts).all():
        return None
    return amounts


@dataclass(frozen=True, slots=True)
class PatternPass:
    """v0 Importance features and v1 Amount time fractions from one search."""

    features: PatternFeatures
    pattern_amounts: np.ndarray | None


def analyze_patterns_once(path: Path | str) -> PatternPass:
    """Parse the chart once and search for patterns once."""
    chart = parse_osu_mania(str(path))
    found = find(chart)
    features = features_from_report(from_chart(chart, found))
    duration = float(chart.LastNote - chart.FirstNote)
    return PatternPass(features, amounts_from_found(found, duration))
