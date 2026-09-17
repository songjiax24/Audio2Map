"""Run pattern analysis on an .osu file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from audio2map.features.cond.pattern_analyser.osu_parser import parse_osu_mania
from audio2map.features.cond.pattern_analyser.patterns_def import CorePattern
from audio2map.features.cond.pattern_analyser.summary import from_chart


CORE_PATTERNS = (
    CorePattern.Stream,
    CorePattern.Chordstream,
    CorePattern.Jacks,
    CorePattern.Coordination,
    CorePattern.Density,
    CorePattern.Wildcard,
)


@dataclass(slots=True)
class PatternFeatures:
    ln_percent: float
    hb_row_ratio: float
    core_dist: dict[str, float] = field(default_factory=dict)
    mode_tag: str = ""
    category: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _core_distribution(clusters) -> dict[str, float]:
    totals = {p.value: 0.0 for p in CORE_PATTERNS}
    for cluster in clusters:
        name = cluster.Pattern.value
        if name in totals:
            totals[name] += float(cluster.Importance)
    total = sum(totals.values())
    if total <= 0:
        return {k: 0.0 for k in totals}
    return {k: v / total for k, v in totals.items()}


def analyze_osu(path: Path | str) -> PatternFeatures:
    """Analyze pattern features for a mania .osu file."""
    path = Path(path)
    chart = parse_osu_mania(str(path))
    report = from_chart(chart)
    return PatternFeatures(
        ln_percent=float(report.LNPercent),
        hb_row_ratio=float(report.HBRowRatio),
        core_dist=_core_distribution(report.Clusters),
        mode_tag=str(report.ModeTag),
        category=str(report.Category),
    )
