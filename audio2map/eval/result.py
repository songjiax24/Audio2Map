"""Normalized chart-level generation result.

Backends fill this object. The shared evaluator does not know whether the
notes came from a 16-bar overlap decode or from another generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from audio2map.features.cond import ChartMeta
from audio2map.grid import CanonicalTiming
from audio2map.osu.schema import ManiaNote


@dataclass(slots=True)
class GeneratedChart:
    relpath: str
    path: Path
    notes: list[ManiaNote]
    timing: CanonicalTiming
    reference_notes: list[ManiaNote]
    target_meta: ChartMeta
    adherence_meta: ChartMeta
    validity: str
    decode_issues: list[dict[str, str | int]] = field(default_factory=list)
    scenario: str = "matched_full"
    mode: str = "constrained"
    seed: int = 0
    donor: str | None = None
