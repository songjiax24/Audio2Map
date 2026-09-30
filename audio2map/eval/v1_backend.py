"""v1 chart-level adapter.

v1 fills ``GeneratedChart`` from its own generator. It does not build teacher
windows, audio grids, or 16-bar overlap sequences.
"""

from __future__ import annotations

from pathlib import Path

from audio2map.eval.result import GeneratedChart
from audio2map.features.cond import ChartMeta
from audio2map.grid import CanonicalTiming
from audio2map.osu.schema import ManiaNote


def generated_chart(
    *,
    relpath: str,
    path: Path,
    notes: list[ManiaNote],
    timing: CanonicalTiming,
    reference_notes: list[ManiaNote],
    target_meta: ChartMeta,
    adherence_meta: ChartMeta,
    validity: str,
    decode_issues: list[dict[str, str | int]] | None = None,
    scenario: str,
    mode: str,
    seed: int,
    donor: str | None = None,
) -> GeneratedChart:
    return GeneratedChart(
        relpath=relpath,
        path=path,
        notes=notes,
        timing=timing,
        reference_notes=reference_notes,
        target_meta=target_meta,
        adherence_meta=adherence_meta,
        validity=validity,
        decode_issues=list(decode_issues or []),
        scenario=scenario,
        mode=mode,
        seed=seed,
        donor=donor,
    )
