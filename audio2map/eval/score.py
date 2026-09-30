"""Shared chart-level scoring for any backend that can fill ``GeneratedChart``."""

from __future__ import annotations

from pathlib import Path

from audio2map.eval.adherence import condition_mae
from audio2map.eval.diagnostics import enters_chart_metrics
from audio2map.eval.note_match import reference_match
from audio2map.eval.result import GeneratedChart
from audio2map.features.cond import compute_chart_meta
from audio2map.osu.export import write_osu


def score_chart(chart: GeneratedChart, adherence_path: Path) -> dict | None:
    """Score one completed chart. Unconstrained results return ``None``.

    A constrained chart with ``missing_hold_tail`` still returns match and
    adherence. The issue list is copied through and does not change validity.
    """
    if not enters_chart_metrics(chart.mode, chart.validity):
        return None
    write_osu(adherence_path, chart.notes, source_osu=chart.path)
    return {
        "osu": str(chart.path),
        "donor": chart.donor,
        "validity": chart.validity,
        "decode_issues": list(chart.decode_issues),
        "match": reference_match(chart.notes, chart.reference_notes, chart.timing),
        "adherence": condition_mae(chart.adherence_meta, compute_chart_meta(adherence_path)),
    }
