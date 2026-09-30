"""Validity-adjacent diagnostics that do not change validity labels."""

from __future__ import annotations


def enters_chart_metrics(mode: str, validity: str | None) -> bool:
    """Constrained completions are scored. Unconstrained stays on validity only.

    ``missing_hold_tail`` is not an argument: it does not remove a constrained
    chart from adherence, matching, statistics, motif, or diversity.
    """
    if mode == "unconstrained":
        return False
    if mode == "constrained":
        return validity == "valid"
    raise ValueError(f"unknown generation mode: {mode}")


def missing_hold_tail_report(issues_per_chart: list[list[dict]]) -> dict:
    """Count open-hold clips. Rate is charts with at least one such issue."""
    count = 0
    charts_with = 0
    for issues in issues_per_chart:
        n = sum(1 for issue in issues if issue.get("kind") == "missing_hold_tail")
        count += n
        if n:
            charts_with += 1
    n_charts = len(issues_per_chart)
    return {
        "missing_hold_tail_count": count,
        "missing_hold_tail_rate": charts_with / n_charts if n_charts else 0.0,
        "charts": n_charts,
    }
