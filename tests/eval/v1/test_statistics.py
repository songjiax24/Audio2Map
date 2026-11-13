import pytest

from audio2map.eval.v1.statistics import (
    chart_values, condition_responses, describe, mean_interval, paired_difference,
)


def test_chart_bootstrap_does_not_count_seeds_as_independent_charts():
    rows = [{"relpath": "a", "score": 0.2}] * 5 + [{"relpath": "b", "score": 0.8}]
    values = chart_values(rows, lambda row: row["score"])
    result = mean_interval(list(values.values()))
    assert result["charts"] == 2
    assert result["mean"] == pytest.approx(0.5)
    assert result == mean_interval(list(values.values()))
    assert describe([None])["mean"] is None
    assert mean_interval([0.5])["low"] is None
    paired = paired_difference({"a": 0.2, "b": 0.4}, {"a": 0.3, "c": 0.8})
    assert paired["mean"] == pytest.approx(0.1)
    assert paired["left_only"] == paired["right_only"] == 1


def test_condition_response_excludes_self_draws_missing_and_unchanged_targets():
    rows = []
    for path, donor, left, right, output in (
        ("a", "b", 1, 3, 2), ("b", "b", 1, 4, 2),
        ("c", "a", 2, 2, 2), ("d", "a", 1, None, 2),
    ):
        for scenario, target, value in (("matched_full", left, 1), ("random_full", right, output)):
            rows.append({
                "relpath": path, "scenario": scenario, "mode": "constrained", "seed": 0,
                "condition": {"donor": donor, "meta": {"official_sr": target}},
                "generated_meta": {"official_sr": value}, "tokens_sha256": scenario,
            })
    result = condition_responses(rows, seed=0)
    assert result["self_draws_excluded"] == 1
    field = result["fields"]["official_sr"]
    assert field["pairs"] == field["missing"] == field["unchanged_target"] == 1
    assert field["slope"] == 0.5
