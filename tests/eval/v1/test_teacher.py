import pytest

from audio2map.eval.v1.teacher import summarize_teacher


def row(loss, accuracy, tokens, *, kind="row"):
    result = {
        "tokens": tokens,
        "type_counts": {
            name: tokens if name == kind else 0 for name in ("pos", "row", "beat", "eos")
        },
    }
    for suffix in ("chart", "token"):
        result.update(
            {
                f"loss_{suffix}": loss,
                f"acc_{suffix}": accuracy,
                f"legal_top1_{suffix}": 1.0,
                f"legal_prob_{suffix}": 0.9,
            }
        )
    for name in ("pos", "row", "beat", "eos"):
        result.update(
            {
                f"loss_{name}": loss if name == kind else 0,
                name: accuracy if name == kind else 0,
                f"legal_top1_{name}": 1.0,
                f"legal_prob_{name}": 0.9,
            }
        )
    return result


def test_aggregation_keeps_chart_and_token_weighting_distinct():
    report = summarize_teacher([row(1, 0.5, 2), row(3, 0.25, 6)])
    assert report["loss_chart"] == 2
    assert report["loss_token"] == 2.5
    assert report["acc_chart"] == 0.375
    assert report["acc_token"] == pytest.approx(0.3125)
    assert report["by_type"]["row"]["nll"] == 2.5
    assert report["by_type"]["row"]["tokens"] == 8
    assert report["by_type"]["eos"]["nll"] is None
