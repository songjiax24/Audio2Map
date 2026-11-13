"""v0 teacher and generation share donor style while retaining target audio BPM."""

from types import SimpleNamespace

import numpy as np
import torch

from audio2map.eval import chart_eval
from audio2map.features.cond import CANONICAL_BPM_NORM_INDEX, COND_VEC_DIM, ChartMeta


def test_random_full_keeps_target_bpm_in_teacher_and_generation(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("a.osu", "b.osu")]
    target = np.arange(COND_VEC_DIM, dtype=np.float32)
    donor = target + 100
    vectors = [target, donor]
    originals = [vector.copy() for vector in vectors]
    prepared = [
        chart_eval.EvalChart(
            path, path.name, ChartMeta(str(path)), vector,
            SimpleNamespace(cond_vec=vector),
        )
        for path, vector in zip(paths, vectors)
    ]
    mapping = SimpleNamespace(pairs={"a.osu": "b.osu", "b.osu": "a.osu"})
    monkeypatch.setattr(chart_eval, "materialize_eval_charts", lambda *a, **k: prepared)
    monkeypatch.setattr(chart_eval, "require_donor_map", lambda *a, **k: mapping)
    monkeypatch.setattr(chart_eval, "donor_pool_report", lambda *a: {
        "candidate_count": 2, "universe_id": "test", "pairs": [],
    })
    seen = []

    def capture(_model, _sample, condition, **kwargs):
        seen.append(condition.copy())

    monkeypatch.setattr(chart_eval, "_score_teacher_sample", capture)
    result = chart_eval.eval_teacher_forcing(
        torch.nn.Identity(), paths, device=torch.device("cpu"),
        scenarios=("matched_full", "random_full"), raw_root=tmp_path,
    )
    assert result.to_dict()["random_full_policy"] == "donor_style_target_bpm_v1"
    by_rel = {item.relpath: (item.path, item.meta, item.cond_vec) for item in prepared}
    for index, item in enumerate(prepared):
        np.testing.assert_array_equal(seen[index], vectors[index])
        expected = vectors[1 - index].copy()
        expected[CANONICAL_BPM_NORM_INDEX] = vectors[index][CANONICAL_BPM_NORM_INDEX]
        np.testing.assert_array_equal(seen[2 + index], expected)
        used, donor_path, adherence = chart_eval._condition_for(
            item.path, item.meta, item.cond_vec, "random_full", mapping, by_rel, tmp_path,
        )
        np.testing.assert_array_equal(used, expected)
        assert donor_path == prepared[1 - index].relpath
        assert adherence is prepared[1 - index].meta
        np.testing.assert_array_equal(vectors[index], originals[index])
