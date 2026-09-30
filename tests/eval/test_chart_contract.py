"""Chart-level eval contract: cache, motif frequency, validity, frozen inputs."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from audio2map.dataset.chart_subset import ChartSubsetError, require_chart_subset, save_chart_subset
from audio2map.dataset.chart_subset import ChartSubset
from audio2map.dataset.donor import (
    DonorMapError,
    assert_exact_donor_universe,
    create_donor_map,
    require_donor_map,
    save_donor_map,
)
from audio2map.eval import chart_eval, v1_backend
from audio2map.eval.cache import cache_get_or_put, generation_cache_key
from audio2map.eval.chart_eval import (
    _generation_body,
    assert_materialization_complete,
    eval_generation,
)
from audio2map.eval.diagnostics import enters_chart_metrics, missing_hold_tail_report
from audio2map.eval.diversity_eval import exact_duplicate_report, seed_pair_report
from audio2map.eval.frozen_inputs import (
    DIVERSITY_GENERATION_SEEDS,
    MOTIF_LIST_COUNT,
    MOTIF_SCOPES,
    assert_motif_list_count,
    default_motif_lists_path,
)
from audio2map.eval.motif import motif_key
from audio2map.eval.motif_freq import (
    SCOPES,
    MotifCorpus,
    corpus_distance,
    frequency_distance,
    load_ordered_motif_vocab,
    spearman,
)
from audio2map.eval.result import GeneratedChart
from audio2map.eval.score import score_chart
from audio2map.features.cond import ChartMeta
from audio2map.generate.overlap import GenerationRangeConfig, OverlapConfig
from audio2map.grid import CanonicalTiming, TICKS_PER_BAR, tick_to_ms
from audio2map.osu.export import write_osu
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.tokens import TOKEN_BAR, LaneState, tokens_to_notes


def _timing() -> CanonicalTiming:
    return CanonicalTiming(0, 120.0, 120.0, 0)


def _note(tick: int, col: int, *, end: int | None = None) -> ManiaNote:
    timing = _timing()
    if end is None:
        return ManiaNote(time_ms=tick_to_ms(tick, timing), col=col, note_type=NoteType.TAP)
    return ManiaNote(
        time_ms=tick_to_ms(tick, timing),
        col=col,
        note_type=NoteType.HOLD,
        end_time_ms=tick_to_ms(end, timing),
    )


def _meta() -> ChartMeta:
    return ChartMeta(osu_path="t.osu", canonical_bpm=120.0, official_sr=1.0)


def _key(**overrides) -> tuple:
    overlap = OverlapConfig()
    base = dict(
        backend="v0",
        relpath="a.osu",
        scenario="matched_full",
        mode="constrained",
        seed=0,
        cond_vec=np.zeros(4, dtype=np.float32),
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        range_mode="audio_full",
        pre_margin_bars=GenerationRangeConfig().pre_margin_bars,
        post_margin_bars=4,
        window_bars=overlap.window_bars,
        context_bars=overlap.context_bars,
        keep_bars=overlap.keep_bars,
        future_bars=overlap.future_bars,
        max_seq_len=overlap.max_seq_len,
    )
    base.update(overrides)
    return generation_cache_key(**base)


def test_cache_key_separates_scenarios_with_the_same_condition() -> None:
    matched = _key(scenario="matched_full")
    random_full = _key(scenario="random_full")
    assert matched != random_full
    assert _key(window_bars=16) != _key(window_bars=32)
    assert _key(pre_margin_bars=0) != _key(pre_margin_bars=2)


def test_diversity_seed_zero_reuses_the_matched_constrained_key() -> None:
    calls = {"n": 0}

    def produce():
        calls["n"] += 1
        return "notes"

    cache: dict = {}
    baseline = _key(scenario="matched_full", mode="constrained", seed=0)
    diversity = _key(scenario="matched_full", mode="constrained", seed=0)
    assert baseline == diversity
    assert cache_get_or_put(cache, baseline, produce) == "notes"
    assert cache_get_or_put(cache, diversity, produce) == "notes"
    assert calls["n"] == 1
    other = _key(scenario="random_full", mode="constrained", seed=0)
    assert cache_get_or_put(cache, other, produce) == "notes"
    assert calls["n"] == 2


def test_frozen_inputs_are_load_only(tmp_path: Path) -> None:
    missing = tmp_path / "random_full.json"
    with pytest.raises(DonorMapError, match="missing"):
        require_donor_map(missing, ["a.osu"], seed=2028)
    assert not missing.exists()
    subset = tmp_path / "diversity_subset.json"
    with pytest.raises(ChartSubsetError, match="missing"):
        require_chart_subset(subset, ["a.osu"], seed=2026, count=1)
    assert not subset.exists()


def test_require_donor_map_rejects_a_different_seed(tmp_path: Path) -> None:
    path = tmp_path / "random_full.json"
    save_donor_map(create_donor_map(["a.osu", "b.osu"], seed=1), path)
    with pytest.raises(DonorMapError, match="seed"):
        require_donor_map(path, ["a.osu"], seed=2028)
    loaded = require_donor_map(path, ["b.osu", "a.osu"], seed=1)
    assert loaded.pairs["a.osu"] == create_donor_map(["a.osu", "b.osu"], seed=1).pairs["a.osu"]


def test_require_subset_rejects_charts_outside_the_pool(tmp_path: Path) -> None:
    path = tmp_path / "diversity_subset.json"
    save_chart_subset(ChartSubset(seed=2026, charts=("gone.osu",)), path)
    with pytest.raises(ChartSubsetError, match="not in this eval pool"):
        require_chart_subset(path, ["stay.osu"], seed=2026, count=1)


def test_zero_count_motifs_stay_in_frequency() -> None:
    timing = _timing()
    notes = [_note(0, 0), _note(48, 0)]
    present = motif_key([LaneState.TAP, LaneState.TAP], [0, 48], 0, 2)
    absent = motif_key([LaneState.HOLD_START], [0], 0, 1)
    ordered = load_ordered_motif_vocab(
        {
            "lane": ["TAP/48/TAP", "HOLD_START"],
            "hand": ["EMPTY+TAP"],
            "row": ["1000"],
        }
    )
    corpus = MotifCorpus(ordered)
    corpus.add_chart(notes, timing)
    lane = corpus.frequency("lane")
    assert len(lane) == 2
    assert corpus.as_dict()["scopes"]["lane"]["vocabulary"] == 2
    assert corpus.as_dict()["scopes"]["lane"]["zero_count"] == 1
    assert lane[ordered.texts["lane"].index("HOLD_START")] == 0.0
    assert corpus.counts["lane"][absent] == 0
    assert corpus.counts["lane"][present] == 1
    assert corpus.observations["lane"] == 2
    assert lane[0] == 0.5


def test_hand_frequency_uses_the_pooled_observation_count() -> None:
    timing = _timing()
    notes = [_note(0, 1), _note(0, 2)]
    ordered = load_ordered_motif_vocab(
        {"lane": ["TAP"], "hand": ["EMPTY+TAP", "TAP+TAP"], "row": ["0100"]}
    )
    corpus = MotifCorpus(ordered)
    corpus.add_chart(notes, timing)
    assert corpus.observations["hand"] == 2
    assert corpus.counts["hand"][((LaneState.EMPTY, LaneState.TAP),)] == 2
    freq = corpus.frequency("hand")
    assert freq[0] == 1.0
    assert freq[1] == 0.0
    distance = frequency_distance(freq, [1.0, 0.0], ordered.texts["hand"])
    assert distance["vocabulary"] == 2
    assert distance["mae"] == 0.0


def test_frequency_distance_uses_the_full_list_including_zeros() -> None:
    texts = ("TAP", "HOLD_START", "HOLD_END")
    distance = frequency_distance([0.5, 0.0, 0.0], [0.0, 0.5, 0.0], texts)
    assert distance["vocabulary"] == 3
    assert distance["mae"] == pytest.approx(1.0 / 3.0)
    assert distance["largest_generated_excess"]["motif"] == "TAP"
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        frequency_distance([0.1], [0.1, 0.2], ("only",))


def test_load_ordered_vocab_rejects_extra_keys() -> None:
    with pytest.raises(ValueError, match="exactly"):
        load_ordered_motif_vocab({"lane": ["TAP"], "hand": ["EMPTY+TAP"], "row": ["1000"], "picked": ["TAP"]})


def test_missing_hold_tail_does_not_change_validity_or_eligibility(tmp_path: Path, monkeypatch) -> None:
    timing = _timing()
    decoded = tokens_to_notes([TOKEN_BAR, "<POS_0>", "<ROW_2000>"], timing, start_bar=0)
    assert decoded.issues[0].kind == "missing_hold_tail"
    validity = "valid"
    issues = [{"kind": issue.kind, "tick": issue.tick, "col": issue.col} for issue in decoded.issues]
    assert validity == "valid"
    assert enters_chart_metrics("constrained", validity)
    assert not enters_chart_metrics("unconstrained", validity)
    assert not enters_chart_metrics("unconstrained", "illegal_token")
    assert not enters_chart_metrics("unconstrained", "truncated")
    report = missing_hold_tail_report([issues, []])
    assert report["missing_hold_tail_count"] == 1
    assert report["missing_hold_tail_rate"] == 0.5

    source = tmp_path / "source.osu"
    write_osu(source, decoded.notes, original_bpm=120.0, offset_ms=0)
    chart = GeneratedChart(
        relpath="source.osu",
        path=source,
        notes=decoded.notes,
        timing=timing,
        reference_notes=decoded.notes,
        target_meta=_meta(),
        adherence_meta=_meta(),
        validity="valid",
        decode_issues=issues,
        scenario="matched_full",
        mode="constrained",
        seed=0,
    )
    monkeypatch.setattr(
        "audio2map.eval.score.compute_chart_meta",
        lambda path: _meta(),
    )
    scored = score_chart(chart, tmp_path / "gen.osu")
    assert scored is not None
    assert scored["validity"] == "valid"
    assert scored["decode_issues"][0]["kind"] == "missing_hold_tail"
    assert scored["match"]["note"]["lane_aware"]["hold"]["f1"] == 1.0
    blocked = GeneratedChart(
        relpath="source.osu",
        path=source,
        notes=decoded.notes,
        timing=timing,
        reference_notes=decoded.notes,
        target_meta=_meta(),
        adherence_meta=_meta(),
        validity="illegal_token",
        decode_issues=issues,
        mode="unconstrained",
    )
    assert score_chart(blocked, tmp_path / "unused.osu") is None
    assert not (tmp_path / "unused.osu").exists()


def test_seed_pairs_keep_matching_separate_from_exact_duplicates() -> None:
    timing = _timing()
    left = [_note(0, 0), _note(48, 1, end=96)]
    right = [_note(0, 0), _note(48, 1, end=144)]
    same = exact_duplicate_report([left, left], timing)
    assert same["exact_pair_count"] == 1
    mixed = seed_pair_report([left, right], timing)
    assert mixed["exact"]["exact_pair_count"] == 0
    pair = mixed["matching"][0]["match"]
    assert pair["note"]["lane_aware"]["hold"]["f1"] == 0.0
    assert pair["event"]["lane_aware"]["hold_head"]["f1"] == 1.0
    assert pair["note"]["lane_agnostic"]["all_notes"]["f1"] == 0.5


def test_v1_adapter_does_not_use_v0_generation_constraints() -> None:
    source = inspect.getsource(v1_backend)
    for banned in ("build_sample", "OverlapConfig", "audio_grid", "generate_chart_notes"):
        assert banned not in source
    chart = v1_backend.generated_chart(
        relpath="v1.osu",
        path=Path("v1.osu"),
        notes=[],
        timing=_timing(),
        reference_notes=[],
        target_meta=_meta(),
        adherence_meta=_meta(),
        validity="valid",
        scenario="matched_full",
        mode="constrained",
        seed=0,
    )
    assert chart.mode == "constrained"
    assert chart.notes == []


def test_hold_tail_clip_is_still_a_note() -> None:
    timing = _timing()
    decoded = tokens_to_notes([TOKEN_BAR, "<POS_0>", "<ROW_2000>"], timing, start_bar=0)
    assert decoded.notes[0].note_type == NoteType.HOLD
    assert decoded.notes[0].end_time_ms == tick_to_ms(TICKS_PER_BAR, timing)


def test_several_open_holds_count_per_lane_and_rate_per_chart() -> None:
    timing = _timing()
    decoded = tokens_to_notes([TOKEN_BAR, "<POS_0>", "<ROW_2200>"], timing, start_bar=0)
    tails = [issue for issue in decoded.issues if issue.kind == "missing_hold_tail"]
    assert {issue.col for issue in tails} == {0, 1}
    issues = [{"kind": issue.kind, "tick": issue.tick, "col": issue.col} for issue in tails]
    report = missing_hold_tail_report([issues, []])
    assert report["missing_hold_tail_count"] == 2
    assert report["charts"] == 2
    assert report["missing_hold_tail_rate"] == 0.5
    assert enters_chart_metrics("constrained", "valid")


def test_avg_count_is_not_a_frequency_alias() -> None:
    ordered = load_ordered_motif_vocab(
        {"lane": ["TAP", "HOLD_END"], "hand": ["EMPTY+TAP"], "row": ["1000"]}
    )
    corpus = MotifCorpus(ordered)
    corpus.add_chart([_note(0, 0), _note(48, 0)], _timing())
    assert corpus.observations["lane"] == 2
    assert corpus.frequency("lane") == [1.0, 0.0]
    assert corpus.avg_count("lane") == [2.0, 0.0]
    assert ordered.texts["lane"][0] == "TAP"


def test_frequency_follows_frozen_list_order() -> None:
    ordered = load_ordered_motif_vocab(
        {"lane": ["TAP", "HOLD_END"], "hand": ["EMPTY+TAP"], "row": ["1000"]}
    )
    corpus = MotifCorpus(ordered)
    corpus.add_chart([_note(0, 0)], _timing())
    assert corpus.frequency("lane")[0] == 1.0
    assert corpus.frequency("lane")[1] == 0.0


def test_spearman_uses_average_ranks_and_rejects_constant_vectors() -> None:
    assert spearman([1.0, 1.0, 2.0], [1.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert spearman([1.0, 1.0, 2.0], [2.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert spearman([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) is None
    assert spearman([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) is None


def test_zero_observations_do_not_become_a_zero_distance() -> None:
    ordered = load_ordered_motif_vocab(
        {"lane": ["TAP", "HOLD_END"], "hand": ["EMPTY+TAP"], "row": ["1000"]}
    )
    empty = MotifCorpus(ordered)
    empty.add_chart([], _timing())
    present = MotifCorpus(ordered)
    present.add_chart([_note(0, 0)], _timing())
    assert empty.as_dict()["scopes"]["lane"]["frequency_defined"] is False
    assert empty.frequency("lane") == [0.0, 0.0]
    distance = corpus_distance(empty, present)
    assert distance["lane"]["mae"] is None
    assert distance["lane"]["rmse"] is None
    assert distance["lane"]["spearman"] is None
    assert distance["lane"]["observations_generated"] == 0
    assert distance["lane"]["observations_human"] == 1
    defined = corpus_distance(present, present)
    assert defined["lane"]["mae"] == 0.0


def test_empty_length_bucket_is_undefined() -> None:
    distance = frequency_distance([1.0], [0.0], ("TAP",))
    assert distance["length_buckets"]["1"] == 1.0
    assert distance["length_buckets"]["2"] is None


def test_short_motif_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="50000"):
        assert_motif_list_count({"lane": ["TAP"], "hand": ["EMPTY+TAP"], "row": ["1000"]})


def test_frozen_motif_lists_are_50000_in_file_order() -> None:
    path = default_motif_lists_path()
    if not path.is_file():
        pytest.skip("frozen motif lists are not in this environment")
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert_motif_list_count(data, MOTIF_LIST_COUNT)
    loaded = load_ordered_motif_vocab(data)
    for scope in ("lane", "hand", "row"):
        assert loaded.texts[scope] == tuple(data[scope])
        assert len(loaded.keys[scope]) == MOTIF_LIST_COUNT
        assert list(loaded.texts[scope]) != sorted(loaded.texts[scope])


def test_materialization_failure_is_not_dropped(tmp_path: Path) -> None:
    kept = tmp_path / "a.osu"
    dropped = tmp_path / "b.osu"
    prepared = [SimpleNamespace(path=kept)]
    with pytest.raises(ValueError, match="failed"):
        assert_materialization_complete([kept, dropped], prepared)  # type: ignore[list-item]
    assert_materialization_complete([kept], prepared)  # type: ignore[list-item]


def test_official_pool_must_match_the_donor_map_exactly() -> None:
    mapping = create_donor_map(["a.osu", "b.osu"], seed=2028)
    assert_exact_donor_universe(mapping, ["b.osu", "a.osu"])
    with pytest.raises(DonorMapError, match="universe"):
        assert_exact_donor_universe(mapping, ["a.osu"])


def test_unconstrained_scope_is_matched_full_seed_zero(tmp_path: Path) -> None:
    path = tmp_path / "a.osu"
    cond = np.zeros(4, dtype=np.float32)
    by_rel = {"a.osu": (path, _meta(), cond)}
    issues = [{"kind": "missing_hold_tail", "tick": 0, "col": 0}]

    def run(_path, _cond, mode, seed, scenario):
        return [], "illegal_token", _timing(), issues

    with pytest.raises(ValueError, match="matched_full"):
        _generation_body(
            run,
            [(path, _meta(), cond)],
            scenarios=("matched_full", "random_full"),
            modes=("unconstrained",),
            seeds=(0,),
            donor=None,
            root=tmp_path,
            by_rel=by_rel,
            diversity_rels=None,
            unconstrained_rels=["a.osu"],
            vocab=None,
            ordered_vocab=None,
            adherence_root=tmp_path,
        )
    with pytest.raises(ValueError, match="seed 0"):
        _generation_body(
            run,
            [(path, _meta(), cond)],
            scenarios=("matched_full",),
            modes=("unconstrained",),
            seeds=(0, 1),
            donor=None,
            root=tmp_path,
            by_rel=by_rel,
            diversity_rels=None,
            unconstrained_rels=["a.osu"],
            vocab=None,
            ordered_vocab=None,
            adherence_root=tmp_path,
        )
    out = _generation_body(
        run,
        [(path, _meta(), cond)],
        scenarios=("matched_full",),
        modes=("unconstrained",),
        seeds=(0,),
        donor=None,
        root=tmp_path,
        by_rel=by_rel,
        diversity_rels=None,
        unconstrained_rels=["a.osu"],
        vocab=None,
        ordered_vocab=None,
        adherence_root=tmp_path,
    )
    row = out["scenarios"]["matched_full"]["unconstrained"][0]
    assert row["charts"][0]["validity"] == "illegal_token"
    assert row["charts"][0]["decode_issues"] == issues
    assert "missing_hold_tail" not in row
    assert "statistics" not in row


def test_diversity_uses_fixed_seeds_and_reuses_seed_zero(monkeypatch, tmp_path: Path) -> None:
    calls = {"n": 0}

    def fake_generate_v0(*_args, **_kwargs):
        calls["n"] += 1
        return [], "valid", [{"kind": "missing_hold_tail", "tick": 1, "col": 0}]

    monkeypatch.setattr(chart_eval, "generate_v0", fake_generate_v0)
    monkeypatch.setattr(chart_eval, "parse_beatmap", lambda path: SimpleNamespace(notes=[]))
    monkeypatch.setattr(chart_eval.CanonicalTiming, "from_beatmap", lambda _beatmap: _timing())
    monkeypatch.setattr(
        chart_eval,
        "score_chart",
        lambda chart, _path: {
            "osu": str(chart.path),
            "donor": chart.donor,
            "validity": chart.validity,
            "decode_issues": list(chart.decode_issues),
            "match": {},
            "adherence": {},
        },
    )
    path = tmp_path / "a.osu"
    path.write_bytes(b"")
    cond = np.zeros(18, dtype=np.float32)
    report = eval_generation(
        object(),  # type: ignore[arg-type]
        [(path, _meta(), cond)],
        device=object(),  # type: ignore[arg-type]
        scenarios=("matched_full",),
        modes=("constrained",),
        seeds=(0,),
        raw_root=tmp_path,
        diversity_rels=["a.osu"],
    )
    assert DIVERSITY_GENERATION_SEEDS == (0, 1, 2, 3, 4)
    assert calls["n"] == 1 + len(DIVERSITY_GENERATION_SEEDS) - 1
    diversity = report["diversity"]["matched_full"][0]
    assert len(diversity["decode_issues"]) == len(DIVERSITY_GENERATION_SEEDS)
    assert diversity["decode_issues"][0][0]["kind"] == "missing_hold_tail"
    assert len(report["scenarios"]["matched_full"]["constrained"]) == 1
    assert report["scenarios"]["matched_full"]["constrained"][0]["seed"] == 0


def test_official_cli_does_not_create_frozen_inputs() -> None:
    import audio2map.cli.eval as cli_eval

    source = inspect.getsource(cli_eval)
    for banned in (
        "load_or_create_donor_map",
        "load_or_create_chart_subset",
        "sample_chart_subset",
        "create_donor_map",
    ):
        assert banned not in source
    assert "args.diversity_seed" not in source


def test_motif_scope_names_match() -> None:
    assert MOTIF_SCOPES == SCOPES


def test_diversity_membership_fails_before_generation(tmp_path: Path) -> None:
    calls = {"n": 0}

    def run(*_args, **_kwargs):
        calls["n"] += 1
        return [], "valid", _timing(), []

    with pytest.raises(ValueError, match="not in this eval pool"):
        _generation_body(
            run,
            [],
            scenarios=("matched_full",),
            modes=("constrained",),
            seeds=(0,),
            donor=None,
            root=tmp_path,
            by_rel={},
            diversity_rels=["missing.osu"],
            unconstrained_rels=None,
            vocab=None,
            ordered_vocab=None,
            adherence_root=tmp_path,
        )
    assert calls["n"] == 0


def test_motif_distance_requires_observation_counts() -> None:
    vocab = load_ordered_motif_vocab({"lane": ["TAP"], "hand": ["EMPTY+TAP"], "row": ["1000"]})
    with pytest.raises(ValueError, match="observation"):
        seed_pair_report(
            [[], []],
            _timing(),
            frequencies=[
                {"lane": [0.0], "hand": [0.0], "row": [0.0]},
                {"lane": [0.0], "hand": [0.0], "row": [0.0]},
            ],
            vocab=vocab,
        )
