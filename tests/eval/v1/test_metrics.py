from dataclasses import replace

from audio2map.eval.v1.metrics import (
    condition_error,
    diversity,
    endpoint_match,
    event_distribution,
    plausibility,
)
from audio2map.features.cond.chart import ChartMeta
from audio2map.features.cond.v1.vec import COND_SOURCE_FIELDS
from audio2map.generate.v1.notes import V1Note
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1.codec import decode_tokens, encode_chart


def encoded():
    return encode_chart(
        [ManiaNote(0, 0, NoteType.TAP), ManiaNote(500, 1, NoteType.HOLD, end_time_ms=750)],
        [Redline(0.0, 500.0, 4, 0)],
        1000.0,
    )


def test_matching_uses_grid_identity_and_multisets():
    chart = encoded()
    events = decode_tokens(chart.tokens, chart.grid)
    shifted = [replace(event, time_ms=event.time_ms + 0.5) for event in events]
    assert endpoint_match(shifted, events)["all"]["f1"] == 1
    alternate = [replace(events[0], section=1)] + events[1:]
    assert endpoint_match(alternate, events)["all"]["tp"] == 2
    duplicate = events + [events[0]]
    assert endpoint_match(duplicate, events)["all"]["fp"] == 1
    assert endpoint_match(events, events)["by_kind"]["tail"]["tp"] == 1


def test_plausibility_thresholds_are_independent_and_inclusive_where_required():
    notes = (
        V1Note(0, 0, 100),
        V1Note(102, 0, None),
        V1Note(111, 0, None),
        V1Note(121, 0, None),
        V1Note(5, 1, 5),
        V1Note(5, 1, None),
        V1Note(0, 2, 1000),
        V1Note(200, 2, None),
    )
    score = plausibility(notes)
    assert score["concurrent"] == 3
    assert score["almost_concurrent"] == 3
    assert score["duplicate_start"] == 1
    assert score["nonpositive_hold"] == 1
    assert score["has_concurrent"] and score["has_almost_concurrent"]


def test_distributions_keep_hold_endpoints_and_exact_spans():
    chart = encoded()
    events = decode_tokens(chart.tokens, chart.grid)
    stats = event_distribution(events, chart.grid)
    assert stats["endpoints"] == 3
    assert stats["lanes"]["1"] == {"tap": 0, "head": 1, "tail": 1}
    assert stats["denominator"] == {"1": 2, "2": 1}
    assert stats["hold_beat_span"] == {"1/2": 1}
    assert sum(stats["hold_grid_span"].values()) == 1


def test_missing_condition_errors_are_not_zero_and_diversity_counts_identical_pairs():
    assert condition_error(ChartMeta("a"), ChartMeta("b", official_sr=5))["official_sr"] is None
    assert (
        condition_error(ChartMeta("a", official_sr=3), ChartMeta("b", official_sr=5))["official_sr"]
        == 2
    )
    chart = encoded()
    events = decode_tokens(chart.tokens, chart.grid)
    score = diversity({0: events, 1: events, 2: events[:-1]})
    assert len(score["pairs"]) == 3
    assert score["identical_pair_rate"] == 1 / 3


def test_all_eighteen_original_fields_enter_condition_error():
    target = ChartMeta("target", **{name: 0.0 for name in COND_SOURCE_FIELDS})
    generated = ChartMeta("generated", **{name: 1.0 for name in COND_SOURCE_FIELDS})
    errors = condition_error(target, generated)
    assert len(errors) == 18
    assert set(errors.values()) == {1.0}


def test_lane_agnostic_matching_preserves_chord_multiplicity_and_complete_holds():
    chart = encoded()
    events = decode_tokens(chart.tokens, chart.grid)
    moved = [replace(event, lane=(event.lane + 2) % 4) for event in events]
    result = endpoint_match(moved, events)
    assert result["event"]["lane_aware"]["all"]["f1"] == 0
    assert result["event"]["lane_agnostic"]["all"]["f1"] == 1
    assert result["note"]["lane_agnostic"]["hold"]["f1"] == 1
    tail_changed = events[:-1] + [replace(events[-1], numer=3, denom=4)]
    result = endpoint_match(tail_changed, events)
    assert result["event"]["lane_aware"]["head"]["tp"] == 1
    assert result["note"]["lane_aware"]["hold"]["tp"] == 0
    chord = [events[0], replace(events[0], lane=1)]
    result = endpoint_match(chord[:1], chord)
    assert result["event"]["lane_agnostic"]["all"]["fn"] == 1
    assert result["note"]["lane_agnostic"]["all"]["fn"] == 1


def test_distribution_includes_row_states_and_counts_holds_once():
    chart = encoded()
    stats = event_distribution(decode_tokens(chart.tokens, chart.grid), chart.grid, chart.tokens)
    assert stats["notes"] == 2 and stats["holds"] == 1 and stats["taps"] == 1
    assert stats["row_pattern"] == {"1000": 1, "0200": 1, "0400": 1}
    assert sum(stats["position"].values()) == stats["endpoints"]
