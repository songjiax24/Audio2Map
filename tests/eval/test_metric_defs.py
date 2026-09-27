"""Locked eval semantics: matching scope, pooled frequency, motif counts."""

from __future__ import annotations

from audio2map.eval.adherence import condition_mae
from audio2map.eval.chart_stats import ChartStatAccumulator, snap_name
from audio2map.eval.motif import (
    MotifAccumulator,
    MotifSequences,
    mirror_hand_symbol,
    motif_key,
)
from audio2map.eval.note_match import compare_note_lists, reference_match
from audio2map.features.cond import USER_COND_SOURCE_FIELDS
from audio2map.grid import CanonicalTiming, tick_to_ms
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.tokens import EVENT_LANE_STATES, LaneState, chart_event_rows


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


def test_lane_agnostic_drops_column_in_one_multiset() -> None:
    timing = _timing()
    # 1100 vs 0011 at the same tick: two taps, different lanes.
    left = [_note(0, 0), _note(0, 1)]
    right = [_note(0, 2), _note(0, 3)]
    matched = reference_match(left, right, timing)
    assert matched["event"]["lane_aware"]["all_event"]["f1"] == 0.0
    assert matched["event"]["lane_agnostic"]["tap"]["f1"] == 1.0
    assert matched["event"]["lane_agnostic"]["all_event"]["f1"] == 1.0
    assert matched["note"]["lane_agnostic"]["all_notes"]["f1"] == 1.0
    assert matched["note"]["lane_aware"]["all_notes"]["f1"] == 0.0


def test_all_is_the_union_of_atomic_identities() -> None:
    timing = _timing()
    notes = [_note(0, 0), _note(48, 1, end=96)]
    matched = reference_match(notes, notes, timing)
    assert matched["event"]["lane_aware"]["tap"]["tp"] == 1
    assert matched["event"]["lane_aware"]["hold_head"]["tp"] == 1
    assert matched["event"]["lane_aware"]["hold_tail"]["tp"] == 1
    assert matched["event"]["lane_aware"]["all_event"]["tp"] == 3
    assert matched["note"]["lane_aware"]["all_notes"]["tp"] == 2
    assert compare_note_lists(notes, notes, timing).f1 == matched["note"]["lane_aware"]["all_notes"]["f1"]


def test_pooled_frequency_uses_the_whole_event_universe() -> None:
    timing = _timing()
    stats = ChartStatAccumulator()
    stats.add_chart([_note(0, 0), _note(0, 1)], timing)
    report = stats.to_dict()
    freq = report["tick"]["pooled_frequency"]
    assert freq["lane_1"]["tap"]["0"] == 0.5
    assert freq["lane_2"]["tap"]["0"] == 0.5
    assert freq["all_lanes"]["tap"]["0"] == 1.0
    atomic = 0.0
    for lane in ("lane_1", "lane_2", "lane_3", "lane_4"):
        for state in ("tap", "hold_head", "hold_tail"):
            atomic += sum(freq[lane][state].values())
    assert abs(atomic - 1.0) < 1e-9
    assert snap_name(0) == "1/1"
    assert report["snap"]["pooled_frequency"]["lane_1"]["tap"]["1/1"] == 0.5
    assert report["note"]["pooled_frequency"]["lane_1"]["tap"] == 0.5
    assert report["note"]["pooled_frequency"]["all_lanes"]["all_notes"] == 1.0


def test_lane_motif_skips_other_lanes_and_pooled_count_is_a_sum() -> None:
    timing = _timing()
    notes = [_note(10, 0), _note(20, 1), _note(30, 0)]
    seqs = MotifSequences.from_notes(notes, timing)
    assert [tick for tick, _state in seqs.lane[0]] == [10, 30]
    assert [tick for tick, _state in seqs.lane[1]] == [20]
    key = motif_key([LaneState.TAP, LaneState.TAP], [10, 30], 0, 2)
    acc = MotifAccumulator({"lane": {key}, "hand": set(), "row": set()})
    acc.add_chart(notes, timing)
    report = acc.to_dict()
    assert report["lane"]["lane_1"]["TAP/20/TAP"] == 1.0
    assert report["lane"]["lane_2"]["TAP/20/TAP"] == 0.0
    assert report["all_lanes"]["TAP/20/TAP"] == 1.0
    assert "frequency" not in report


def test_hand_mirror_pools_counts_on_the_original_hand() -> None:
    assert mirror_hand_symbol((LaneState.TAP, LaneState.EMPTY)) == (LaneState.EMPTY, LaneState.TAP)
    timing = _timing()
    # Hand 12 at tick 0 is (EMPTY, TAP). Hand 34 (TAP, EMPTY) mirrors to the same identity.
    notes = [_note(0, 1), _note(0, 2)]
    key = ((LaneState.EMPTY, LaneState.TAP),)
    acc = MotifAccumulator({"lane": set(), "hand": {key}, "row": set()})
    acc.add_chart(notes, timing)
    report = acc.to_dict()
    assert report["hand"]["12"]["EMPTY+TAP"] == 1.0
    assert report["hand"]["34"]["EMPTY+TAP"] == 1.0
    assert report["both_hands"]["EMPTY+TAP"] == 2.0


def test_hold_end_is_part_of_note_identity() -> None:
    timing = _timing()
    left = [_note(0, 0, end=48)]
    right = [_note(0, 0, end=96)]
    matched = reference_match(left, right, timing)
    assert matched["note"]["lane_aware"]["hold"]["f1"] == 0.0
    assert matched["note"]["lane_agnostic"]["all_notes"]["f1"] == 0.0
    assert matched["event"]["lane_aware"]["hold_head"]["f1"] == 1.0
    assert matched["event"]["lane_aware"]["hold_tail"]["f1"] == 0.0


def test_marginals_are_sums_of_atomic_counts() -> None:
    timing = _timing()
    notes = [_note(0, 0), _note(1, 1), _note(2, 2), _note(3, 3), _note(4, 0, end=52)]
    report = ChartStatAccumulator()
    report.add_chart(notes, timing)
    freq = report.to_dict()
    tick = freq["tick"]["pooled_frequency"]
    atomic = 0.0
    for lane in ("lane_1", "lane_2", "lane_3", "lane_4"):
        for state in ("tap", "hold_head", "hold_tail"):
            atomic += sum(tick[lane][state].values())
    assert abs(atomic - 1.0) < 1e-9
    for q, value in tick["all_lanes"]["tap"].items():
        lane_sum = sum(tick[lane]["tap"].get(q, 0.0) for lane in ("lane_1", "lane_2", "lane_3", "lane_4"))
        assert abs(lane_sum - value) < 1e-9
    for q, value in tick["lane_1"]["all_event"].items():
        state_sum = sum(tick["lane_1"][state].get(q, 0.0) for state in ("tap", "hold_head", "hold_tail"))
        assert abs(state_sum - value) < 1e-9
    note = freq["note"]["pooled_frequency"]
    assert abs(note["all_lanes"]["tap"] + note["all_lanes"]["hold"] - note["all_lanes"]["all_notes"]) < 1e-9
    lane_notes = sum(note[lane]["all_notes"] for lane in ("lane_1", "lane_2", "lane_3", "lane_4"))
    assert abs(lane_notes - note["all_lanes"]["all_notes"]) < 1e-9
    assert abs(note["all_lanes"]["all_notes"] - 1.0) < 1e-9


def test_motif_scopes_follow_their_own_events() -> None:
    timing = _timing()
    notes = [_note(10, 0), _note(20, 1), _note(30, 3)]
    seqs = MotifSequences.from_notes(notes, timing)
    rows = chart_event_rows(notes, timing)
    assert [tick for tick, _row in seqs.row] == sorted(rows)
    assert [tick for tick, _state in seqs.lane[0]] == [10]
    assert [tick for tick, _state in seqs.hand["12"]] == [10, 20]
    assert [tick for tick, _state in seqs.hand["34"]] == [30]
    assert all(state in EVENT_LANE_STATES for _tick, state in seqs.lane[0])
    assert seqs.hand["34"][0][1] == (LaneState.EMPTY, LaneState.TAP)


def test_adherence_keeps_missing_values_and_ignores_bpm_slot() -> None:
    names = ("official_sr", "analyzer_ln_percent")
    mae = condition_mae(
        {"official_sr": 5.0, "analyzer_ln_percent": None},
        {"official_sr": 4.0, "analyzer_ln_percent": 0.2},
        names,
    )
    assert mae["official_sr"] == 1.0
    assert mae["analyzer_ln_percent"] is None
    assert "canonical_bpm_norm" not in USER_COND_SOURCE_FIELDS
    assert "canonical_bpm" not in USER_COND_SOURCE_FIELDS
