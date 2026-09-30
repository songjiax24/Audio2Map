"""Exhaustive checks for global Top-K motif fitting."""

from __future__ import annotations

import json
from collections import Counter

from audio2map.eval.chart_eval import _motif_vocab_from_json
from audio2map.eval.motif import format_motif, motif_key, parse_motif
from audio2map.eval.motif_fit import (
    TIE_BREAK,
    extend_motif_key,
    fit_scope,
    fit_scopes,
    scope_sequences,
    write_vocab_files,
)
from audio2map.grid import CanonicalTiming, tick_to_ms
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.tokens import LaneState


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


def _pair(left: LaneState, right: LaneState):
    return (left, right)


def _sequence(symbols: list, gaps: list[int]) -> tuple[list, list[int]]:
    ticks = [0]
    for gap in gaps:
        ticks.append(ticks[-1] + gap)
    assert len(ticks) == len(symbols)
    return symbols, ticks


def _brute(sequences: list[tuple[list, list[int]]], k: int) -> list[tuple[tuple, int]]:
    counts: Counter[tuple] = Counter()
    for symbols, ticks in sequences:
        n = len(symbols)
        for length in range(1, n + 1):
            for start in range(n - length + 1):
                counts[motif_key(symbols, ticks, start, length)] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], format_motif(item[0])))
    return ranked[:k]


def _assert_matches_brute(sequences: list[tuple[list, list[int]]], ks: tuple[int, ...]) -> None:
    largest = max(ks)
    fitted_large = fit_scope(sequences, k_fit=largest)
    brute_large = _brute(sequences, largest)
    assert [(item.key, item.occurrence) for item in fitted_large] == brute_large
    for k in ks:
        fitted = fit_scope(sequences, k_fit=k)
        assert [(item.key, item.occurrence) for item in fitted] == brute_large[:k]
        assert [(item.key, item.occurrence) for item in fitted_large[:k]] == [
            (item.key, item.occurrence) for item in fitted
        ]


def test_extension_sorts_after_its_prefix_and_matches_motif_key() -> None:
    lane_symbols = [LaneState.TAP, LaneState.HOLD_START, LaneState.HOLD_END, LaneState.TAP]
    hand_symbols = [
        _pair(LaneState.TAP, LaneState.EMPTY),
        _pair(LaneState.EMPTY, LaneState.HOLD_END),
        _pair(LaneState.HOLD_START, LaneState.TAP),
    ]
    row_symbols = [
        (LaneState.TAP, LaneState.EMPTY, LaneState.EMPTY, LaneState.EMPTY),
        (LaneState.EMPTY, LaneState.HOLD_START, LaneState.EMPTY, LaneState.HOLD_END),
        (LaneState.HOLD_END, LaneState.TAP, LaneState.TAP, LaneState.EMPTY),
    ]
    for symbols in (lane_symbols, hand_symbols, row_symbols):
        ticks = [48 * index for index in range(len(symbols))]
        for length in range(1, len(symbols)):
            for start in range(len(symbols) - length):
                parent = motif_key(symbols, ticks, start, length)
                child = motif_key(symbols, ticks, start, length + 1)
                gap = ticks[start + length] - ticks[start + length - 1]
                assert extend_motif_key(parent, gap, symbols[start + length]) == child
                assert format_motif(parent) < format_motif(child)
                assert (-1, format_motif(parent)) < (-1, format_motif(child))


def test_overlapping_containment_and_ties_match_exhaustive_counts() -> None:
    taps = [LaneState.TAP] * 4
    sequences = [_sequence(taps, [48, 48, 48])]
    _assert_matches_brute(sequences, (1, 2, 3, 5))
    fitted = fit_scope(sequences, k_fit=5)
    by_text = {item.text(): item for item in fitted}
    assert by_text["TAP"].occurrence == 4
    assert by_text["TAP/48/TAP"].occurrence == 3
    assert by_text["TAP"].occurrence >= by_text["TAP/48/TAP"].occurrence
    tap_rank = [item.text() for item in fitted].index("TAP")
    child_rank = [item.text() for item in fitted].index("TAP/48/TAP")
    assert tap_rank < child_rank

    tied = [
        _sequence([LaneState.HOLD_END] * 3, [1000, 2000]),
        _sequence([LaneState.HOLD_START] * 3, [3000, 4000]),
    ]
    _assert_matches_brute(tied, (1, 2))
    top = fit_scope(tied, k_fit=2)
    assert [item.text() for item in top] == ["HOLD_END", "HOLD_START"]
    assert top[0].occurrence == top[1].occurrence == 3


def test_lane_pooling_sums_columns_before_ranking() -> None:
    pattern = _sequence([LaneState.TAP, LaneState.HOLD_END], [24])
    sequences = [pattern, pattern]
    _assert_matches_brute(sequences, (1, 2, 4))
    fitted = fit_scope(sequences, k_fit=2)
    assert [item.text() for item in fitted] == ["HOLD_END", "TAP"]
    assert [item.occurrence for item in fitted] == [2, 2]


def test_global_count_keeps_a_motif_each_chart_would_prune() -> None:
    def chart(symbols: list) -> tuple[list, list[int]]:
        return _sequence(symbols, [1000 * (index + 1) for index in range(len(symbols) - 1)])

    hold_start = _pair(LaneState.HOLD_START, LaneState.EMPTY)
    hold_end = _pair(LaneState.HOLD_END, LaneState.EMPTY)
    tap = _pair(LaneState.TAP, LaneState.EMPTY)
    empty_tap = _pair(LaneState.EMPTY, LaneState.TAP)
    shared = _pair(LaneState.EMPTY, LaneState.HOLD_END)
    chart_a = chart([hold_start] * 5 + [tap] * 4 + [shared] * 2)
    chart_b = chart([hold_end] * 5 + [empty_tap] * 4 + [shared] * 2)
    combined = [chart_a, chart_b]
    _assert_matches_brute(combined, (1, 2, 3, 5))

    local_a = {item.text() for item in fit_scope([chart_a], k_fit=2)}
    local_b = {item.text() for item in fit_scope([chart_b], k_fit=2)}
    global_top = [item.text() for item in fit_scope(combined, k_fit=3)]
    assert "EMPTY+HOLD_END" not in local_a
    assert "EMPTY+HOLD_END" not in local_b
    assert "EMPTY+HOLD_END" in global_top
    assert global_top[:3] == ["HOLD_END+EMPTY", "HOLD_START+EMPTY", "EMPTY+HOLD_END"]


def test_notes_cover_mirror_row_and_round_trip(tmp_path) -> None:
    notes = [
        _note(0, 1),
        _note(0, 2),
        _note(48, 0),
        _note(48, 1),
        _note(96, 0),
        _note(96, 1),
        _note(144, 0, end=192),
    ]
    scopes = scope_sequences_from_notes(notes)
    for name in ("lane", "hand", "row"):
        _assert_matches_brute(scopes[name], (1, 3, 8))

    hand = {item.text(): item.occurrence for item in fit_scope(scopes["hand"], k_fit=8)}
    assert hand["EMPTY+TAP"] == 2

    row = {item.text(): item.occurrence for item in fit_scope(scopes["row"], k_fit=8)}
    assert row["1100/48/1100"] == 1

    fit = fit_scopes(scopes, k_fit=8)
    for item in fit.lane + fit.hand + fit.row:
        assert parse_motif(format_motif(item.key)) == item.key

    vocab_path = tmp_path / "vocab.json"
    sidecar_path = tmp_path / "sidecar.json"
    corpus = {"charts": 1, "split": "synthetic"}
    write_vocab_files(fit, corpus, vocab_path=vocab_path, sidecar_path=sidecar_path)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    assert set(vocab) == {"lane", "hand", "row"}
    parsed = _motif_vocab_from_json(vocab)
    assert parsed is not None
    assert {format_motif(key) for key in parsed["hand"]} == set(vocab["hand"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["k_fit"] == 8
    assert sidecar["tie_break"] == TIE_BREAK
    assert sidecar["l_max"] is None
    assert sidecar["containment"] == "retained"
    assert sidecar["corpus"] == corpus
    assert [row["motif"] for row in sidecar["lane"]] == vocab["lane"]
    assert sidecar["lane"][0]["occurrence"] >= sidecar["lane"][-1]["occurrence"]
    assert "length" in sidecar["row"][0]


def scope_sequences_from_notes(notes: list[ManiaNote]) -> dict[str, list]:
    from audio2map.eval.motif import MotifSequences

    return scope_sequences(MotifSequences.from_notes(notes, _timing()))
