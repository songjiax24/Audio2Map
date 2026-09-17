"""Tests for note matching and split token accuracy."""

from __future__ import annotations

import torch

from audio2map.eval.note_match import compare_note_lists
from audio2map.eval.token_accuracy import SplitTokenAccuracy, split_token_accuracy
from audio2map.grid import CanonicalTiming
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.tokens import (
    TOKEN_BAR,
    LaneState,
    build_vocab,
    invert_vocab,
    row_state_to_token,
)


def test_exact_match_f1() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    notes = [
        ManiaNote(time_ms=0, col=0, note_type=NoteType.TAP),
        ManiaNote(time_ms=0, col=1, note_type=NoteType.HOLD, end_time_ms=100),
    ]
    stats = compare_note_lists(notes, notes, timing)
    assert stats.f1 == 1.0
    assert stats.tp == 2


def test_partial_match() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    exp = [ManiaNote(time_ms=0, col=0, note_type=NoteType.TAP)]
    pred = [
        ManiaNote(time_ms=0, col=0, note_type=NoteType.TAP),
        ManiaNote(time_ms=10, col=1, note_type=NoteType.TAP),
    ]
    stats = compare_note_lists(pred, exp, timing)
    assert stats.tp == 1 and stats.fp == 1 and stats.fn == 0
    assert stats.precision == 0.5 and stats.recall == 1.0


def test_duplicate_notes_are_counted() -> None:
    timing = CanonicalTiming(0, 120.0, 120.0, 0)
    tap = ManiaNote(time_ms=0, col=0, note_type=NoteType.TAP)
    stats = compare_note_lists([tap, tap], [tap, tap], timing)
    assert stats.tp == 2
    stats = compare_note_lists([tap], [tap, tap], timing)
    assert stats.tp == 1 and stats.fn == 1 and stats.fp == 0
    stats = compare_note_lists([tap, tap], [tap], timing)
    assert stats.tp == 1 and stats.fp == 1 and stats.fn == 0


def test_hold_recall_is_per_lane() -> None:
    vocab = build_vocab()
    id_to_token = invert_vocab(vocab)
    empty = (LaneState.EMPTY, LaneState.EMPTY, LaneState.EMPTY, LaneState.EMPTY)
    gt_start = row_state_to_token((LaneState.HOLD_START, *empty[1:]))
    wrong_lane = row_state_to_token((*empty[:3], LaneState.HOLD_START))
    tgt = torch.tensor([vocab[gt_start]])
    mask = torch.tensor([1.0])

    miss = split_token_accuracy(
        torch.tensor([vocab[wrong_lane]]), tgt, mask, id_to_token=id_to_token
    )
    assert miss.hold_start_recall_den == 1
    assert miss.hold_start_recall_num == 0
    assert miss.row_lane_correct == 2

    hit = split_token_accuracy(
        torch.tensor([vocab[gt_start]]), tgt, mask, id_to_token=id_to_token
    )
    assert hit.hold_start_recall_num == 1
    assert hit.row_lane_correct == 4
    assert hit.row_correct == 1

    gt_end = row_state_to_token((LaneState.HOLD_END, *empty[1:]))
    end_tgt = torch.tensor([vocab[gt_end]])
    end_miss = split_token_accuracy(
        torch.tensor([vocab[TOKEN_BAR]]), end_tgt, mask, id_to_token=id_to_token
    )
    assert end_miss.hold_end_recall_den == 1
    assert end_miss.hold_end_recall_num == 0
    assert end_miss.row_lane_correct == 0
    assert end_miss.row_correct == 0


def test_split_token_empty_rates_are_zero() -> None:
    assert SplitTokenAccuracy().to_dict()["hold_start_recall"] == 0.0
    assert SplitTokenAccuracy().to_dict()["bar_acc"] == 0.0
