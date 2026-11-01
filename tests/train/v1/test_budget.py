"""Rate-1 window packing. plain CE is a token mean."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from audio2map.dataset.v1.shape import chart_shape
from audio2map.features.audio.v1.mel import frame_index
from audio2map.model.v1.loss import accuracy_counts, ce_by_type, nonpad_count, plain_ce
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1.codec import encode_chart
from audio2map.tokens.v1.vocab import TOKEN_BOS, TOKEN_PAD, build_vocab
from audio2map.train.v1.batches import (
    FRAME_BUDGET,
    cut_windows,
    epoch_batches,
    pack_in_order,
)
from audio2map.train.v1.lengths import ChartLength, _read_cache, _write_cache


def test_plain_ce_is_the_mean_over_tokens() -> None:
    vocab = len(build_vocab())
    logits = torch.zeros(2, 4, vocab)
    targets = torch.randint(0, vocab, (2, 4))
    loss = plain_ce(logits, targets)
    assert abs(float(loss) - math.log(vocab)) < 1e-4
    pad = build_vocab()[TOKEN_PAD]
    counted = torch.full((2, 4), pad + 1)
    counted[0, 0] = pad
    assert nonpad_count(counted) == 7


def test_accuracy_ignores_pad_and_splits_gold_types() -> None:
    vocab = build_vocab()
    width = len(vocab)
    pos = next(index for token, index in vocab.items() if token.startswith("<POS_"))
    row = next(index for token, index in vocab.items() if token.startswith("<ROW_"))
    beat = vocab["<BEAT>"]
    eos = vocab["<EOS>"]
    pad = vocab[TOKEN_PAD]
    targets = torch.tensor([[pos, pos, row, beat, eos, pad]])
    pred = torch.tensor([pos, pos + 1, row, pos, eos, row])
    logits = torch.full((1, 6, width), -100.0)
    logits[0, torch.arange(6), pred] = 10.0
    counts = accuracy_counts(logits, targets)
    assert counts["all"] == (3, 5)
    assert counts["pos"] == (1, 2)
    assert counts["row"] == (1, 1)
    assert counts["beat"] == (0, 1)
    assert counts["eos"] == (1, 1)


def test_ce_by_type_is_the_mean_on_each_gold_type() -> None:
    vocab = build_vocab()
    width = len(vocab)
    pos = next(index for token, index in vocab.items() if token.startswith("<POS_"))
    row = next(index for token, index in vocab.items() if token.startswith("<ROW_"))
    beat = vocab["<BEAT>"]
    eos = vocab["<EOS>"]
    pad = vocab[TOKEN_PAD]
    targets = torch.tensor([[pos, pos, row, beat, eos, pad]])
    logits = torch.zeros(1, targets.shape[1], width)
    parts = ce_by_type(logits, targets)
    assert parts["pos"][1] == 2
    assert parts["row"][1] == 1
    assert parts["beat"][1] == 1
    assert parts["eos"][1] == 1
    total = sum(value for value, _count in parts.values())
    count = sum(_count for _value, _count in parts.values())
    mean = total / count
    assert abs(mean - math.log(width)) < 1e-4
    assert abs(mean - float(plain_ce(logits, targets))) < 1e-4
    missing = torch.tensor([[pos, row, eos, pad]])
    gone = ce_by_type(logits[:, :4], missing)
    assert gone["beat"] == (0.0, 0)
    try:
        ce_by_type(logits[:, :1], torch.tensor([[vocab[TOKEN_BOS]]]))
    except RuntimeError:
        return
    raise AssertionError("BOS should not be a loss type")


def test_chart_shape_matches_the_forward_grid() -> None:
    notes = [ManiaNote(0, 0, NoteType.TAP)]
    redlines = [Redline(0.0, 500.0, 4, 0)]
    frames, tokens = chart_shape(notes, redlines, 1500.0)
    encoded = encode_chart(notes, redlines, 1500.0)
    span = frame_index(encoded.grid.coverage_end_ms) - frame_index(encoded.grid.coverage_start_ms)
    assert frames == span
    assert tokens == len(encoded.tokens)
    assert frames > 0


def test_val_mix_uses_the_training_branch_weights() -> None:
    from audio2map.train.v1.loop import _ValReadout, _mix_readout

    def row(chart: float, token: float, acc_chart: float, acc_token: float) -> _ValReadout:
        return _ValReadout(chart, token, acc_chart, acc_token, 0.0, 0.0, 0.0, 0.0)

    mixed = _mix_readout(
        row(1.0, 10.0, 0.4, 0.5),
        row(2.0, 20.0, 0.6, 0.7),
        row(3.0, 30.0, 0.8, 0.9),
    )
    assert abs(mixed.loss_chart - (0.1 * 1.0 + 0.8 * 2.0 + 0.1 * 3.0)) < 1e-9
    assert abs(mixed.loss_token - (0.1 * 10.0 + 0.8 * 20.0 + 0.1 * 30.0)) < 1e-9
    assert abs(mixed.acc_chart - (0.1 * 0.4 + 0.8 * 0.6 + 0.1 * 0.8)) < 1e-9
    assert abs(mixed.acc_token - (0.1 * 0.5 + 0.8 * 0.7 + 0.1 * 0.9)) < 1e-9


def test_windows_follow_the_shortest_chart() -> None:
    # 100 * 1.1 includes 110. 121 starts the next window.
    assert cut_windows([100, 110, 121, 200]) == [[0, 1], [2], [3]]


def test_pack_stops_when_the_padded_product_exceeds_the_budget() -> None:
    assert pack_in_order([0, 1, 2], [10, 10, 100], budget=50) == [[0, 1], [2]]


def test_two_charts_past_half_the_budget_stay_alone() -> None:
    frames = [300_000, 300_000]
    assert pack_in_order([0, 1], frames, budget=FRAME_BUDGET) == [[0], [1]]


def test_epoch_order_repeats_for_the_same_seed() -> None:
    frames = [10, 20, 0, 30, 40]
    first = epoch_batches(frames, seed=7, epoch=1, budget=50, frac=0.1)
    second = epoch_batches(frames, seed=7, epoch=1, budget=50, frac=0.1)
    assert first == second
    assert sorted(index for batch in first for index in batch) == [0, 1, 2, 3, 4]
    assert [2] in first


def test_resume_cursor_keeps_the_tail_of_a_batch() -> None:
    from audio2map.train.v1.loop import _queue_from

    assert _queue_from([[0, 1, 2], [3, 4]], 2) == [([2], 3), ([3, 4], 5)]


def test_length_cache_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "lengths.jsonl"
    rows = {
        "a.osu": ChartLength("a.osu", 100, 20, 1000.0),
        "b.osu": ChartLength("b.osu", 0, 0, 0.0, error="ChartRejected: no"),
    }
    _write_cache(path, rows)
    loaded = _read_cache(path)
    assert loaded["a.osu"].n_frames == 100
    assert loaded["a.osu"].n_tokens == 20
    assert loaded["b.osu"].error == "ChartRejected: no"
    assert not path.with_name("lengths.jsonl.part").exists()
