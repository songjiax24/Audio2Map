"""Logit-level legality metrics."""

from __future__ import annotations

import pytest
import torch

from audio2map.grid import TICKS_PER_BAR
from audio2map.metrics.validity import logit_validity
from audio2map.tokens import (
    TOKEN_BAR,
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_PAD,
    ChartState,
    LaneState,
    build_vocab,
    pos_to_token,
    row_state_to_token,
)

_EMPTY = (
    LaneState.EMPTY,
    LaneState.EMPTY,
    LaneState.EMPTY,
    LaneState.EMPTY,
)
_HOLD_START = (
    LaneState.HOLD_START,
    LaneState.EMPTY,
    LaneState.EMPTY,
    LaneState.EMPTY,
)


def _framed(*body: str) -> list[str]:
    return [TOKEN_BOS, row_state_to_token(_EMPTY), *body]


def test_legal_argmax_on_ground_truth() -> None:
    vocab = build_vocab()
    tokens = _framed(TOKEN_BAR, TOKEN_EOS)
    ids = torch.tensor([[vocab[t] for t in tokens]])
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    logits = torch.full((1, 3, len(vocab)), -20.0)
    for t, tok in enumerate(tokens[1:]):
        logits[0, t, vocab[tok]] = 10.0
    stats = logit_validity(logits, ids, mask, window_bars=1, vocab=vocab)
    assert stats.n_valid == 2.0
    assert stats.legal_top1 == 1.0
    assert stats.legal_probability > 0.9


def test_illegal_argmax_is_not_counted() -> None:
    vocab = build_vocab()
    tokens = _framed(TOKEN_BAR, TOKEN_EOS)
    ids = torch.tensor([[vocab[t] for t in tokens]])
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    logits = torch.full((1, 3, len(vocab)), -20.0)
    logits[0, :, vocab[TOKEN_PAD]] = 10.0
    stats = logit_validity(logits, ids, mask, window_bars=1, vocab=vocab)
    assert stats.legal_top1 == 0.0
    assert stats.legal_probability < 0.1


def test_legal_probability_is_allowed_softmax_mass() -> None:
    vocab = build_vocab()
    tokens = _framed(TOKEN_BAR, TOKEN_EOS)
    ids = torch.tensor([[vocab[t] for t in tokens]])
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    logits = torch.zeros(1, 3, len(vocab))
    stats = logit_validity(logits, ids, mask, window_bars=1, vocab=vocab)
    v = float(len(vocab))
    first = 1.0 / v
    second = (float(TICKS_PER_BAR) + 1.0) / v
    assert stats.n_valid == 2.0
    assert stats.legal_probability == pytest.approx((first + second) / 2.0)


def test_empty_allowed_set_raises() -> None:
    vocab = build_vocab()
    tokens = _framed(TOKEN_BAR, TOKEN_EOS, TOKEN_BAR)
    ids = torch.tensor([[vocab[t] for t in tokens]])
    mask = torch.ones(1, len(tokens))
    mask[0, :2] = 0.0
    logits = torch.zeros(1, len(tokens) - 1, len(vocab))
    with pytest.raises(RuntimeError, match="empty legal token set"):
        logit_validity(logits, ids, mask, window_bars=1, vocab=vocab)


def test_validity_window_eos_with_active_hold() -> None:
    vocab = build_vocab()
    tokens = _framed(
        TOKEN_BAR,
        pos_to_token(0),
        row_state_to_token(_HOLD_START),
        TOKEN_EOS,
    )
    state = ChartState.from_initial_row(_EMPTY, window_bars=1, vocab=vocab)
    ids_list = [vocab[t] for t in tokens]
    for tid in ids_list[2:-1]:
        assert tid in state.allowed_token_ids()
        state.observe(tid)
    assert vocab[TOKEN_EOS] in state.allowed_token_ids()

    ids = torch.tensor([ids_list])
    mask = torch.zeros(1, len(ids_list))
    mask[0, 2:] = 1.0
    logits = torch.full((1, len(ids_list) - 1, len(vocab)), -20.0)
    for t, tok in enumerate(tokens[1:]):
        logits[0, t, vocab[tok]] = 10.0
    stats = logit_validity(logits, ids, mask, window_bars=1, vocab=vocab)
    assert stats.legal_top1 == 1.0
