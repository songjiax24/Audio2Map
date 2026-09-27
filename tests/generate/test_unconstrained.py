"""Unconstrained decoding records validity without masking or raising."""

from __future__ import annotations

import torch

from audio2map.generate.overlap import DecodeConfig, generate_window_decode
from audio2map.tokens import (
    TOKEN_BAR,
    TOKEN_BOS,
    TOKEN_EOS,
    LaneState,
    RowState,
    build_vocab,
    pos_to_token,
    row_state_to_token,
)

_EMPTY: RowState = (LaneState.EMPTY, LaneState.EMPTY, LaneState.EMPTY, LaneState.EMPTY)
_TAP: RowState = (LaneState.TAP, LaneState.EMPTY, LaneState.EMPTY, LaneState.EMPTY)


class _Stub:
    def __init__(self, favored: int, vocab_size: int) -> None:
        self.max_decoder_len = 2048
        self.favored = favored
        self.vocab_size = vocab_size

    def eval(self):
        return self

    def next_token_logits(self, audio, cond, token_ids, **kwargs):
        logits = torch.full((token_ids.shape[0], self.vocab_size), -1e9)
        logits[:, self.favored] = 10.0
        return logits


def test_unconstrained_illegal_token_does_not_raise() -> None:
    vocab = build_vocab()
    decoded = generate_window_decode(
        _Stub(0, len(vocab)),  # type: ignore[arg-type]
        audio=torch.zeros(8, 4),
        cond_vec=torch.zeros(18),
        initial_row=_EMPTY,
        window_bars=1,
        device=torch.device("cpu"),
        decode=DecodeConfig(temperature=0.0),
        vocab=vocab,
        constrain=False,
    )
    assert decoded.status == "illegal_token"
    assert decoded.token_ids[-1] == 0


def test_constrained_still_masks_and_unconstrained_truncation_returns() -> None:
    vocab = build_vocab()
    eos = vocab[TOKEN_EOS]
    decoded = generate_window_decode(
        _Stub(0, len(vocab)),  # type: ignore[arg-type]
        audio=torch.zeros(8, 4),
        cond_vec=torch.zeros(18),
        initial_row=_EMPTY,
        window_bars=1,
        device=torch.device("cpu"),
        decode=DecodeConfig(temperature=0.0),
        vocab=vocab,
        constrain=True,
    )
    assert decoded.status == "valid"
    assert decoded.token_ids[-1] == eos

    pos0 = vocab[pos_to_token(0)]
    truncated = generate_window_decode(
        _Stub(pos0, len(vocab)),  # type: ignore[arg-type]
        audio=torch.zeros(8, 4),
        cond_vec=torch.zeros(18),
        initial_row=_EMPTY,
        window_bars=2,
        device=torch.device("cpu"),
        decode=DecodeConfig(temperature=0.0),
        max_seq_len=4,
        vocab=vocab,
        constrain=False,
    )
    assert truncated.status == "truncated"


def test_prompt_replay_is_not_scored_as_a_sample() -> None:
    vocab = build_vocab()
    eos = vocab[TOKEN_EOS]
    bar = vocab[TOKEN_BAR]
    prompt = [
        vocab[TOKEN_BOS],
        vocab[row_state_to_token(_EMPTY)],
        bar,
        vocab[pos_to_token(0)],
        vocab[row_state_to_token(_TAP)],
    ]
    valid = generate_window_decode(
        _Stub(eos, len(vocab)),  # type: ignore[arg-type]
        audio=torch.zeros(8, 4),
        cond_vec=torch.zeros(18),
        initial_row=_EMPTY,
        window_bars=2,
        device=torch.device("cpu"),
        decode=DecodeConfig(temperature=0.0),
        prompt_token_ids=prompt,
        vocab=vocab,
        constrain=False,
    )
    assert valid.status == "valid"
    assert valid.token_ids[: len(prompt) + 1] == prompt + [bar]
    assert valid.token_ids[-1] == eos

    illegal = generate_window_decode(
        _Stub(0, len(vocab)),  # type: ignore[arg-type]
        audio=torch.zeros(8, 4),
        cond_vec=torch.zeros(18),
        initial_row=_EMPTY,
        window_bars=2,
        device=torch.device("cpu"),
        decode=DecodeConfig(temperature=0.0),
        prompt_token_ids=prompt,
        vocab=vocab,
        constrain=False,
    )
    assert illegal.status == "illegal_token"
    assert illegal.token_ids[: len(prompt) + 1] == prompt + [bar]
    assert illegal.token_ids[-1] == 0
