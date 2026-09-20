"""ChartState grammar: legal next tokens and window EOS semantics."""

from __future__ import annotations

from audio2map.tokens import (
    TOKEN_BAR,
    TOKEN_EOS,
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


def test_window_eos_allows_active_hold() -> None:
    vocab = build_vocab()
    state = ChartState.from_initial_row(_EMPTY, window_bars=1, vocab=vocab)
    state.observe(vocab[TOKEN_BAR])
    state.observe(vocab[pos_to_token(0)])
    state.observe(vocab[row_state_to_token(_HOLD_START)])
    assert state.active_hold == [True, False, False, False]
    assert vocab[TOKEN_EOS] in state.allowed_token_ids()
    state.observe(vocab[TOKEN_EOS])
    assert state.finished
    assert state.bars_done == 1


def test_first_token_must_be_bar() -> None:
    vocab = build_vocab()
    state = ChartState.from_initial_row(_EMPTY, window_bars=16, vocab=vocab)
    assert state.allowed_token_ids() == {vocab[TOKEN_BAR]}
