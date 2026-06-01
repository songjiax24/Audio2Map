"""Tests for constrained decoding and overlap windows."""

from __future__ import annotations

from audio2map.osu.row_tokens import (
    TOKEN_BAR,
    TOKEN_BOS,
    TOKEN_EOS,
    build_vocab,
    encode_window_tokens,
    invert_vocab,
    row_state_to_token,
    validate_token_sequence,
)
from audio2map.training.decode import ChartDecodeState


def test_decode_state_matches_valid_sequence() -> None:
    vocab = build_vocab()
    tokens = encode_window_tokens(
        {
            0: {
                48: (0, 0, 2, 0),
                96: (0, 0, 3, 1),
            }
        },
        start_bar=0,
        end_bar=2,
        initial_row=(0, 0, 0, 0),
        include_bos_eos=True,
    )
    assert validate_token_sequence(tokens) == []

    state = ChartDecodeState.from_initial_row((0, 0, 0, 0), window_bars=2, vocab=vocab)
    ids = [vocab[t] for t in tokens]
    for tid in ids[2:]:
        allowed = state.allowed_token_ids()
        assert tid in allowed, (invert_vocab()[tid], allowed)
        state.observe_and_advance_bar_if_needed(tid)
    assert state.finished


def test_decode_empty_bars() -> None:
    vocab = build_vocab()
    tokens = [TOKEN_BOS, row_state_to_token((0, 0, 0, 0)), TOKEN_BAR, TOKEN_BAR, TOKEN_EOS]
    state = ChartDecodeState.from_initial_row((0, 0, 0, 0), window_bars=2, vocab=vocab)
    for tid in [vocab[t] for t in tokens[2:]]:
        assert tid in state.allowed_token_ids()
        state.observe_and_advance_bar_if_needed(tid)
    assert state.finished


def test_overlap_windows_cover_range() -> None:
    from audio2map.training.inference import OverlapConfig, overlap_inference_bar_windows

    cfg = OverlapConfig(window_bars=8, context_bars=4, keep_bars=4, future_bars=0)
    wins = overlap_inference_bar_windows(-2, 20, cfg)
    assert wins[0][0] <= -2
    assert wins[0][2] == wins[0][0], "first window must keep from win_start (context=0)"
    assert wins[0][4] == 0
    assert wins[-1][2] < wins[-1][3]
    if len(wins) > 1:
        assert wins[1][4] == cfg.context_bars
