"""Tests for constrained decoding and overlap windows."""

from __future__ import annotations

from audio2map.osu.row_tokens import (
    TOKEN_BAR,
    TOKEN_BOS,
    TOKEN_EOS,
    CanonicalTiming,
    build_vocab,
    encode_window_tokens,
    invert_vocab,
    row_state_to_token,
    tick_to_ms,
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


def test_force_bar_after_context_prompt_replay() -> None:
    from audio2map.osu.row_tokens import pos_token

    vocab = build_vocab()
    body = encode_window_tokens(
        {0: {0: (1, 0, 0, 0)}},
        start_bar=0,
        end_bar=1,
        initial_row=(0, 0, 0, 0),
        include_bos_eos=False,
    )
    state = ChartDecodeState.from_initial_row((0, 0, 0, 0), window_bars=16, vocab=vocab)
    for tid in [vocab[t] for t in body]:
        state.observe(tid)
    assert vocab[pos_token(48)] in state.allowed_token_ids()
    state.require_bar_after_prompt()
    assert state.allowed_token_ids() == {vocab[TOKEN_BAR]}
    state.observe(vocab[TOKEN_BAR])
    assert not state.force_next_bar
    assert vocab[pos_token(0)] in state.allowed_token_ids()


def test_no_context_still_forces_first_bar() -> None:
    vocab = build_vocab()
    state = ChartDecodeState.from_initial_row((0, 0, 0, 0), window_bars=16, vocab=vocab)
    assert state.allowed_token_ids() == {vocab[TOKEN_BAR]}


def test_decode_empty_bars() -> None:
    vocab = build_vocab()
    tokens = [TOKEN_BOS, row_state_to_token((0, 0, 0, 0)), TOKEN_BAR, TOKEN_BAR, TOKEN_EOS]
    state = ChartDecodeState.from_initial_row((0, 0, 0, 0), window_bars=2, vocab=vocab)
    for tid in [vocab[t] for t in tokens[2:]]:
        assert tid in state.allowed_token_ids()
        state.observe_and_advance_bar_if_needed(tid)
    assert state.finished


def _union_covers_range(intervals: list[tuple[int, int]], start: int, end: int) -> bool:
    if start >= end:
        return True
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(intervals):
        if not merged or lo > merged[-1][1]:
            merged.append((lo, hi))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
    if not merged or merged[0][0] > start:
        return False
    cur = start
    for lo, hi in merged:
        if lo > cur:
            return False
        cur = max(cur, hi)
    return cur >= end


def test_overlap_windows_cover_range() -> None:
    from audio2map.osu.grid_config import WINDOW_BARS
    from audio2map.training.inference import OverlapConfig, overlap_inference_bar_windows

    cfg = OverlapConfig()
    assert cfg.context_bars == 8 and cfg.keep_bars == 4 and cfg.future_bars == 4
    wins = overlap_inference_bar_windows(-2, 20, cfg)
    assert wins[0][0] == -2, "first window starts at gen_start"
    assert wins[0][2] == wins[0][0], "first window must keep from win_start (context=0)"
    assert wins[0][3] == wins[0][0] + cfg.context_bars, "first window keeps context_bars"
    assert wins[0][4] == 0
    assert wins[-1][2] < wins[-1][3]
    if len(wins) > 1:
        assert wins[1][4] == cfg.context_bars


def test_overlap_short_span_uses_single_window() -> None:
    from audio2map.training.inference import OverlapConfig, overlap_inference_bar_windows

    cfg = OverlapConfig()
    for gen in [(0, 5), (0, 12), (0, 16), (-2, 3)]:
        wins = overlap_inference_bar_windows(*gen, cfg)
        assert len(wins) == 1, gen
        win_start, win_end, keep_start, keep_end, ctx = wins[0]
        assert ctx == 0
        assert keep_start == gen[0] and keep_end == gen[1]
        assert win_start == gen[0]
        assert win_end - win_start == cfg.window_bars


def test_overlap_audio_window_length_is_fixed() -> None:
    from audio2map.training.inference import OverlapConfig, overlap_inference_bar_windows

    cfg = OverlapConfig()
    for gen in [(0, 5), (0, 12), (0, 16)]:
        wins = overlap_inference_bar_windows(*gen, cfg)
        for win_start, win_end, *_ in wins:
            assert win_end - win_start == cfg.window_bars, (gen, win_start, win_end)

    wins = overlap_inference_bar_windows(0, 32, cfg)
    for win_start, win_end, *_ in wins:
        assert win_end - win_start == cfg.window_bars
    assert len(overlap_inference_bar_windows(0, 12, cfg)) == 1
    assert len(overlap_inference_bar_windows(0, 17, cfg)) > 1


def test_overlap_last_window_uses_tail_window_bars() -> None:
    from audio2map.training.inference import OverlapConfig, overlap_inference_bar_windows

    cfg = OverlapConfig()
    wins = overlap_inference_bar_windows(0, 32, cfg)
    win_start, win_end, keep_start, keep_end, _ = wins[-1]
    assert win_start == 16
    assert win_end == 32
    assert win_end - win_start == cfg.window_bars
    assert keep_start == 28
    assert keep_end == 32


def test_initial_row_from_committed_bars_open_hold() -> None:
    from audio2map.osu.row_tokens import LaneState, build_vocab, encode_window_tokens
    from audio2map.training.inference import initial_row_from_committed_bars

    vocab = build_vocab()
    hold_row = (LaneState.HOLD_START, 0, 0, 0)
    body = encode_window_tokens(
        {4: {0: hold_row}},
        start_bar=4,
        end_bar=5,
        initial_row=(0, 0, 0, 0),
        include_bos_eos=False,
    )
    committed = {4: [vocab[t] for t in body]}
    row = initial_row_from_committed_bars(committed, gen_start=0, win_start=8, vocab=vocab)
    assert row[0] == LaneState.HOLD_ACTIVE


def test_assemble_chart_token_ids() -> None:
    from audio2map.osu.row_tokens import TOKEN_BAR, TOKEN_BOS, TOKEN_EOS, build_vocab
    from audio2map.training.inference import assemble_chart_token_ids

    vocab = build_vocab()
    id_to_token = invert_vocab(vocab)
    committed = {
        0: [vocab[TOKEN_BAR]],
        1: [vocab[TOKEN_BAR]],
    }
    ids = assemble_chart_token_ids(committed, gen_start=0, gen_end=2, vocab=vocab)
    toks = [id_to_token[i] for i in ids]
    assert toks[0] == TOKEN_BOS
    assert toks[-1] == TOKEN_EOS
    assert toks.count(TOKEN_BAR) == 2


def test_build_context_prompt_token_ids_from_committed_bars() -> None:
    from audio2map.osu.row_tokens import TOKEN_BAR, TOKEN_BOS, TOKEN_EOS, build_vocab, encode_window_tokens
    from audio2map.training.inference import build_context_prompt_token_ids, extract_bar_token_ids

    vocab = build_vocab()
    id_to_token = invert_vocab(vocab)
    initial = (0, 0, 0, 0)
    body = encode_window_tokens(
        {0: {0: (1, 0, 0, 0)}, 1: {0: (1, 0, 0, 0)}},
        start_bar=0,
        end_bar=2,
        initial_row=initial,
        include_bos_eos=False,
    )
    win_tokens = [vocab[TOKEN_BOS], vocab["<ROW_0000>"]] + [vocab[t] for t in body] + [vocab[TOKEN_EOS]]
    committed = {
        bar: extract_bar_token_ids(win_tokens, id_to_token, win_start_bar=0, bar=bar)
        for bar in range(2)
    }
    prompt = build_context_prompt_token_ids(
        committed_bars=committed,
        win_start=0,
        keep_start=2,
        initial_row=initial,
        vocab=vocab,
    )
    assert prompt is not None
    assert [id_to_token[i] for i in prompt[:2]] == [TOKEN_BOS, "<ROW_0000>"]
    assert id_to_token[prompt[2]] == TOKEN_BAR
    assert build_context_prompt_token_ids(
        committed_bars=committed,
        win_start=0,
        keep_start=0,
        initial_row=initial,
        vocab=vocab,
    ) is None


def test_overlap_keep_regions_tile_without_gaps() -> None:
    from audio2map.training.inference import OverlapConfig, overlap_inference_bar_windows

    cfg = OverlapConfig()
    wins = overlap_inference_bar_windows(0, 32, cfg)
    covered: list[tuple[int, int]] = [tuple(w[2:4]) for w in wins]  # type: ignore[misc]
    assert covered[0] == (0, 8), "first window keeps context_bars only"
    assert _union_covers_range(covered, 0, 32)
    assert covered[-1][1] == 32
