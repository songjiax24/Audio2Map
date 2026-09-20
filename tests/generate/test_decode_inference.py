"""Tests for constrained decoding and overlap windows."""

from __future__ import annotations

from audio2map.grid import (
    CanonicalTiming,
    tick_to_ms,
)
from audio2map.tokens import (
    TOKEN_BAR,
    TOKEN_BOS,
    TOKEN_EOS,
    ChartState,
    LaneState,
    build_vocab,
    invert_vocab,
    pos_to_token,
    row_state_to_token,
)

_EMPTY = (
    LaneState.EMPTY,
    LaneState.EMPTY,
    LaneState.EMPTY,
    LaneState.EMPTY,
)
_TAP = (
    LaneState.TAP,
    LaneState.EMPTY,
    LaneState.EMPTY,
    LaneState.EMPTY,
)


def test_decode_state_matches_valid_sequence() -> None:
    vocab = build_vocab()
    tokens = [
        TOKEN_BOS,
        row_state_to_token(_EMPTY),
        TOKEN_BAR,
        pos_to_token(48),
        row_state_to_token(
            (LaneState.EMPTY, LaneState.EMPTY, LaneState.HOLD_START, LaneState.EMPTY)
        ),
        pos_to_token(96),
        row_state_to_token(
            (LaneState.EMPTY, LaneState.EMPTY, LaneState.HOLD_ACTIVE, LaneState.TAP)
        ),
        TOKEN_BAR,
        TOKEN_EOS,
    ]

    state = ChartState.from_initial_row(_EMPTY, window_bars=2, vocab=vocab)
    ids = [vocab[t] for t in tokens]
    for tid in ids[2:]:
        allowed = state.allowed_token_ids()
        assert tid in allowed, (invert_vocab()[tid], allowed)
        state.observe(tid)
    assert state.finished
    assert state.bars_done == 2


def test_overlap_keep_starts_with_injected_bar() -> None:
    import torch

    from audio2map.generate.overlap import DecodeConfig, generate_window_tokens

    vocab = build_vocab()
    bar_id = vocab[TOKEN_BAR]
    eos_id = vocab[TOKEN_EOS]
    pos48 = vocab[pos_to_token(48)]
    prompt = [
        vocab[TOKEN_BOS],
        vocab[row_state_to_token(_EMPTY)],
        bar_id,
        vocab[pos_to_token(0)],
        vocab[row_state_to_token(_TAP)],
    ]
    state = ChartState.from_initial_row(_EMPTY, window_bars=1, vocab=vocab)
    for tid in prompt[2:]:
        state.observe(tid)
    assert pos48 in state.allowed_token_ids()

    class _Stub:
        max_decoder_len = 2048

        def eval(self):
            return self

        def next_token_logits(self, audio, cond, token_ids, **kwargs):
            logits = torch.full((token_ids.shape[0], len(vocab)), -1e9)
            logits[:, eos_id] = 10.0
            logits[:, pos48] = 5.0
            return logits

    out = generate_window_tokens(
        _Stub(),  # type: ignore[arg-type]
        audio=torch.zeros(8, 4),
        cond_vec=torch.zeros(18),
        initial_row=_EMPTY,
        window_bars=1,
        device=torch.device("cpu"),
        decode=DecodeConfig(temperature=0.0),
        prompt_token_ids=prompt,
        vocab=vocab,
    )
    assert out[: len(prompt) + 1] == prompt + [bar_id]
    assert out[-1] == eos_id


def test_no_context_starts_with_injected_bar() -> None:
    import torch

    from audio2map.generate.overlap import DecodeConfig, generate_window_tokens
    from audio2map.tokens import bos_initial_prefix

    vocab = build_vocab()
    bar_id = vocab[TOKEN_BAR]
    eos_id = vocab[TOKEN_EOS]
    prefix = bos_initial_prefix(vocab[row_state_to_token(_EMPTY)], vocab=vocab)

    class _Stub:
        max_decoder_len = 2048

        def eval(self):
            return self

        def next_token_logits(self, audio, cond, token_ids, **kwargs):
            logits = torch.full((token_ids.shape[0], len(vocab)), -1e9)
            logits[:, eos_id] = 0.0
            return logits

    out = generate_window_tokens(
        _Stub(),  # type: ignore[arg-type]
        audio=torch.zeros(8, 4),
        cond_vec=torch.zeros(18),
        initial_row=_EMPTY,
        window_bars=1,
        device=torch.device("cpu"),
        decode=DecodeConfig(temperature=0.0),
        vocab=vocab,
    )
    assert out[:3] == prefix + [bar_id]
    assert out[-1] == eos_id


def test_decode_empty_bars() -> None:
    vocab = build_vocab()
    tokens = [TOKEN_BOS, row_state_to_token(_EMPTY), TOKEN_BAR, TOKEN_BAR, TOKEN_EOS]
    state = ChartState.from_initial_row(_EMPTY, window_bars=2, vocab=vocab)
    for tid in [vocab[t] for t in tokens[2:]]:
        assert tid in state.allowed_token_ids()
        state.observe(tid)
    assert state.finished
    assert state.bars_done == 2


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
    from audio2map.model.config import WINDOW_BARS
    from audio2map.generate.overlap import OverlapConfig, overlap_inference_bar_windows

    cfg = OverlapConfig()
    assert cfg.window_bars == WINDOW_BARS
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
    from audio2map.generate.overlap import OverlapConfig, overlap_inference_bar_windows

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
    from audio2map.generate.overlap import OverlapConfig, overlap_inference_bar_windows

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
    from audio2map.generate.overlap import OverlapConfig, overlap_inference_bar_windows

    cfg = OverlapConfig()
    wins = overlap_inference_bar_windows(0, 32, cfg)
    win_start, win_end, keep_start, keep_end, ctx = wins[-1]
    assert win_start == 16
    assert win_end == 32
    assert win_end - win_start == cfg.window_bars
    assert keep_start == 28
    assert keep_end == 32
    assert ctx == keep_start - win_start


def test_initial_row_from_committed_bars_open_hold() -> None:
    from audio2map.generate.overlap import initial_row_from_committed_bars

    vocab = build_vocab()
    hold_row = (
        LaneState.HOLD_START,
        LaneState.EMPTY,
        LaneState.EMPTY,
        LaneState.EMPTY,
    )
    body = [TOKEN_BAR, pos_to_token(0), row_state_to_token(hold_row)]
    committed = {4: [vocab[t] for t in body]}
    row = initial_row_from_committed_bars(committed, gen_start=0, win_start=8, vocab=vocab)
    assert row[0] == LaneState.HOLD_ACTIVE


def test_assemble_chart_token_ids() -> None:
    from audio2map.tokens import (
        TOKEN_BAR,
        build_vocab,
    )
    from audio2map.generate.overlap import assemble_chart_token_ids

    vocab = build_vocab()
    id_to_token = invert_vocab(vocab)
    committed = {
        0: [vocab[TOKEN_BAR]],
        1: [vocab[TOKEN_BAR]],
    }
    ids = assemble_chart_token_ids(committed, gen_start=0, gen_end=2, vocab=vocab)
    toks = [id_to_token[i] for i in ids]
    assert toks == [TOKEN_BAR, TOKEN_BAR]


def test_build_context_prompt_token_ids_from_committed_bars() -> None:
    from audio2map.generate.overlap import build_context_prompt_token_ids, extract_bar_token_ids

    vocab = build_vocab()
    id_to_token = invert_vocab(vocab)
    initial = _EMPTY
    tap = row_state_to_token(_TAP)
    body = [
        TOKEN_BAR,
        pos_to_token(0),
        tap,
        TOKEN_BAR,
        pos_to_token(0),
        tap,
    ]
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
    from audio2map.generate.overlap import OverlapConfig, overlap_inference_bar_windows

    cfg = OverlapConfig()
    for gen in [(0, 32), (0, 17), (-2, 20), (3, 40)]:
        wins = overlap_inference_bar_windows(*gen, cfg)
        covered: list[tuple[int, int]] = [tuple(w[2:4]) for w in wins]  # type: ignore[misc]
        assert covered[0][0] == gen[0]
        assert covered[-1][1] == gen[1]
        for i in range(len(covered) - 1):
            assert covered[i][1] == covered[i + 1][0], (gen, covered)
        assert _union_covers_range(covered, *gen)
        for w in wins:
            assert w.win_start <= w.keep_start < w.keep_end <= w.win_end
            assert w.win_end - w.win_start == cfg.window_bars
            if w.context_bars:
                assert w.context_bars == w.keep_start - w.win_start


def test_overlap_config_rejects_zero_keep() -> None:
    import pytest

    from audio2map.generate.overlap import OverlapConfig

    with pytest.raises(ValueError, match="keep_bars"):
        OverlapConfig(window_bars=16, context_bars=16, keep_bars=0, future_bars=0)


def test_extract_bar_drops_trailing_pos() -> None:
    from audio2map.generate.overlap import extract_bar_token_ids

    vocab = build_vocab()
    id_to_token = invert_vocab(vocab)
    tokens = [
        vocab[TOKEN_BOS],
        vocab[row_state_to_token(_EMPTY)],
        vocab[TOKEN_BAR],
        vocab[pos_to_token(0)],
        vocab[row_state_to_token(_TAP)],
        vocab[TOKEN_BAR],
        vocab[pos_to_token(0)],
        vocab[TOKEN_EOS],
    ]
    bar0 = extract_bar_token_ids(tokens, id_to_token, win_start_bar=0, bar=0)
    bar1 = extract_bar_token_ids(tokens, id_to_token, win_start_bar=0, bar=1)
    assert [id_to_token[i] for i in bar0] == [
        TOKEN_BAR,
        pos_to_token(0),
        row_state_to_token(_TAP),
    ]
    assert [id_to_token[i] for i in bar1] == [TOKEN_BAR]


def test_generate_window_tokens_raises_if_truncated() -> None:
    import pytest
    import torch

    from audio2map.generate.overlap import DecodeConfig, generate_window_tokens

    vocab = build_vocab()

    class _Stub:
        max_decoder_len = 2048

        def eval(self):
            return self

        def next_token_logits(self, audio, cond, token_ids, **kwargs):
            return torch.zeros(token_ids.shape[0], len(vocab))

    with pytest.raises(RuntimeError, match="truncated"):
        generate_window_tokens(
            _Stub(),  # type: ignore[arg-type]
            audio=torch.zeros(8, 4),
            cond_vec=torch.zeros(18),
            initial_row=_EMPTY,
            window_bars=2,
            device=torch.device("cpu"),
            decode=DecodeConfig(temperature=0.0),
            max_seq_len=3,
            vocab=vocab,
        )
