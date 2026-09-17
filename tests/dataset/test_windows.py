"""Tests for training-window sampling."""

from __future__ import annotations

import random

from audio2map.dataset.windows import audio_covers_training_window, build_loss_mask, sample_training_window
from audio2map.features.audio.grid import AudioGridMeta
from audio2map.grid import TICKS_PER_BAR
from audio2map.tokens import TOKEN_BOS, TOKEN_EOS, LaneState, row_state_to_token


def test_build_loss_mask() -> None:
    tokens = [
        TOKEN_BOS,
        row_state_to_token(
            (LaneState.EMPTY, LaneState.HOLD_ACTIVE, LaneState.EMPTY, LaneState.EMPTY)
        ),
        "<BAR>",
        "<POS_0>",
        row_state_to_token((LaneState.TAP, LaneState.EMPTY, LaneState.EMPTY, LaneState.EMPTY)),
        TOKEN_EOS,
    ]
    assert build_loss_mask(tokens) == [0, 0, 1, 1, 1, 1]


def test_sample_training_window_allows_negative_start() -> None:
    train_start, train_end = -5, 50
    assert train_end - train_start > 16
    rng = random.Random(0)
    result = sample_training_window(
        audio_start_bar=-5,
        audio_end_bar=50,
        rng=rng,
    )
    assert result is not None
    assert result.window_bars == 16
    assert result.end_bar - result.start_bar == 16
    assert train_start <= result.start_bar < train_end
    assert result.start_bar == random.Random(0).randint(-5, 50 - 16)


def test_sample_training_window_rejects_short_span() -> None:
    assert sample_training_window(audio_start_bar=0, audio_end_bar=8, rng=random.Random(0)) is None


def test_audio_covers_training_window() -> None:
    assert audio_covers_training_window(0, 16)
    assert audio_covers_training_window(-2, 14)
    assert not audio_covers_training_window(0, 15)
    assert not audio_covers_training_window(0, 8)


def test_audio_covers_training_window_from_grid_meta() -> None:
    short = AudioGridMeta(
        tick_min=0,
        tick_max=8 * TICKS_PER_BAR,
        offset_ms=0,
        canonical_bpm=120.0,
        audio_hash="x",
    )
    start, end = short.audio_bar_range
    assert not audio_covers_training_window(start, end)
    full = AudioGridMeta(
        tick_min=0,
        tick_max=16 * TICKS_PER_BAR,
        offset_ms=0,
        canonical_bpm=120.0,
        audio_hash="x",
    )
    start, end = full.audio_bar_range
    assert audio_covers_training_window(start, end)
