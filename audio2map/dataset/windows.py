"""Training-window framing, loss mask, and bar sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass

from audio2map.model.config import WINDOW_BARS
from audio2map.tokens import TOKEN_BOS, TOKEN_EOS, TOKEN_ROW_PREFIX, RowState, row_state_to_token


@dataclass(frozen=True, slots=True)
class SampleWindow:
    start_bar: int
    window_bars: int = WINDOW_BARS

    @property
    def end_bar(self) -> int:
        return self.start_bar + self.window_bars


def frame_window_tokens(chart_tokens: list[str], initial_row: RowState) -> list[str]:
    """``<BOS> <ROW_initial>`` + chart tokens + ``<EOS>`` for the decoder."""
    return [TOKEN_BOS, row_state_to_token(initial_row), *chart_tokens, TOKEN_EOS]


def unframe_window_tokens(tokens: list[str]) -> list[str]:
    """Strip ``<BOS> <ROW_initial>`` … ``<EOS>``; return chart tokens."""
    if len(tokens) < 3 or tokens[0] != TOKEN_BOS or tokens[-1] != TOKEN_EOS:
        raise ValueError("expected <BOS> … <EOS> window sequence")
    if not tokens[1].startswith(TOKEN_ROW_PREFIX):
        raise ValueError("expected initial ROW after <BOS>")
    return tokens[2:-1]


def build_loss_mask(tokens: list[str]) -> list[int]:
    """``loss=0`` for ``<BOS>`` and initial ``<ROW_*>``; ``1`` elsewhere."""
    mask = [1] * len(tokens)
    if not tokens:
        return mask
    mask[0] = 0
    if len(tokens) > 1 and tokens[1].startswith(TOKEN_ROW_PREFIX):
        mask[1] = 0
    return mask


def audio_covers_training_window(
    audio_start_bar: int,
    audio_end_bar: int,
    *,
    window_bars: int = WINDOW_BARS,
) -> bool:
    """True if ``[audio_start_bar, audio_end_bar)`` can hold one training window."""
    return audio_end_bar - audio_start_bar >= window_bars


def sample_training_window(
    *,
    audio_start_bar: int,
    audio_end_bar: int,
    rng: random.Random,
    window_bars: int = WINDOW_BARS,
) -> SampleWindow | None:
    """Sample ``[start_bar, start_bar + window_bars)``; ``None`` if the span is too short."""
    if not audio_covers_training_window(
        audio_start_bar, audio_end_bar, window_bars=window_bars
    ):
        return None
    start_bar = rng.randint(audio_start_bar, audio_end_bar - window_bars)
    return SampleWindow(start_bar=start_bar, window_bars=window_bars)
