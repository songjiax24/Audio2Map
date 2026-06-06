"""Training window sampling for v2 (see docs/v2_spec.md §5.2)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from audio2map.osu.grid_config import TICKS_PER_BAR, WINDOW_BARS
from audio2map.osu.row_tokens import TOKEN_ROW_PREFIX, CanonicalTiming, chart_event_bar_range
from audio2map.osu.schema import Beatmap
from audio2map.osu.tick_range import audio_bar_range_ms


@dataclass(frozen=True, slots=True)
class WindowSamplingConfig:
    """Placeholder for future training window sampling options."""


def build_loss_mask(tokens: list[str]) -> list[int]:
    """``loss=0`` for ``<BOS>`` and initial ``<ROW_*>``; ``1`` elsewhere."""
    mask = [1] * len(tokens)
    if not tokens:
        return mask
    mask[0] = 0
    if len(tokens) > 1 and tokens[1].startswith(TOKEN_ROW_PREFIX):
        mask[1] = 0
    return mask


def chart_bar_range(beatmap: Beatmap, timing: CanonicalTiming) -> tuple[int, int]:
    return chart_event_bar_range(beatmap.notes, timing)


def audio_bar_range_from_duration(duration_ms: float, timing: CanonicalTiming) -> tuple[int, int]:
    return audio_bar_range_ms(
        duration_ms,
        offset_ms=float(timing.offset_ms),
        tick_ms=timing.tick_ms,
    )


def train_bar_range(
    *,
    audio_start_bar: int,
    audio_end_bar: int,
) -> tuple[int, int]:
    """Sampleable bar range: full audio span ``[audio_start_bar, audio_end_bar)``."""
    return audio_start_bar, audio_end_bar


def sample_window_bars(
    *,
    audio_start_bar: int,
    audio_end_bar: int,
    cfg: WindowSamplingConfig | None = None,
    rng: random.Random,
) -> tuple[int, int] | None:
    """Sample ``[start_bar, start_bar + WINDOW_BARS)``; return ``None`` if range too narrow."""
    del cfg
    train_start, train_end = train_bar_range(
        audio_start_bar=audio_start_bar,
        audio_end_bar=audio_end_bar,
    )
    if train_end - train_start < WINDOW_BARS:
        return None
    start_bar = rng.randint(train_start, train_end - WINDOW_BARS)
    return start_bar, start_bar + WINDOW_BARS


def window_tick_span(start_bar: int, end_bar: int) -> tuple[int, int]:
    return start_bar * TICKS_PER_BAR, end_bar * TICKS_PER_BAR


def assert_window_start_bar_aligned(start_bar: int) -> None:
    """Window-relative tick ``t`` uses ``pos_in_bar[t % TICKS_PER_BAR]``.

    ``start_bar`` may be negative (pre-offset audio); bar-aligned starts still
    place window tick 0 at within-bar position 0 because
    ``(start_bar * TICKS_PER_BAR + t) % TICKS_PER_BAR == t % TICKS_PER_BAR``.
    """
    if start_bar * TICKS_PER_BAR % TICKS_PER_BAR != 0:
        raise AssertionError(f"window start_bar={start_bar} is not bar-aligned")
