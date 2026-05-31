"""Training window sampling for v2 (see docs/v2_spec.md §5.2)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from audio2map.osu.grid_config import DEFAULT_WINDOW_BARS_CHOICES, TICKS_PER_BAR
from audio2map.osu.row_tokens import TOKEN_ROW_PREFIX, CanonicalTiming, chart_event_bar_range
from audio2map.osu.schema import Beatmap
from audio2map.osu.tick_range import audio_bar_range_ms, train_sample_bar_range


@dataclass(frozen=True, slots=True)
class WindowSamplingConfig:
    window_bars: int = 8
    pre_event_margin_bars: int = 4
    post_event_margin_bars: int = 4
    window_bars_choices: tuple[int, ...] = DEFAULT_WINDOW_BARS_CHOICES

    def pick_window_bars(self, rng: random.Random) -> int:
        return rng.choice(self.window_bars_choices)


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
    chart_start_bar: int,
    chart_end_bar: int,
    audio_start_bar: int,
    audio_end_bar: int,
    cfg: WindowSamplingConfig,
) -> tuple[int, int]:
    return train_sample_bar_range(
        chart_start_bar=chart_start_bar,
        chart_end_bar=chart_end_bar,
        audio_start_bar=audio_start_bar,
        audio_end_bar=audio_end_bar,
        pre_event_margin_bars=cfg.pre_event_margin_bars,
        post_event_margin_bars=cfg.post_event_margin_bars,
    )


def sample_window_bars(
    *,
    chart_start_bar: int,
    chart_end_bar: int,
    audio_start_bar: int,
    audio_end_bar: int,
    cfg: WindowSamplingConfig,
    rng: random.Random,
) -> tuple[int, int] | None:
    """Sample ``[start_bar, end_bar)``; return ``None`` if range too narrow."""
    window_bars = cfg.pick_window_bars(rng)
    train_start, train_end = train_bar_range(
        chart_start_bar=chart_start_bar,
        chart_end_bar=chart_end_bar,
        audio_start_bar=audio_start_bar,
        audio_end_bar=audio_end_bar,
        cfg=cfg,
    )
    if train_end - train_start < window_bars:
        return None
    start_bar = rng.randint(train_start, train_end - window_bars)
    return start_bar, start_bar + window_bars


def inference_bar_windows(
    audio_start_bar: int,
    audio_end_bar: int,
    window_bars: int,
) -> list[tuple[int, int]]:
    """Non-overlapping windows covering ``[audio_start_bar, audio_end_bar)``."""
    windows: list[tuple[int, int]] = []
    bar = audio_start_bar
    while bar < audio_end_bar:
        end = min(bar + window_bars, audio_end_bar)
        windows.append((bar, end))
        if end >= audio_end_bar:
            break
        bar = end
    return windows


def window_tick_span(start_bar: int, end_bar: int) -> tuple[int, int]:
    return start_bar * TICKS_PER_BAR, end_bar * TICKS_PER_BAR
