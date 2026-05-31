"""Absolute tick / bar range helpers.

``offset_ms`` is the beat-grid origin (``absolute_tick == 0``), not chart or
audio start.  Ticks and bars may be negative.
"""

from __future__ import annotations

import math

from audio2map.osu.grid_config import TICKS_PER_BAR


def ceil_div(n: int, d: int) -> int:
    """``ceil(n / d)`` for signed integers."""
    return -((-n) // d)


def split_absolute_tick(absolute_tick: int) -> tuple[int, int]:
    """Return ``(absolute_bar, pos)`` with ``0 <= pos < TICKS_PER_BAR``."""
    absolute_bar = absolute_tick // TICKS_PER_BAR
    pos = absolute_tick - absolute_bar * TICKS_PER_BAR
    return absolute_bar, pos


def chart_event_bar_range_from_ticks(ticks: list[int]) -> tuple[int, int]:
    """Bar-aligned chart span ``[chart_start_bar, chart_end_bar)``."""
    if not ticks:
        return 0, 0
    first = min(ticks)
    last = max(ticks)
    return first // TICKS_PER_BAR, ceil_div(last + 1, TICKS_PER_BAR)


def chart_total_ticks_from_bar_range(start_bar: int, end_bar: int) -> int:
    return (end_bar - start_bar) * TICKS_PER_BAR


def ms_to_absolute_tick(time_ms: int | float, *, offset_ms: float, tick_ms: float) -> int:
    return round((time_ms - offset_ms) / tick_ms)


def audio_tick_range_ms(
    duration_ms: float,
    *,
    offset_ms: float,
    tick_ms: float,
) -> tuple[int, int]:
    """Half-open ``[tick_min_audio, tick_max_audio)`` for the audio file."""
    tick_min = math.floor((0.0 - offset_ms) / tick_ms)
    tick_max = math.ceil((duration_ms - offset_ms) / tick_ms)
    return tick_min, tick_max


def audio_bar_range_ms(
    duration_ms: float,
    *,
    offset_ms: float,
    tick_ms: float,
) -> tuple[int, int]:
    tick_min, tick_max = audio_tick_range_ms(duration_ms, offset_ms=offset_ms, tick_ms=tick_ms)
    return tick_min // TICKS_PER_BAR, ceil_div(tick_max, TICKS_PER_BAR)


def train_sample_bar_range(
    *,
    chart_start_bar: int,
    chart_end_bar: int,
    audio_start_bar: int,
    audio_end_bar: int,
    pre_event_margin_bars: int = 4,
    post_event_margin_bars: int = 4,
) -> tuple[int, int]:
    """``[train_start_bar, train_end_bar)`` for window sampling."""
    train_start = max(audio_start_bar, chart_start_bar - pre_event_margin_bars)
    train_end = min(audio_end_bar, chart_end_bar + post_event_margin_bars)
    return train_start, train_end


def tick_to_array_index(absolute_tick: int, tick_min_audio: int) -> int:
    return absolute_tick - tick_min_audio


def audio_grid_cache_stem(
    audio_hash: str,
    *,
    canonical_bpm: float,
    offset_ms: float,
) -> str:
    return f"{audio_hash}_bpm{canonical_bpm:.3f}_offset{offset_ms:.1f}"
