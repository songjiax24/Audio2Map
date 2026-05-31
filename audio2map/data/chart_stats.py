"""Chart-level hold statistics."""

from __future__ import annotations

from audio2map.osu.bpm import canonicalize_bpm
from audio2map.osu.grid_config import TICKS_PER_BAR
from audio2map.osu.row_tokens import (
    CanonicalTiming,
    active_hold_at_tick,
    all_event_ticks,
    chart_event_bar_range,
    notes_to_raw_events,
)
from audio2map.osu.schema import Beatmap, NoteType, TimingPoint
from audio2map.osu.tick_range import chart_total_ticks_from_bar_range
from audio2map.osu.timing import beat_length_to_bpm


def hold_ratio(beatmap: Beatmap) -> float:
    n = beatmap.note_count
    if n == 0:
        return 0.0
    ln = sum(1 for note in beatmap.notes if note.note_type == NoteType.HOLD)
    return ln / n


def hold_coverage(beatmap: Beatmap, timing: CanonicalTiming | None = None) -> float:
    """``total_hold_lane_ticks / (chart_total_ticks * 4)`` over bar-aligned event range."""
    timing = timing or CanonicalTiming.from_beatmap(beatmap)
    if not all_event_ticks(beatmap.notes, timing):
        return 0.0

    start_bar, end_bar = chart_event_bar_range(beatmap.notes, timing)
    chart_total_ticks = chart_total_ticks_from_bar_range(start_bar, end_bar)
    if chart_total_ticks <= 0:
        return 0.0

    start_tick = start_bar * TICKS_PER_BAR
    end_tick = end_bar * TICKS_PER_BAR
    last_tick = end_tick - 1

    events = notes_to_raw_events(beatmap.notes, timing)
    hold_lane_ticks = 0
    ticks_sorted = sorted(t for t in events if start_tick <= t <= last_tick)
    active = active_hold_at_tick(events, start_tick)
    prev = start_tick
    for tick in ticks_sorted:
        span = max(0, tick - prev)
        hold_lane_ticks += span * sum(1 for v in active if v)
        active = active_hold_at_tick(events, tick, active)
        prev = tick
    span = max(0, last_tick - prev + 1)
    hold_lane_ticks += span * sum(1 for v in active if v)

    return hold_lane_ticks / (chart_total_ticks * 4)


def timing_metadata(beatmap: Beatmap) -> dict:
    tp = next(tp for tp in beatmap.timing_points if tp.uninherited)
    original_bpm = beat_length_to_bpm(tp.beat_length_ms)
    canonical_bpm, scale_exp = canonicalize_bpm(original_bpm)
    return {
        "offset_ms": tp.offset_ms,
        "original_bpm": original_bpm,
        "canonical_bpm": canonical_bpm,
        "bpm_scale_exp": scale_exp,
        "meter": tp.meter,
    }
