"""Offset + BPM → tick/bar geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

from audio2map.grid.config import TICKS_PER_BAR, TICKS_PER_BEAT
from audio2map.osu.schema import Beatmap, ManiaNote, NoteType, TimingPoint
from audio2map.osu.timing import beat_length_to_bpm


def canonicalize_bpm(original_bpm: float) -> tuple[float, int]:
    """Map BPM to ``[120, 240)`` via powers of two. Returns ``(canonical_bpm, scale_exp)``."""
    if original_bpm <= 0:
        raise ValueError(f"invalid BPM: {original_bpm}")
    bpm = float(original_bpm)
    scale_exp = 0
    while bpm < 120.0:
        bpm *= 2.0
        scale_exp += 1
    while bpm >= 240.0:
        bpm /= 2.0
        scale_exp -= 1
    return bpm, scale_exp


def _ceil_div(n: int, d: int) -> int:
    return -((-n) // d)


@dataclass(frozen=True, slots=True)
class CanonicalTiming:
    offset_ms: int
    original_bpm: float
    canonical_bpm: float
    bpm_scale_exp: int

    @property
    def beat_ms(self) -> float:
        return 60_000.0 / self.canonical_bpm

    @property
    def tick_ms(self) -> float:
        return self.beat_ms / TICKS_PER_BEAT

    @classmethod
    def from_timing_points(cls, timing_points: list[TimingPoint]) -> CanonicalTiming:
        for tp in timing_points:
            if tp.uninherited:
                original_bpm = beat_length_to_bpm(tp.beat_length_ms)
                canonical_bpm, scale_exp = canonicalize_bpm(original_bpm)
                return cls(
                    offset_ms=tp.offset_ms,
                    original_bpm=original_bpm,
                    canonical_bpm=canonical_bpm,
                    bpm_scale_exp=scale_exp,
                )
        raise ValueError("no uninherited timing point")

    @classmethod
    def from_beatmap(cls, beatmap: Beatmap) -> CanonicalTiming:
        return cls.from_timing_points(beatmap.timing_points)


def ms_to_tick(time_ms: int | float, timing: CanonicalTiming) -> int:
    return round((time_ms - timing.offset_ms) / timing.tick_ms)


def tick_to_ms(tick: int, timing: CanonicalTiming) -> int:
    return timing.offset_ms + round(tick * timing.tick_ms)


def split_tick(tick: int) -> tuple[int, int]:
    """``(bar, pos)`` with ``0 <= pos < TICKS_PER_BAR``. ``tick`` may be negative."""
    bar = tick // TICKS_PER_BAR
    return bar, tick - bar * TICKS_PER_BAR


def all_event_ticks(notes: list[ManiaNote], timing: CanonicalTiming) -> list[int]:
    ticks: list[int] = []
    for note in notes:
        ticks.append(ms_to_tick(note.time_ms, timing))
        if note.note_type == NoteType.HOLD and note.end_time_ms is not None:
            ticks.append(ms_to_tick(note.end_time_ms, timing))
    return ticks


def chart_event_bar_range(
    notes: list[ManiaNote],
    timing: CanonicalTiming,
) -> tuple[int, int]:
    """Bar-aligned chart span ``[start, end)`` covering all note heads and hold tails."""
    ticks = all_event_ticks(notes, timing)
    if not ticks:
        return 0, 0
    first = min(ticks)
    last = max(ticks)
    return first // TICKS_PER_BAR, _ceil_div(last + 1, TICKS_PER_BAR)


def audio_tick_range_from_duration(duration_ms: float, timing: CanonicalTiming) -> tuple[int, int]:
    """Half-open ``[tick_min, tick_max)`` covering audio ``[0, duration_ms)``."""
    tick_ms = timing.tick_ms
    offset = float(timing.offset_ms)
    tick_min = math.floor((0.0 - offset) / tick_ms)
    tick_max = math.ceil((duration_ms - offset) / tick_ms)
    return tick_min, tick_max


def audio_bar_range_from_ticks(tick_min: int, tick_max: int) -> tuple[int, int]:
    """Bar-aligned span covering ``[tick_min, tick_max)``."""
    return tick_min // TICKS_PER_BAR, _ceil_div(tick_max, TICKS_PER_BAR)


def audio_bar_range_from_duration(duration_ms: float, timing: CanonicalTiming) -> tuple[int, int]:
    tick_min, tick_max = audio_tick_range_from_duration(duration_ms, timing)
    return audio_bar_range_from_ticks(tick_min, tick_max)
