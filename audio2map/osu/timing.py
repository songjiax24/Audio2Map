"""Timing point analysis (BPM and meter)."""

from __future__ import annotations

from dataclasses import dataclass

from audio2map.osu.schema import Beatmap, TimingPoint


def beat_length_to_bpm(beat_length_ms: float) -> float:
    if beat_length_ms <= 0:
        raise ValueError(f"invalid beat length: {beat_length_ms}")
    return 60_000.0 / beat_length_ms


def uninherited_beat_lengths(timing_points: list[TimingPoint]) -> list[float]:
    return [tp.beat_length_ms for tp in timing_points if tp.uninherited]


def uninherited_meters(timing_points: list[TimingPoint]) -> list[int]:
    return [tp.meter for tp in timing_points if tp.uninherited]


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """BPM and meter of one beatmap."""

    constant_bpm: bool
    bpm: float | None  # uninherited BPM when constant
    bpm_segments: tuple[float, ...]  # distinct uninherited BPM values (rounded 2dp)
    constant_meter: bool
    meter: int | None  # uninherited meter when constant
    meter_segments: tuple[int, ...]
    n_uninherited: int
    n_inherited: int


def summarize_timing(timing_points: list[TimingPoint]) -> TimingSummary:
    uninherited = uninherited_beat_lengths(timing_points)
    inherited_n = sum(1 for tp in timing_points if not tp.uninherited)

    meters = uninherited_meters(timing_points)
    if not uninherited:
        return TimingSummary(
            constant_bpm=False,
            bpm=None,
            bpm_segments=(),
            constant_meter=False,
            meter=None,
            meter_segments=(),
            n_uninherited=0,
            n_inherited=inherited_n,
        )

    bpms = tuple(round(beat_length_to_bpm(bl), 2) for bl in uninherited)
    bpm_unique = tuple(dict.fromkeys(bpms))  # preserve order
    constant_bpm = len(bpm_unique) <= 1

    meter_unique = tuple(dict.fromkeys(meters))
    constant_meter = len(meter_unique) <= 1

    return TimingSummary(
        constant_bpm=constant_bpm,
        bpm=bpm_unique[0] if constant_bpm else None,
        bpm_segments=bpm_unique,
        constant_meter=constant_meter,
        meter=meter_unique[0] if constant_meter else None,
        meter_segments=meter_unique,
        n_uninherited=len(uninherited),
        n_inherited=inherited_n,
    )


def summarize_beatmap_timing(beatmap: Beatmap) -> TimingSummary:
    return summarize_timing(beatmap.timing_points)
