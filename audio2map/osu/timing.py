"""Timing point analysis (BPM, constant vs variable tempo)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from audio2map.osu.parser import parse_sections, parse_timing_points
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
    """BPM / meter / tempo characteristics of one beatmap."""

    constant_bpm: bool
    bpm_primary: float | None  # first uninherited BPM when constant; else mean of segments
    bpm_segments: tuple[float, ...]  # distinct uninherited BPM values (rounded 2dp)
    constant_meter: bool
    meter_primary: int | None  # first uninherited meter when constant
    meter_segments: tuple[int, ...]
    n_uninherited: int
    n_inherited: int
    has_negative_timing: bool


def summarize_timing(timing_points: list[TimingPoint]) -> TimingSummary:
    uninherited = uninherited_beat_lengths(timing_points)
    inherited_n = sum(1 for tp in timing_points if not tp.uninherited)
    negative = any(tp.offset_ms < 0 for tp in timing_points)

    meters = uninherited_meters(timing_points)
    if not uninherited:
        return TimingSummary(
            constant_bpm=False,
            bpm_primary=None,
            bpm_segments=(),
            constant_meter=False,
            meter_primary=None,
            meter_segments=(),
            n_uninherited=0,
            n_inherited=inherited_n,
            has_negative_timing=negative,
        )

    bpms = tuple(round(beat_length_to_bpm(bl), 2) for bl in uninherited)
    bpm_unique = tuple(dict.fromkeys(bpms))  # preserve order
    constant_bpm = len(bpm_unique) <= 1
    bpm_primary = bpm_unique[0] if constant_bpm else round(sum(bpms) / len(bpms), 2)

    meter_unique = tuple(dict.fromkeys(meters))
    constant_meter = len(meter_unique) <= 1
    meter_primary = meter_unique[0] if constant_meter else None

    return TimingSummary(
        constant_bpm=constant_bpm,
        bpm_primary=bpm_primary,
        bpm_segments=bpm_unique,
        constant_meter=constant_meter,
        meter_primary=meter_primary,
        meter_segments=meter_unique,
        n_uninherited=len(uninherited),
        n_inherited=inherited_n,
        has_negative_timing=negative,
    )


def parse_timing_points_only(path: Path | str) -> list[TimingPoint]:
    """Lightweight: read only ``[TimingPoints]`` from an ``.osu`` file."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    sections = parse_sections(text)
    return parse_timing_points(sections.get("[TimingPoints]", []))


def summarize_beatmap_timing(beatmap: Beatmap) -> TimingSummary:
    return summarize_timing(beatmap.timing_points)
