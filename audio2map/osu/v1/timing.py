"""Float64 uninherited timing (red lines) for v1.

Green lines are ignored. Exact duplicate offsets reject the chart.
Near-duplicates ``0 < Δ < EPS`` are legal and reported.

``offset`` and ``beatLength`` are required. A missing ``meter`` is 4.
A missing uninherited field is a red line, which is how older timing
lines with only two fields were written. Other optional fields are not
stored on ``Redline``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from audio2map.utils.errors import (
    DUPLICATE_REDLINE,
    INVALID_TIMING_FIELD,
    NO_REDLINE,
    NON_FINITE_OFFSET,
    NONPOSITIVE_BEAT_LENGTH,
    ChartRejected,
)
from audio2map.osu.parser import read_osu_sections
from audio2map.utils.tolerance import EPS

DEFAULT_METER = 4
_OFFSET = 0
_BEAT_LENGTH = 1
_METER = 2
_UNINHERITED = 6
_RED_LINE = "1"


@dataclass(frozen=True, slots=True)
class Redline:
    """One uninherited timing point, in time order."""

    offset_ms: float
    beat_length_ms: float
    meter: int
    source_index: int


@dataclass(frozen=True, slots=True)
class RedlineParse:
    redlines: tuple[Redline, ...]
    # Consecutive pairs with 0 < later - earlier < EPS. Each pair is (earlier, later) ms.
    near_duplicate_pairs: tuple[tuple[float, float], ...]


def _field(parts: list[str], index: int, default: str) -> str:
    if index >= len(parts):
        return default
    text = parts[index].strip()
    return text if text else default


def _parse_line(source_index: int, line: str) -> Redline | None:
    parts = line.split(",")
    if len(parts) <= _BEAT_LENGTH or not parts[_OFFSET].strip() or not parts[_BEAT_LENGTH].strip():
        raise ChartRejected(INVALID_TIMING_FIELD, f"offset and beatLength are required: {line}")
    try:
        offset = float(parts[_OFFSET])
        beat_length = float(parts[_BEAT_LENGTH])
    except ValueError as exc:
        raise ChartRejected(
            INVALID_TIMING_FIELD,
            f"offset or beatLength is not a number: {line}",
        ) from exc
    if not math.isfinite(offset):
        raise ChartRejected(NON_FINITE_OFFSET, f"offset={offset}")
    # Green lines store a negative beat length. Skip them before the > 0 check.
    if _field(parts, _UNINHERITED, _RED_LINE) != _RED_LINE:
        return None
    if not math.isfinite(beat_length) or beat_length <= 0.0:
        raise ChartRejected(NONPOSITIVE_BEAT_LENGTH, f"offset={offset} beatLength={beat_length}")
    meter_text = _field(parts, _METER, "")
    if not meter_text:
        meter = DEFAULT_METER
    else:
        try:
            meter = int(float(meter_text))
        except ValueError:
            meter = DEFAULT_METER
    return Redline(offset, beat_length, meter, source_index)


def parse_redlines(lines: list[str]) -> RedlineParse:
    """Read uninherited timing points as float64. Reject exact duplicates."""
    found: list[Redline] = []
    for index, line in enumerate(lines):
        red = _parse_line(index, line)
        if red is not None:
            found.append(red)
    if not found:
        raise ChartRejected(NO_REDLINE, "no uninherited timing point")
    found.sort(key=lambda item: (item.offset_ms, item.source_index))
    for prev, nxt in zip(found, found[1:]):
        if nxt.offset_ms == prev.offset_ms:
            raise ChartRejected(DUPLICATE_REDLINE, f"offset={prev.offset_ms}")
    near = [
        (prev.offset_ms, nxt.offset_ms)
        for prev, nxt in zip(found, found[1:])
        if 0.0 < nxt.offset_ms - prev.offset_ms < EPS
    ]
    return RedlineParse(redlines=tuple(found), near_duplicate_pairs=tuple(near))


def parse_redlines_file(path: Path | str) -> RedlineParse:
    sections = read_osu_sections(path)
    return parse_redlines(sections.get("[TimingPoints]", []))
