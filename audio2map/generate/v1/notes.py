"""Floor decoded endpoints into notes. Nothing is dropped."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from audio2map.osu.parser import HIT_CIRCLE, HIT_HOLD
from audio2map.tokens.v1.codec import DecodedEvent
from audio2map.tokens.v1.structure import EndpointKind

_X = (64, 192, 320, 448)
_SAMPLE = "0:0:0:0:"
_SHORT_MS = 10


@dataclass(frozen=True, slots=True)
class V1Note:
    """One floored hit. A hold's ``end_time_ms`` may be at or before ``time_ms``."""

    time_ms: int
    col: int
    end_time_ms: int | None


def notes_from_events(events: Sequence[DecodedEvent]) -> tuple[tuple[V1Note, ...], tuple[str, ...]]:
    """Integer milliseconds, floor. A shut or unclosed hold is still a hold."""
    warnings: list[str] = []
    open_at: dict[int, int] = {}
    notes: list[V1Note] = []
    for event in events:
        start = int(math.floor(event.time_ms))
        lane = event.lane
        if event.kind is EndpointKind.TAP:
            notes.append(V1Note(start, lane, None))
        elif event.kind is EndpointKind.HEAD:
            if lane in open_at:
                warnings.append(f"column {lane} opened a hold at {start} while one was open")
            open_at[lane] = start
        elif event.kind is EndpointKind.TAIL:
            head = open_at.pop(lane, None)
            if head is None:
                warnings.append(f"column {lane} hold tail at {start} has no head")
                continue
            if start <= head:
                warnings.append(f"column {lane} hold end {start} is not after head {head}")
            notes.append(V1Note(head, lane, start))
        else:
            raise ValueError(f"unknown endpoint {event.kind}")
    for lane, head in sorted(open_at.items()):
        warnings.append(f"column {lane} hold from {head} is unclosed")
        notes.append(V1Note(head, lane, head))
    notes.sort(key=lambda note: (note.col, note.time_ms, _end(note)))
    warnings.extend(_adjacent_warnings(notes))
    return tuple(notes), tuple(warnings)


def hit_lines_from_notes(notes: Sequence[V1Note]) -> list[str]:
    return [_hit_line(note) for note in notes]


def hit_lines_from_events(events: list[DecodedEvent]) -> tuple[list[str], list[str]]:
    notes, warnings = notes_from_events(events)
    return hit_lines_from_notes(notes), list(warnings)


def _end(note: V1Note) -> int:
    return note.time_ms if note.end_time_ms is None else note.end_time_ms


def _adjacent_warnings(notes: Sequence[V1Note]) -> list[str]:
    found: list[str] = []
    previous: dict[int, tuple[int, int]] = {}
    for note in notes:
        start, end = note.time_ms, _end(note)
        prior = previous.get(note.col)
        if prior is not None:
            prev_start, prev_end = prior
            if start == prev_start:
                found.append(f"column {note.col} has two events at {start} ms")
            elif start <= prev_end + 2:
                found.append(
                    f"column {note.col} event at {start} overlaps the one ending {prev_end}"
                )
            elif start - prev_end < _SHORT_MS:
                found.append(
                    f"column {note.col} event at {start} is under {_SHORT_MS} ms after {prev_end}"
                )
        previous[note.col] = (start, end)
    return found


def _hit_line(note: V1Note) -> str:
    x = _X[note.col]
    if note.end_time_ms is None:
        return f"{x},192,{note.time_ms},{HIT_CIRCLE},0,{_SAMPLE}"
    return f"{x},192,{note.time_ms},{HIT_HOLD},0,{note.end_time_ms}:{_SAMPLE}"
