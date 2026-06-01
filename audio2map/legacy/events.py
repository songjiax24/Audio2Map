"""Sparse absolute chart events for autoregressive models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from audio2map.osu.schema import Beatmap, ManiaNote, NoteType

DEFAULT_HOP_MS = 10


class EventType(IntEnum):
    TAP = 0
    HOLD = 1


@dataclass(frozen=True, slots=True)
class ChartEvent:
    """One sparse chart event on an absolute frame grid.

    All times are quantised as ``frame = time_ms // hop_ms`` (default 10 ms).
    No delta-time or relative hold length — only absolute indices.
    """

    frame: int
    col: int
    type_id: EventType
    end_frame: int | None = None
    hop_ms: int = DEFAULT_HOP_MS

    def __post_init__(self) -> None:
        if self.col < 0 or self.col > 3:
            raise ValueError(f"4K column must be 0–3, got {self.col}")
        if self.frame < 0:
            raise ValueError(f"frame must be non-negative, got {self.frame}")
        if self.type_id == EventType.HOLD:
            if self.end_frame is None:
                raise ValueError("hold events require end_frame")
            if self.end_frame < self.frame:
                raise ValueError(
                    f"hold end_frame ({self.end_frame}) must be >= frame ({self.frame})"
                )
        elif self.end_frame is not None:
            raise ValueError("tap events must not have end_frame")

    @property
    def time_ms(self) -> int:
        return self.frame * self.hop_ms

    @property
    def end_time_ms(self) -> int | None:
        if self.end_frame is None:
            return None
        return self.end_frame * self.hop_ms


def ms_to_frame(time_ms: int, hop_ms: int = DEFAULT_HOP_MS) -> int:
    return time_ms // hop_ms


def notes_to_events(
    notes: list[ManiaNote],
    *,
    hop_ms: int = DEFAULT_HOP_MS,
) -> list[ChartEvent]:
    """Convert notes to sparse events sorted by ``(frame, col)``."""
    events: list[ChartEvent] = []
    for note in notes:
        frame = ms_to_frame(note.time_ms, hop_ms)
        if note.note_type == NoteType.TAP:
            events.append(
                ChartEvent(frame=frame, col=note.col, type_id=EventType.TAP, hop_ms=hop_ms),
            )
        else:
            assert note.end_time_ms is not None
            events.append(
                ChartEvent(
                    frame=frame,
                    col=note.col,
                    type_id=EventType.HOLD,
                    end_frame=ms_to_frame(note.end_time_ms, hop_ms),
                    hop_ms=hop_ms,
                ),
            )
    events.sort(key=lambda e: (e.frame, e.col, e.type_id))
    return events


def beatmap_to_events(
    beatmap: Beatmap,
    *,
    hop_ms: int = DEFAULT_HOP_MS,
) -> list[ChartEvent]:
    return notes_to_events(beatmap.notes, hop_ms=hop_ms)


def event_ar_tokens(event: ChartEvent) -> tuple[int, ...]:
    """Flatten one event into the AR token tuple (absolute values only).

    TAP:  ``(frame, col, TYPE_TAP)``
    HOLD: ``(frame, col, TYPE_HOLD, end_frame)``
    """
    if event.type_id == EventType.HOLD:
        assert event.end_frame is not None
        return (event.frame, event.col, int(EventType.HOLD), event.end_frame)
    return (event.frame, event.col, int(EventType.TAP))


def events_to_ar_sequence(events: list[ChartEvent]) -> list[tuple[int, ...]]:
    return [event_ar_tokens(e) for e in events]


def count_event_types(events: list[ChartEvent]) -> dict[str, int]:
    taps = sum(1 for e in events if e.type_id == EventType.TAP)
    holds = sum(1 for e in events if e.type_id == EventType.HOLD)
    return {"tap": taps, "hold": holds, "total": len(events)}
