"""Optional dense frame view (debug / visualisation — not the primary training target)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from audio2map.osu.events import ChartEvent, DEFAULT_HOP_MS, EventType, ms_to_frame
from audio2map.osu.schema import Beatmap, ManiaNote, NoteType

__all__ = [
    "CellState",
    "FrameChart",
    "beatmap_to_frames",
    "count_active_cells",
    "events_to_frames",
    "notes_to_frames",
]


class CellState(IntEnum):
    EMPTY = 0
    TAP = 1
    HOLD = 2


@dataclass(slots=True)
class FrameChart:
    hop_ms: int
    grid: np.ndarray  # (T, 4)

    @property
    def num_frames(self) -> int:
        return int(self.grid.shape[0])

    def column_states(self, frame: int) -> tuple[int, int, int, int]:
        row = self.grid[frame]
        return (int(row[0]), int(row[1]), int(row[2]), int(row[3]))


def duration_ms_from_notes(notes: list[ManiaNote], *, padding_ms: int = 500) -> int:
    if not notes:
        return 0
    last = max(
        n.end_time_ms if n.note_type == NoteType.HOLD and n.end_time_ms is not None else n.time_ms
        for n in notes
    )
    return last + padding_ms


def events_to_frames(
    events: list[ChartEvent],
    *,
    duration_ms: int | None = None,
    hop_ms: int = DEFAULT_HOP_MS,
) -> FrameChart:
    """Expand sparse events to a dense grid (for visualisation / checks)."""
    if hop_ms <= 0:
        raise ValueError(f"hop_ms must be positive, got {hop_ms}")

    if duration_ms is None:
        if events:
            last_ms = max(
                (e.end_frame if e.end_frame is not None else e.frame) * hop_ms for e in events
            )
            duration_ms = last_ms + 500
        else:
            duration_ms = hop_ms

    num_frames = max(1, (duration_ms + hop_ms - 1) // hop_ms)
    grid = np.zeros((num_frames, 4), dtype=np.int8)

    for event in events:
        if event.frame >= num_frames:
            continue
        col = event.col
        if event.type_id == EventType.TAP:
            if grid[event.frame, col] == CellState.EMPTY:
                grid[event.frame, col] = CellState.TAP
        else:
            end_f = min(event.end_frame or event.frame, num_frames - 1)
            for f in range(event.frame, end_f + 1):
                grid[f, col] = CellState.HOLD

    return FrameChart(hop_ms=hop_ms, grid=grid)


def notes_to_frames(
    notes: list[ManiaNote],
    *,
    duration_ms: int | None = None,
    hop_ms: int = DEFAULT_HOP_MS,
) -> FrameChart:
    from audio2map.osu.events import notes_to_events

    events = notes_to_events(notes, hop_ms=hop_ms)
    dur = duration_ms if duration_ms is not None else duration_ms_from_notes(notes)
    return events_to_frames(events, duration_ms=dur, hop_ms=hop_ms)


def beatmap_to_frames(
    beatmap: Beatmap,
    *,
    hop_ms: int = DEFAULT_HOP_MS,
    padding_ms: int = 500,
) -> FrameChart:
    dur = max(beatmap.duration_ms, duration_ms_from_notes(beatmap.notes, padding_ms=padding_ms))
    from audio2map.osu.events import beatmap_to_events

    return events_to_frames(beatmap_to_events(beatmap, hop_ms=hop_ms), duration_ms=dur, hop_ms=hop_ms)


def count_active_cells(chart: FrameChart) -> dict[str, int]:
    g = chart.grid
    return {
        "empty": int(np.sum(g == CellState.EMPTY)),
        "tap": int(np.sum(g == CellState.TAP)),
        "hold": int(np.sum(g == CellState.HOLD)),
    }
