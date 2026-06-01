"""Archived v1 helpers (10 ms sparse events). Not used by v2 training.

The v2 pipeline uses ``audio2map.osu.row_tokens`` instead. These modules remain
for optional debugging and parser regression tests only.
"""

from audio2map.legacy.events import (
    ChartEvent,
    EventType,
    beatmap_to_events,
    count_event_types,
    event_ar_tokens,
    events_to_ar_sequence,
    ms_to_frame,
    notes_to_events,
)
from audio2map.legacy.frames import (
    CellState,
    FrameChart,
    beatmap_to_frames,
    events_to_frames,
    notes_to_frames,
)

__all__ = [
    "CellState",
    "ChartEvent",
    "EventType",
    "FrameChart",
    "beatmap_to_events",
    "beatmap_to_frames",
    "count_event_types",
    "event_ar_tokens",
    "events_to_ar_sequence",
    "events_to_frames",
    "ms_to_frame",
    "notes_to_events",
    "notes_to_frames",
]
