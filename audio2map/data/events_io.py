"""Serialize chart events for numpy storage."""

from __future__ import annotations

import numpy as np

from audio2map.osu.events import ChartEvent, EventType


def events_to_arrays(events: list[ChartEvent]) -> dict[str, np.ndarray]:
    n = len(events)
    frames = np.empty(n, dtype=np.int32)
    cols = np.empty(n, dtype=np.int8)
    types = np.empty(n, dtype=np.int8)
    end_frames = np.full(n, -1, dtype=np.int32)

    for i, e in enumerate(events):
        frames[i] = e.frame
        cols[i] = e.col
        types[i] = int(e.type_id)
        if e.type_id == EventType.HOLD and e.end_frame is not None:
            end_frames[i] = e.end_frame

    return {
        "event_frame": frames,
        "event_col": cols,
        "event_type": types,
        "event_end_frame": end_frames,
    }


def arrays_to_events(
    frames: np.ndarray,
    cols: np.ndarray,
    types: np.ndarray,
    end_frames: np.ndarray,
    *,
    hop_ms: int,
) -> list[ChartEvent]:
    events: list[ChartEvent] = []
    for frame, col, type_id, end_frame in zip(frames, cols, types, end_frames, strict=True):
        if int(type_id) == int(EventType.HOLD):
            events.append(
                ChartEvent(
                    frame=int(frame),
                    col=int(col),
                    type_id=EventType.HOLD,
                    end_frame=int(end_frame),
                    hop_ms=hop_ms,
                )
            )
        else:
            events.append(
                ChartEvent(
                    frame=int(frame),
                    col=int(col),
                    type_id=EventType.TAP,
                    hop_ms=hop_ms,
                )
            )
    return events
