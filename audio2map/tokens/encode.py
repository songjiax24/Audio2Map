"""Encode mania notes into ROW token windows. Does not parse beatmaps from disk."""

from __future__ import annotations

from audio2map.grid import CanonicalTiming, TICKS_PER_BAR, chart_event_bar_range, ms_to_tick, split_tick
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.tokens.vocab import (
    TOKEN_BAR,
    LaneState,
    RowState,
    initial_row_from_active_hold,
    pos_to_token,
    row_state_to_token,
)


def _add_event(
    events: dict[int, dict[int, LaneState]],
    tick: int,
    col: int,
    state: LaneState,
) -> None:
    cells = events.setdefault(tick, {})
    if col in cells:
        raise ValueError(f"conflicting lane {col}: {cells[col]} vs {state}")
    cells[col] = state


def _notes_to_raw_events(
    notes: list[ManiaNote],
    timing: CanonicalTiming,
) -> dict[int, dict[int, LaneState]]:
    """``tick -> {col: LaneState}`` for TAP / HOLD_START / HOLD_END only."""
    events: dict[int, dict[int, LaneState]] = {}
    for note in notes:
        head = ms_to_tick(note.time_ms, timing)
        if note.note_type == NoteType.TAP:
            _add_event(events, head, note.col, LaneState.TAP)
        else:
            assert note.end_time_ms is not None
            _add_event(events, head, note.col, LaneState.HOLD_START)
            _add_event(events, ms_to_tick(note.end_time_ms, timing), note.col, LaneState.HOLD_END)
    return events


def _build_event_row(raw: dict[int, LaneState], hold: list[bool]) -> RowState:
    lanes: list[LaneState] = []
    for col in range(4):
        if col in raw:
            lanes.append(raw[col])
        elif hold[col]:
            lanes.append(LaneState.HOLD_ACTIVE)
        else:
            lanes.append(LaneState.EMPTY)
    a, b, c, d = lanes
    return (a, b, c, d)


def _apply_raw_events(raw: dict[int, LaneState], hold: list[bool]) -> None:
    for col, ev in raw.items():
        holding = hold[col]
        if ev == LaneState.TAP:
            if holding:
                raise ValueError(f"illegal TAP on lane {col} while holding")
        elif ev == LaneState.HOLD_START:
            if holding:
                raise ValueError(f"illegal HOLD_START on lane {col} while holding")
            hold[col] = True
        elif ev == LaneState.HOLD_END:
            if not holding:
                raise ValueError(f"illegal HOLD_END on lane {col} while not holding")
            hold[col] = False
        else:
            raise ValueError(f"expected event lane state, got {ev} on lane {col}")


def _active_hold_at_tick(
    raw_events: dict[int, dict[int, LaneState]],
    tick: int,
) -> list[bool]:
    """Replay raw events up to (but not including) ``tick`` to get active holds."""
    state = [False, False, False, False]
    for t in sorted(k for k in raw_events if k < tick):
        _apply_raw_events(raw_events[t], state)
    return state


def _event_rows(
    raw_events: dict[int, dict[int, LaneState]],
    *,
    start_tick: int,
    end_tick: int,
    active: list[bool],
) -> dict[int, RowState]:
    """``tick -> RowState``; lanes without events are EMPTY or HOLD_ACTIVE."""
    hold = list(active)
    rows: dict[int, RowState] = {}
    for tick in sorted(t for t in raw_events if start_tick <= t < end_tick):
        raw = raw_events[tick]
        row = _build_event_row(raw, hold)
        _apply_raw_events(raw, hold)
        rows[tick] = row
    return rows


def _tokens_from_event_rows(
    rows: dict[int, RowState],
    *,
    start_bar: int,
    end_bar: int,
) -> list[str]:
    by_bar: dict[int, list[tuple[int, RowState]]] = {}
    for tick, row in rows.items():
        bar, pos = split_tick(tick)
        by_bar.setdefault(bar, []).append((pos, row))

    tokens: list[str] = []
    for bar in range(start_bar, end_bar):
        tokens.append(TOKEN_BAR)
        for pos, row in sorted(by_bar.get(bar, ())):
            tokens.append(pos_to_token(pos))
            tokens.append(row_state_to_token(row))
    return tokens


def initial_row_at_bar(
    notes: list[ManiaNote],
    timing: CanonicalTiming,
    *,
    start_bar: int,
) -> RowState:
    """EMPTY/HOLD_ACTIVE ROW at the start of ``start_bar`` (window framing, not chart tokens)."""
    raw = _notes_to_raw_events(notes, timing)
    active = _active_hold_at_tick(raw, start_bar * TICKS_PER_BAR)
    return initial_row_from_active_hold(active)


def encode_notes(
    notes: list[ManiaNote],
    timing: CanonicalTiming,
    *,
    start_bar: int | None = None,
    end_bar: int | None = None,
) -> list[str]:
    """Encode notes to chart tokens: ``<BAR>`` then ``<POS> <ROW>`` pairs. No BOS/EOS/initial.

    Omit ``start_bar`` / ``end_bar`` to cover the bar-aligned event range.
    Pass both for a half-open window ``[start_bar, end_bar)``.
    """
    if (start_bar is None) != (end_bar is None):
        raise ValueError("start_bar and end_bar must both be set or both omitted")
    if start_bar is None:
        start_bar, end_bar = chart_event_bar_range(notes, timing)
    assert start_bar is not None and end_bar is not None
    if end_bar < start_bar:
        raise ValueError(f"end_bar ({end_bar}) < start_bar ({start_bar})")

    raw = _notes_to_raw_events(notes, timing)
    start_tick = start_bar * TICKS_PER_BAR
    end_tick = end_bar * TICKS_PER_BAR
    active = _active_hold_at_tick(raw, start_tick)
    rows = _event_rows(raw, start_tick=start_tick, end_tick=end_tick, active=active)
    return _tokens_from_event_rows(rows, start_bar=start_bar, end_bar=end_bar)
