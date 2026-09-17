"""Decode chart tokens into mania notes. Does not parse beatmaps from disk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from audio2map.grid import CanonicalTiming, TICKS_PER_BAR, tick_to_ms
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.tokens.vocab import (
    TOKEN_BAR,
    TOKEN_POS_PREFIX,
    TOKEN_ROW_PREFIX,
    LaneState,
    RowState,
    pos_from_token,
    row_state_from_token,
)

DecodeIssueKind = Literal["missing_hold_head", "missing_hold_tail"]


@dataclass(frozen=True, slots=True)
class DecodeIssue:
    kind: DecodeIssueKind
    tick: int
    col: int


@dataclass(frozen=True, slots=True)
class DecodeResult:
    notes: list[ManiaNote]
    issues: list[DecodeIssue]


def _rows_from_tokens(
    tokens: list[str],
    *,
    start_bar: int,
) -> tuple[list[tuple[int, RowState]], int]:
    """Parse ``<BAR> (<POS> <ROW>)*`` into ``(tick, row)`` and exclusive ``end_bar``."""
    rows: list[tuple[int, RowState]] = []
    current_bar = start_bar - 1
    pending_pos: int | None = None
    seen_pos: set[int] = set()
    for tok in tokens:
        if tok == TOKEN_BAR:
            if pending_pos is not None:
                raise ValueError("POS without ROW")
            current_bar += 1
            seen_pos = set()
            continue
        if tok.startswith(TOKEN_POS_PREFIX):
            if current_bar < start_bar:
                raise ValueError(f"POS before BAR: {tok}")
            if pending_pos is not None:
                raise ValueError(f"POS without ROW before {tok}")
            pos = pos_from_token(tok)
            if pos in seen_pos:
                raise ValueError(f"duplicate POS {pos} in bar {current_bar}")
            seen_pos.add(pos)
            pending_pos = pos
            continue
        if tok.startswith(TOKEN_ROW_PREFIX):
            if pending_pos is None:
                raise ValueError(f"ROW without POS: {tok}")
            tick = current_bar * TICKS_PER_BAR + pending_pos
            rows.append((tick, row_state_from_token(tok)))
            pending_pos = None
            continue
        raise ValueError(f"unexpected chart token: {tok!r}")
    if pending_pos is not None:
        raise ValueError("POS without ROW")
    return rows, current_bar + 1


def _notes_from_rows(
    rows: list[tuple[int, RowState]],
    timing: CanonicalTiming,
    *,
    start_tick: int,
    end_tick: int,
    initial_row: RowState,
) -> DecodeResult:
    notes: list[ManiaNote] = []
    issues: list[DecodeIssue] = []
    open_holds: dict[int, int] = {}
    for col, state in enumerate(initial_row):
        if state == LaneState.HOLD_ACTIVE:
            open_holds[col] = start_tick
            issues.append(DecodeIssue("missing_hold_head", start_tick, col))
        elif state != LaneState.EMPTY:
            raise ValueError(f"window initial ROW must be EMPTY/HOLD_ACTIVE only: {initial_row}")

    for tick, row in rows:
        time_ms = tick_to_ms(tick, timing)
        for col, state in enumerate(row):
            holding = col in open_holds
            if state == LaneState.EMPTY:
                if holding:
                    raise ValueError(
                        f"illegal EMPTY on lane {col} while holding (tick {tick})"
                    )
            elif state == LaneState.HOLD_ACTIVE:
                if not holding:
                    raise ValueError(
                        f"illegal HOLD_ACTIVE on lane {col} while not holding (tick {tick})"
                    )
            elif state == LaneState.TAP:
                if holding:
                    raise ValueError(f"illegal TAP on lane {col} while holding (tick {tick})")
                notes.append(ManiaNote(time_ms=time_ms, col=col, note_type=NoteType.TAP))
            elif state == LaneState.HOLD_START:
                if holding:
                    raise ValueError(
                        f"illegal HOLD_START on lane {col} while holding (tick {tick})"
                    )
                open_holds[col] = tick
            elif state == LaneState.HOLD_END:
                if not holding:
                    raise ValueError(
                        f"illegal HOLD_END on lane {col} while not holding (tick {tick})"
                    )
                hold_start = open_holds.pop(col)
                if tick > hold_start:
                    notes.append(
                        ManiaNote(
                            time_ms=tick_to_ms(hold_start, timing),
                            col=col,
                            note_type=NoteType.HOLD,
                            end_time_ms=time_ms,
                        )
                    )

    for col, hold_start in sorted(open_holds.items()):
        issues.append(DecodeIssue("missing_hold_tail", hold_start, col))
        if end_tick > hold_start:
            notes.append(
                ManiaNote(
                    time_ms=tick_to_ms(hold_start, timing),
                    col=col,
                    note_type=NoteType.HOLD,
                    end_time_ms=tick_to_ms(end_tick, timing),
                )
            )

    notes.sort(key=lambda n: (n.time_ms, n.col))
    return DecodeResult(notes=notes, issues=issues)


def tokens_to_notes(
    tokens: list[str],
    timing: CanonicalTiming,
    *,
    start_bar: int = 0,
    initial_row: RowState | None = None,
) -> DecodeResult:
    """Parse chart tokens into notes plus issues.

    ``initial_row`` is the window-start prompt (EMPTY / HOLD_ACTIVE). Holds already
    active there are clipped to ``start_bar`` (missing head). Holds still open after
    the last ``<BAR>`` are clipped to that bar's end (missing tail). Every chart-row
    lane is checked against hold state. Conflicts and syntax errors raise.
    """
    if initial_row is None:
        initial_row = (LaneState.EMPTY, LaneState.EMPTY, LaneState.EMPTY, LaneState.EMPTY)
    rows, end_bar = _rows_from_tokens(tokens, start_bar=start_bar)
    return _notes_from_rows(
        rows,
        timing,
        start_tick=start_bar * TICKS_PER_BAR,
        end_tick=end_bar * TICKS_PER_BAR,
        initial_row=initial_row,
    )
