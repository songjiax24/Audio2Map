"""v1 cursor and constrained-decoding legality.

Constraint (b): while a hold is open, ``<BEAT>`` is legal only when the next
beat still has a legal grid point. For every beat whose start is inside
``[0, T_audio)`` that point includes ``0/1`` at the beat start.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from audio2map.grid.v1.beatgrid import BeatGrid, GridPoint
from audio2map.utils.errors import OPEN_HOLD_AT_END, ChartRejected
from audio2map.tokens.errors import IllegalTokenError
from audio2map.tokens.vocab import (
    LaneState,
    RowState,
    apply_row,
    is_legal_row,
    row_state_from_token,
    row_state_to_token,
)
from audio2map.tokens.v1.vocab import (
    TOKEN_BEAT,
    TOKEN_EOS,
    event_rows,
    parse_pos_token,
    pos_token,
)

_EVENT_ROWS = event_rows()


def _before(left: GridPoint, right: GridPoint) -> bool:
    if left.beat_id != right.beat_id:
        return left.beat_id < right.beat_id
    return left.numer * right.denom < right.numer * left.denom


def _fraction_after(numer: int, denom: int, previous: tuple[int, int] | None) -> bool:
    if previous is None:
        return True
    prev_n, prev_d = previous
    return prev_n * denom < numer * prev_d


@dataclass
class _GridFacts:
    points: dict[tuple[int, int, int], GridPoint]
    last: GridPoint | None

    @classmethod
    def build(cls, grid: BeatGrid) -> _GridFacts:
        points: dict[tuple[int, int, int], GridPoint] = {}
        last: GridPoint | None = None
        for beat_id, beat_points in enumerate(grid.points):
            for point in beat_points:
                points[(beat_id, point.numer, point.denom)] = point
                if last is None or _before(last, point):
                    last = point
        return cls(points, last)


@dataclass
class Cursor:
    beat_id: int = 0
    expect_row: bool = False
    pending: GridPoint | None = None
    last_fraction: tuple[int, int] | None = None
    active: list[bool] = field(default_factory=lambda: [False, False, False, False])
    finished: bool = False
    _facts: _GridFacts | None = field(default=None, repr=False, compare=False)

    def clone(self) -> Cursor:
        return Cursor(
            beat_id=self.beat_id,
            expect_row=self.expect_row,
            pending=self.pending,
            last_fraction=self.last_fraction,
            active=list(self.active),
            finished=self.finished,
            _facts=self._facts,
        )


def initial_cursor() -> Cursor:
    return Cursor()


def _facts(cursor: Cursor, grid: BeatGrid) -> _GridFacts:
    if cursor._facts is None:
        cursor._facts = _GridFacts.build(grid)
    return cursor._facts


def _row_ok(row: RowState, active: list[bool], point: GridPoint, last: GridPoint | None) -> bool:
    if not is_legal_row(row, active):
        return False
    on_last = last is not None and point.beat_id == last.beat_id and point.cell == last.cell
    has_later = last is not None and _before(point, last)
    for lane, state in enumerate(row):
        if state == LaneState.HOLD_START and (on_last or not has_later):
            return False
        if on_last and active[lane] and state != LaneState.HOLD_END:
            return False
    return True


def legal_tokens(cursor: Cursor, grid: BeatGrid) -> set[str]:
    """Tokens legal in ``cursor``. ``<BOS>`` is a prefix and is never included."""
    if cursor.finished:
        return set()
    n_beats = len(grid.beats)
    if cursor.expect_row:
        if cursor.pending is None:
            return set()
        last = _facts(cursor, grid).last
        return {
            row_state_to_token(row)
            for row in _EVENT_ROWS
            if _row_ok(row, cursor.active, cursor.pending, last)
        }
    if cursor.beat_id >= n_beats:
        if any(cursor.active):
            return set()
        return {TOKEN_EOS}
    allowed = {
        pos_token(point.numer, point.denom)
        for point in grid.points[cursor.beat_id]
        if _fraction_after(point.numer, point.denom, cursor.last_fraction)
    }
    hold_open = any(cursor.active)
    next_beat = cursor.beat_id + 1
    if next_beat < n_beats:
        if (not hold_open) or grid.beat_has_legal_point(next_beat):
            allowed.add(TOKEN_BEAT)
    elif not hold_open:
        allowed.add(TOKEN_BEAT)
    return allowed


def step(cursor: Cursor, token: str, grid: BeatGrid) -> None:
    """Apply ``token`` only after it is legal. The cursor is unchanged on failure.

    ``legal_tokens`` only returns a set. An open hold at the end of the grid
    is a bad chart, so ``step`` rejects it before changing beat or hold state.
    """
    facts = _facts(cursor, grid)
    if cursor.beat_id >= len(grid.beats) and any(cursor.active):
        raise ChartRejected(OPEN_HOLD_AT_END, "reached G_end with an open hold")
    allowed = legal_tokens(cursor, grid)
    if token not in allowed:
        raise IllegalTokenError(token)
    if token == TOKEN_EOS:
        cursor.finished = True
        return
    if token == TOKEN_BEAT:
        cursor.beat_id += 1
        cursor.last_fraction = None
        cursor.pending = None
        return
    if token.startswith("<POS_"):
        numer, denom = parse_pos_token(token)
        point = facts.points.get((cursor.beat_id, numer, denom))
        if point is None:
            raise IllegalTokenError(token)
        cursor.pending = point
        cursor.expect_row = True
        return
    row = row_state_from_token(token)
    if cursor.pending is None or not _row_ok(row, cursor.active, cursor.pending, facts.last):
        raise IllegalTokenError(token)
    apply_row(row, cursor.active)
    cursor.last_fraction = (cursor.pending.numer, cursor.pending.denom)
    cursor.pending = None
    cursor.expect_row = False
