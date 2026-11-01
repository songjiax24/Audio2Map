"""v1 cursor and constrained-decoding legality.

Constraint (b): while a hold is open, ``<BEAT>`` is legal only when the next
beat still has a legal grid point. For every beat whose start is inside
``[0, T_audio)`` that point includes ``0/1`` at the beat start.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from audio2map.grid.v1.beatgrid import BeatGrid, GridPoint
from audio2map.grid.v1.positions import v80_cells
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
    build_vocab,
    event_rows,
    invert_vocab,
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


def _token_allowed(cursor: Cursor, token: str, grid: BeatGrid) -> bool:
    """Whether ``token`` is in ``legal_tokens`` at ``cursor``. One token, no list."""
    if cursor.finished:
        return False
    n_beats = len(grid.beats)
    if cursor.expect_row:
        if cursor.pending is None or not token.startswith("<ROW_"):
            return False
        try:
            row = row_state_from_token(token)
        except ValueError:
            return False
        return _row_ok(row, cursor.active, cursor.pending, _facts(cursor, grid).last)
    if cursor.beat_id >= n_beats:
        return token == TOKEN_EOS and not any(cursor.active)
    if token == TOKEN_BEAT:
        hold_open = any(cursor.active)
        next_beat = cursor.beat_id + 1
        if next_beat < n_beats:
            return (not hold_open) or grid.beat_has_legal_point(next_beat)
        return not hold_open
    if not token.startswith("<POS_"):
        return False
    try:
        numer, denom = parse_pos_token(token)
    except ValueError:
        return False
    if not _fraction_after(numer, denom, cursor.last_fraction):
        return False
    return (cursor.beat_id, numer, denom) in _facts(cursor, grid).points


def step(
    cursor: Cursor,
    token: str,
    grid: BeatGrid,
    *,
    allowed: set[str] | None = None,
) -> None:
    """Apply ``token`` only after it is legal. The cursor is unchanged on failure.

    Pass ``allowed`` from a ``legal_tokens`` call on this same cursor so the
    check is not repeated. Omit it and ``step`` checks this token alone, with
    the same membership. An open hold at the end of the grid is a bad chart,
    so ``step`` rejects it before changing beat or hold state.
    """
    facts = _facts(cursor, grid)
    if cursor.beat_id >= len(grid.beats) and any(cursor.active):
        raise ChartRejected(OPEN_HOLD_AT_END, "reached G_end with an open hold")
    if allowed is None:
        legal = _token_allowed(cursor, token, grid)
    else:
        legal = token in allowed
    if not legal:
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


_POS_IDS: dict[tuple[int, int], int] | None = None
_ROW_IDS: tuple[tuple[RowState, int], ...] | None = None
_EOS_ID = 0
_BEAT_ID = 0
_VOCAB_SIZE = 0


def _id_tables() -> tuple[dict[tuple[int, int], int], tuple[tuple[RowState, int], ...]]:
    """POS and ROW ids in the same order ``legal_tokens`` would name them."""
    global _POS_IDS, _ROW_IDS, _EOS_ID, _BEAT_ID, _VOCAB_SIZE
    if _POS_IDS is None or _ROW_IDS is None:
        vocab = build_vocab()
        _POS_IDS = {
            (numer, denom): vocab[pos_token(numer, denom)]
            for numer, denom in v80_cells()
        }
        _ROW_IDS = tuple(
            (row, vocab[row_state_to_token(row)]) for row in event_rows()
        )
        _EOS_ID = vocab[TOKEN_EOS]
        _BEAT_ID = vocab[TOKEN_BEAT]
        _VOCAB_SIZE = len(vocab)
    return _POS_IDS, _ROW_IDS


def legal_id_array(cursor: Cursor, grid: BeatGrid) -> np.ndarray:
    """Legal token ids at ``cursor``. Same set as ``legal_tokens``."""
    pos_ids, row_ids = _id_tables()
    if cursor.finished:
        return np.empty(0, dtype=np.int16)
    n_beats = len(grid.beats)
    if cursor.expect_row:
        if cursor.pending is None:
            return np.empty(0, dtype=np.int16)
        last = _facts(cursor, grid).last
        chosen = [
            row_id
            for row, row_id in row_ids
            if _row_ok(row, cursor.active, cursor.pending, last)
        ]
        return np.asarray(chosen, dtype=np.int16)
    if cursor.beat_id >= n_beats:
        if any(cursor.active):
            return np.empty(0, dtype=np.int16)
        return np.asarray([_EOS_ID], dtype=np.int16)
    chosen = [
        pos_ids[(point.numer, point.denom)]
        for point in grid.points[cursor.beat_id]
        if _fraction_after(point.numer, point.denom, cursor.last_fraction)
    ]
    hold_open = any(cursor.active)
    next_beat = cursor.beat_id + 1
    if next_beat < n_beats:
        if (not hold_open) or grid.beat_has_legal_point(next_beat):
            chosen.append(_BEAT_ID)
    elif not hold_open:
        chosen.append(_BEAT_ID)
    return np.asarray(chosen, dtype=np.int16)


def legal_rows(token_ids: Sequence[int], grid: BeatGrid) -> tuple[np.ndarray, ...]:
    """Legal ids for each target ``token_ids[1:]``.

    The gold id has to be in that set. Validation reuses the rows for every
    condition branch; the branch only changes the logits.
    """
    _id_tables()
    names = invert_vocab()
    present = np.zeros(_VOCAB_SIZE, dtype=bool)
    cursor = initial_cursor()
    rows: list[np.ndarray] = []
    for gold in token_ids[1:]:
        allowed = legal_id_array(cursor, grid)
        if allowed.size == 0:
            raise RuntimeError("empty legal token set on a gold target")
        present[allowed] = True
        if not present[gold]:
            present[allowed] = False
            raise RuntimeError(f"gold token {names[int(gold)]} is not legal")
        present[allowed] = False
        rows.append(allowed)
        step(cursor, names[int(gold)], grid, allowed={names[int(gold)]})
    return tuple(rows)
