"""Cursor time for teacher forcing. The time is the state before the target.

``cursor_times_ms`` lines up with ``token_ids[:-1]``: index ``i`` is the
cursor time used while predicting ``tokens[i + 1]``. ``<BOS>`` is not
stepped. A ``<POS>`` is not applied until after its own prediction, so
that step still sees the previous position. The following ``<ROW>`` sees
the position that token just committed.
"""

from __future__ import annotations

from audio2map.grid.v1.beatgrid import BeatGrid
from audio2map.tokens.errors import IllegalTokenError
from audio2map.tokens.v1.state import Cursor, initial_cursor, step
from audio2map.tokens.v1.vocab import TOKEN_BOS


def state_time_ms(cursor: Cursor, grid: BeatGrid) -> float:
    """Time of ``cursor`` before the next token is applied."""
    if cursor.pending is not None:
        return float(cursor.pending.time_ms)
    if cursor.beat_id < len(grid.beats):
        beat = grid.beats[cursor.beat_id]
        if cursor.last_fraction is not None:
            numer, denom = cursor.last_fraction
            return beat.start_ms + (numer / denom) * beat.nominal_length_ms
        return beat.start_ms
    return grid.coverage_end_ms


def cursor_times_ms(tokens: list[str], grid: BeatGrid) -> list[float]:
    """One time per target, aligned with ``tokens[:-1]``."""
    if not tokens or tokens[0] != TOKEN_BOS:
        raise IllegalTokenError(tokens[0] if tokens else "")
    cursor = initial_cursor()
    times: list[float] = []
    for token in tokens[1:]:
        times.append(state_time_ms(cursor, grid))
        step(cursor, token, grid)
    return times
