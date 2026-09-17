"""ROW token vocabulary and lane-state grammar."""

from __future__ import annotations

from enum import IntEnum
from itertools import product

from audio2map.grid import TICKS_PER_BAR

TOKEN_PAD = "<PAD>"
TOKEN_BOS = "<BOS>"
TOKEN_EOS = "<EOS>"
TOKEN_BAR = "<BAR>"
TOKEN_POS_PREFIX = "<POS_"
TOKEN_ROW_PREFIX = "<ROW_"

# Bump when token grammar / vocab construction changes (checkpoint metadata).
TOKENIZER_VERSION = 1


class LaneState(IntEnum):
    EMPTY = 0
    TAP = 1
    HOLD_START = 2
    HOLD_ACTIVE = 3
    HOLD_END = 4


EVENT_LANE_STATES = frozenset({LaneState.TAP, LaneState.HOLD_START, LaneState.HOLD_END})


RowState = tuple[LaneState, LaneState, LaneState, LaneState]


def row_state_to_token(row: RowState) -> str:
    return TOKEN_ROW_PREFIX + "".join(str(int(s)) for s in row) + ">"


def row_state_from_token(token: str) -> RowState:
    if not token.startswith(TOKEN_ROW_PREFIX) or not token.endswith(">"):
        raise ValueError(f"invalid ROW token: {token!r}")
    digits = token[len(TOKEN_ROW_PREFIX) : -1]
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError(f"invalid ROW token: {token!r}")
    try:
        return (
            LaneState(int(digits[0])),
            LaneState(int(digits[1])),
            LaneState(int(digits[2])),
            LaneState(int(digits[3])),
        )
    except ValueError:
        raise ValueError(f"lane state out of range in {token!r}") from None


def pos_to_token(pos: int) -> str:
    if pos < 0 or pos >= TICKS_PER_BAR:
        raise ValueError(f"pos must be 0..{TICKS_PER_BAR - 1}, got {pos}")
    return f"{TOKEN_POS_PREFIX}{pos}>"


def pos_from_token(token: str) -> int:
    if not token.startswith(TOKEN_POS_PREFIX) or not token.endswith(">"):
        raise ValueError(f"invalid POS token: {token!r}")
    pos = int(token[len(TOKEN_POS_PREFIX) : -1])
    if pos < 0 or pos >= TICKS_PER_BAR:
        raise ValueError(f"pos out of range in {token!r}")
    return pos


def is_event_row(row: RowState) -> bool:
    return any(s in EVENT_LANE_STATES for s in row)


def is_legal_row(row: RowState, active_hold: list[bool]) -> bool:
    if not is_event_row(row):
        return False
    for lane, state in enumerate(row):
        if state == LaneState.EMPTY and active_hold[lane]:
            return False
        if state == LaneState.TAP and active_hold[lane]:
            return False
        if state == LaneState.HOLD_START and active_hold[lane]:
            return False
        if state == LaneState.HOLD_ACTIVE and not active_hold[lane]:
            return False
        if state == LaneState.HOLD_END and not active_hold[lane]:
            return False
    return True


def apply_row(row: RowState, active_hold: list[bool]) -> None:
    for lane, state in enumerate(row):
        if state == LaneState.HOLD_START:
            active_hold[lane] = True
        elif state == LaneState.HOLD_END:
            active_hold[lane] = False


def initial_row_from_active_hold(active_hold: list[bool]) -> RowState:
    a, b, c, d = (
        LaneState.HOLD_ACTIVE if holding else LaneState.EMPTY for holding in active_hold
    )
    return (a, b, c, d)


def active_hold_from_initial_row(row: RowState) -> list[bool]:
    if is_event_row(row):
        raise ValueError(f"window initial ROW must be EMPTY/HOLD_ACTIVE only: {row}")
    return [s == LaneState.HOLD_ACTIVE for s in row]


def build_vocab() -> dict[str, int]:
    vocab: dict[str, int] = {
        TOKEN_PAD: 0,
        TOKEN_BOS: 1,
        TOKEN_EOS: 2,
        TOKEN_BAR: 3,
    }
    idx = 4
    for pos in range(TICKS_PER_BAR):
        vocab[pos_to_token(pos)] = idx
        idx += 1
    for a, b, c, d in product(LaneState, repeat=4):
        vocab[row_state_to_token((a, b, c, d))] = idx
        idx += 1
    return vocab


def invert_vocab(vocab: dict[str, int] | None = None) -> dict[int, str]:
    vocab = vocab or build_vocab()
    return {idx: tok for tok, idx in vocab.items()}


def event_row_token_ids(vocab: dict[str, int]) -> set[int]:
    """ROW token ids the model may emit (at least one TAP / HOLD_START / HOLD_END)."""
    ids: set[int] = set()
    for a, b, c, d in product(LaneState, repeat=4):
        row = (a, b, c, d)
        if is_event_row(row):
            ids.add(vocab[row_state_to_token(row)])
    return ids
