"""v1 token vocabulary: 4 specials + 80 POS + 609 ROW = 693.

Id order is fixed by ``TOKENIZER_VERSION``. POS cells follow ``v80_cells``
(denominator, then numerator). ROW tokens are the event rows of ``LaneState``
in product order, which is lexicographic on the four lane digits.
"""

from __future__ import annotations

from itertools import product

from audio2map.grid.v1.positions import v80_cells
from audio2map.tokens.vocab import LaneState, RowState, is_event_row, row_state_to_token

TOKENIZER_VERSION = "v1-1"

TOKEN_PAD = "<PAD>"
TOKEN_BOS = "<BOS>"
TOKEN_EOS = "<EOS>"
TOKEN_BEAT = "<BEAT>"


def pos_token(numer: int, denom: int) -> str:
    return f"<POS_{numer}/{denom}>"


def parse_pos_token(token: str) -> tuple[int, int]:
    if not token.startswith("<POS_") or not token.endswith(">"):
        raise ValueError(f"invalid POS token: {token!r}")
    body = token[len("<POS_") : -1]
    numer_s, denom_s = body.split("/", 1)
    return int(numer_s), int(denom_s)


_EVENT_ROWS: tuple[RowState, ...] | None = None
_VOCAB: dict[str, int] | None = None
_INVERT: dict[int, str] | None = None


def event_rows() -> tuple[RowState, ...]:
    """Event rows. The scan runs once per process."""
    global _EVENT_ROWS
    if _EVENT_ROWS is None:
        rows = []
        for states in product(LaneState, repeat=4):
            row = (
                LaneState(states[0]),
                LaneState(states[1]),
                LaneState(states[2]),
                LaneState(states[3]),
            )
            if is_event_row(row):
                rows.append(row)
        if len(rows) != 609:
            raise RuntimeError(f"expected 609 event rows, got {len(rows)}")
        _EVENT_ROWS = tuple(rows)
    return _EVENT_ROWS


def build_vocab() -> dict[str, int]:
    """Token ids. The table is built once per process. Do not mutate it."""
    global _VOCAB, _INVERT
    if _VOCAB is None:
        vocab = {
            TOKEN_PAD: 0,
            TOKEN_BOS: 1,
            TOKEN_EOS: 2,
            TOKEN_BEAT: 3,
        }
        index = 4
        for numer, denom in v80_cells():
            vocab[pos_token(numer, denom)] = index
            index += 1
        for row in event_rows():
            vocab[row_state_to_token(row)] = index
            index += 1
        if len(vocab) != 693:
            raise RuntimeError(f"v1 vocab has {len(vocab)} tokens")
        _VOCAB = vocab
        _INVERT = {token_id: token for token, token_id in vocab.items()}
    return _VOCAB


def invert_vocab(vocab: dict[str, int] | None = None) -> dict[int, str]:
    if vocab is None or vocab is _VOCAB:
        build_vocab()
        assert _INVERT is not None
        return _INVERT
    return {index: token for token, index in vocab.items()}
