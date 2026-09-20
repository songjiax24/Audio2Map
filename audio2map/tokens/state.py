"""Window chart-token grammar: legal next tokens and state transitions.

This is the shared prefix machine used by encode/decode checks, constrained
generation, and logit-level validity. It does not sample or score tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from audio2map.grid import TICKS_PER_BAR
from audio2map.tokens.vocab import (
    TOKEN_BAR,
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_POS_PREFIX,
    TOKEN_ROW_PREFIX,
    RowState,
    active_hold_from_initial_row,
    apply_row,
    build_vocab,
    event_row_token_ids,
    invert_vocab,
    is_legal_row,
    pos_from_token,
    pos_to_token,
    row_state_from_token,
)


@dataclass(slots=True)
class ChartState:
    """Track grammar state while walking a framed window sequence.

    Window ``<EOS>`` is allowed while holds are still active; a window is not a
    full chart, so open holds at the boundary are normal.
    """

    vocab: dict[str, int] = field(default_factory=build_vocab)
    id_to_token: dict[int, str] = field(default_factory=dict)
    window_bars: int = 0
    bars_done: int = 0
    in_bar: bool = False
    last_pos: int | None = None
    expect_row: bool = False
    active_hold: list[bool] = field(default_factory=lambda: [False, False, False, False])
    finished: bool = False
    _event_row_ids: set[int] = field(default_factory=set)
    _pos_ids: list[int] = field(default_factory=list)
    _bar_id: int = 0
    _eos_id: int = 0

    def __post_init__(self) -> None:
        if not self.id_to_token:
            self.id_to_token = invert_vocab(self.vocab)
        self._event_row_ids = event_row_token_ids(self.vocab)
        self._pos_ids = [self.vocab[pos_to_token(p)] for p in range(TICKS_PER_BAR)]
        self._bar_id = self.vocab[TOKEN_BAR]
        self._eos_id = self.vocab[TOKEN_EOS]

    @classmethod
    def from_initial_row(
        cls,
        initial_row: RowState,
        *,
        window_bars: int,
        vocab: dict[str, int] | None = None,
    ) -> ChartState:
        state = cls(vocab=vocab or build_vocab(), window_bars=window_bars)
        state.active_hold = active_hold_from_initial_row(initial_row)
        return state

    def allowed_token_ids(self) -> set[int]:
        if self.finished:
            return set()

        allowed: set[int] = set()
        if not self.in_bar:
            if self.bars_done < self.window_bars:
                allowed.add(self._bar_id)
            else:
                allowed.add(self._eos_id)
            return allowed

        if self.expect_row:
            for tid in self._event_row_ids:
                row = row_state_from_token(self.id_to_token[tid])
                if is_legal_row(row, self.active_hold):
                    allowed.add(tid)
            return allowed

        for pos, tid in enumerate(self._pos_ids):
            if self.last_pos is None or pos > self.last_pos:
                allowed.add(tid)

        if self.bars_done + 1 < self.window_bars:
            allowed.add(self._bar_id)
        else:
            allowed.add(self._eos_id)
        return allowed

    def observe(self, token_id: int) -> None:
        tok = self.id_to_token[token_id]
        if tok == TOKEN_EOS:
            if self.in_bar:
                self.bars_done += 1
            self.finished = True
            return
        if tok == TOKEN_BAR:
            if self.in_bar:
                self.bars_done += 1
            self.in_bar = True
            self.last_pos = None
            self.expect_row = False
            return
        if tok.startswith(TOKEN_POS_PREFIX):
            self.last_pos = pos_from_token(tok)
            self.expect_row = True
            return
        if tok.startswith(TOKEN_ROW_PREFIX):
            row = row_state_from_token(tok)
            apply_row(row, self.active_hold)
            self.expect_row = False
            return
        raise ValueError(f"unexpected token during decode: {tok!r}")


def bos_initial_prefix(initial_row_id: int, *, vocab: dict[str, int] | None = None) -> list[int]:
    vocab = vocab or build_vocab()
    return [vocab[TOKEN_BOS], initial_row_id]
