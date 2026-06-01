"""Split token accuracy by token type (BAR / POS / ROW / EOS)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from audio2map.osu.row_tokens import (
    TOKEN_BAR,
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_PAD,
    TOKEN_POS_PREFIX,
    TOKEN_ROW_PREFIX,
    invert_vocab,
)


@dataclass(slots=True)
class SplitTokenAccuracy:
    bar_correct: int = 0
    bar_total: int = 0
    pos_correct: int = 0
    pos_total: int = 0
    row_correct: int = 0
    row_total: int = 0
    row_lane_correct: int = 0
    row_lane_total: int = 0
    eos_correct: int = 0
    eos_total: int = 0
    hold_start_recall_num: int = 0
    hold_start_recall_den: int = 0
    hold_end_recall_num: int = 0
    hold_end_recall_den: int = 0
    other_correct: int = 0
    other_total: int = 0

    def merge(self, other: SplitTokenAccuracy) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self) -> dict:
        def rate(c: int, t: int) -> float | None:
            return c / t if t else None

        return {
            "bar_acc": rate(self.bar_correct, self.bar_total),
            "pos_acc": rate(self.pos_correct, self.pos_total),
            "row_exact_acc": rate(self.row_correct, self.row_total),
            "row_lane_acc": rate(self.row_lane_correct, self.row_lane_total),
            "eos_acc": rate(self.eos_correct, self.eos_total),
            "hold_start_recall": rate(self.hold_start_recall_num, self.hold_start_recall_den),
            "hold_end_recall": rate(self.hold_end_recall_num, self.hold_end_recall_den),
            "overall_acc": rate(
                self.bar_correct
                + self.pos_correct
                + self.row_correct
                + self.eos_correct
                + self.other_correct,
                self.bar_total
                + self.pos_total
                + self.row_total
                + self.eos_total
                + self.other_total,
            ),
        }


def _row_lane_match(pred_tok: str, tgt_tok: str) -> bool:
    if not (pred_tok.startswith(TOKEN_ROW_PREFIX) and tgt_tok.startswith(TOKEN_ROW_PREFIX)):
        return False
    return pred_tok == tgt_tok


def _row_has_state(tok: str, digit: str) -> bool:
    if not tok.startswith(TOKEN_ROW_PREFIX):
        return False
    body = tok[len(TOKEN_ROW_PREFIX) : -1]
    return digit in body


def accumulate_split_accuracy(
    pred_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    id_to_token: dict[int, str] | None = None,
) -> SplitTokenAccuracy:
    """Accumulate split accuracy for one batch row (1D tensors after [:,1:])."""
    id_to_token = id_to_token or invert_vocab()
    out = SplitTokenAccuracy()
    for p, t, m in zip(pred_ids.tolist(), target_ids.tolist(), loss_mask.tolist(), strict=True):
        if not m:
            continue
        pt, tt = id_to_token[int(p)], id_to_token[int(t)]
        if tt == TOKEN_BAR:
            out.bar_total += 1
            out.bar_correct += int(pt == tt)
        elif tt.startswith(TOKEN_POS_PREFIX):
            out.pos_total += 1
            out.pos_correct += int(pt == tt)
        elif tt.startswith(TOKEN_ROW_PREFIX):
            out.row_total += 1
            out.row_correct += int(pt == tt)
            out.row_lane_total += 4
            if _row_lane_match(pt, tt):
                out.row_lane_correct += 4
            else:
                for i in range(4):
                    if pt[i + len(TOKEN_ROW_PREFIX)] == tt[i + len(TOKEN_ROW_PREFIX)]:
                        out.row_lane_correct += 1
            if _row_has_state(tt, "2"):
                out.hold_start_recall_den += 1
                if _row_has_state(pt, "2"):
                    out.hold_start_recall_num += 1
            if _row_has_state(tt, "4"):
                out.hold_end_recall_den += 1
                if _row_has_state(pt, "4"):
                    out.hold_end_recall_num += 1
        elif tt == TOKEN_EOS:
            out.eos_total += 1
            out.eos_correct += int(pt == tt)
        elif tt not in (TOKEN_BOS, TOKEN_PAD):
            out.other_total += 1
            out.other_correct += int(pt == tt)
    return out
