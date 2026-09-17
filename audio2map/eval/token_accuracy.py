"""Split token accuracy by type (BAR / POS / ROW / EOS) and per-lane hold recall."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from audio2map.tokens import (
    TOKEN_BAR,
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_PAD,
    TOKEN_POS_PREFIX,
    TOKEN_ROW_PREFIX,
    LaneState,
    RowState,
    row_state_from_token,
)


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


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
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def to_dict(self) -> dict:
        return {
            "bar_acc": _ratio(self.bar_correct, self.bar_total),
            "pos_acc": _ratio(self.pos_correct, self.pos_total),
            "row_exact_acc": _ratio(self.row_correct, self.row_total),
            "row_lane_acc": _ratio(self.row_lane_correct, self.row_lane_total),
            "eos_acc": _ratio(self.eos_correct, self.eos_total),
            "hold_start_recall": _ratio(self.hold_start_recall_num, self.hold_start_recall_den),
            "hold_end_recall": _ratio(self.hold_end_recall_num, self.hold_end_recall_den),
            "other_acc": _ratio(self.other_correct, self.other_total),
        }


def _pred_row(tok: str) -> RowState | None:
    try:
        return row_state_from_token(tok)
    except ValueError:
        return None


def split_token_accuracy(
    pred_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    id_to_token: dict[int, str],
) -> SplitTokenAccuracy:
    """Per-type accuracy for one sequence (1D tensors after ``[:, 1:]``)."""
    out = SplitTokenAccuracy()
    for p, t, m in zip(pred_ids.tolist(), target_ids.tolist(), loss_mask.tolist(), strict=True):
        if not m:
            continue
        pt, tt = id_to_token[int(p)], id_to_token[int(t)]
        if tt in (TOKEN_BOS, TOKEN_PAD):
            continue
        if tt == TOKEN_BAR:
            out.bar_total += 1
            out.bar_correct += int(pt == tt)
        elif tt.startswith(TOKEN_POS_PREFIX):
            out.pos_total += 1
            out.pos_correct += int(pt == tt)
        elif tt.startswith(TOKEN_ROW_PREFIX):
            trow = row_state_from_token(tt)
            prow = _pred_row(pt)
            out.row_total += 1
            out.row_correct += int(pt == tt)
            out.row_lane_total += 4
            for i, tlane in enumerate(trow):
                plane = prow[i] if prow is not None else None
                if plane == tlane:
                    out.row_lane_correct += 1
                if tlane == LaneState.HOLD_START:
                    out.hold_start_recall_den += 1
                    out.hold_start_recall_num += int(plane == LaneState.HOLD_START)
                elif tlane == LaneState.HOLD_END:
                    out.hold_end_recall_den += 1
                    out.hold_end_recall_num += int(plane == LaneState.HOLD_END)
        elif tt == TOKEN_EOS:
            out.eos_total += 1
            out.eos_correct += int(pt == tt)
        else:
            out.other_total += 1
            out.other_correct += int(pt == tt)
    return out
