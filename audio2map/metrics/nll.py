"""Masked token NLL (shared by train and teacher-forcing eval)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_nll_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(loss_sum, valid_tokens, correct_tokens)`` over the loss mask.

    ``loss_sum`` stays in the autograd graph. ``correct_tokens`` is detached.
    """
    b, l, v = logits.shape
    flat_logits = logits.reshape(b * l, v)
    flat_targets = targets.reshape(b * l)
    mask = loss_mask.reshape(b * l)
    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    loss_sum = (ce * mask).sum()
    valid = mask.sum()
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        correct = ((pred == targets) & loss_mask.bool()).sum().to(dtype=valid.dtype)
    return loss_sum, valid, correct
