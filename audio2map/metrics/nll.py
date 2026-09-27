"""Masked token NLL (shared by train and teacher-forcing eval)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from audio2map.metrics.token import TOKEN_TYPE_BUCKETS, loss_token_bucket


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


def token_nll_by_type(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    id_to_token: dict[int, str],
) -> dict[str, tuple[float, int]]:
    """Per-type ``(loss_sum, count)`` using :func:`loss_token_bucket`.

    Ordinary token CE averaged later by the caller. Does not change
    :func:`token_nll_stats`, which training still calls.
    """
    totals = {name: 0.0 for name in TOKEN_TYPE_BUCKETS}
    counts = {name: 0 for name in TOKEN_TYPE_BUCKETS}
    b, length, vocab = logits.shape
    flat_logits = logits.reshape(b * length, vocab)
    flat_targets = targets.reshape(b * length)
    flat_mask = loss_mask.reshape(b * length)
    with torch.no_grad():
        ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        for loss, tid, m in zip(ce.tolist(), flat_targets.tolist(), flat_mask.tolist(), strict=True):
            if not m:
                continue
            bucket = loss_token_bucket(id_to_token[int(tid)])
            if bucket is None:
                continue
            totals[bucket] += float(loss)
            counts[bucket] += 1
    return {name: (totals[name], counts[name]) for name in TOKEN_TYPE_BUCKETS}
