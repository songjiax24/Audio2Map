"""Masked CE loss and one teacher-forcing step (collate keys live here)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from audio2map.model.model import AudioChartModel


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    b, l, v = logits.shape
    logits = logits.reshape(b * l, v)
    targets = targets.reshape(b * l)
    mask = loss_mask.reshape(b * l)
    ce = F.cross_entropy(logits, targets, reduction="none")
    weighted = ce * mask
    if mask.sum() == 0:
        return weighted.sum()
    return weighted.sum() / mask.sum()


def training_step(
    model: AudioChartModel,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model(
        batch["audio"],
        batch["cond_vec"],
        batch["token_ids"],
        attn_mask=batch.get("attn_mask"),
        audio_mask=batch.get("audio_mask"),
    )
    targets = batch["token_ids"][:, 1:]
    mask = batch["loss_mask"][:, 1:]
    loss = compute_loss(logits, targets, mask)
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        acc = ((pred == targets) & mask.bool()).float().sum() / mask.sum().clamp(min=1)
    return loss, {
        "loss": float(loss.item()),
        "token_acc": float(acc.item()),
        "audio_ticks": float(batch["audio"].shape[1]),
    }
