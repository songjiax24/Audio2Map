"""Masked CE loss and one teacher-forcing step (collate keys live here)."""

from __future__ import annotations

import torch

from audio2map.metrics.nll import token_nll_stats
from audio2map.model.model import AudioChartModel

__all__ = ["token_nll_stats", "compute_loss", "scale_grads_by_valid_tokens", "training_step"]


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    loss_sum, valid, _ = token_nll_stats(logits, targets, loss_mask)
    if float(valid.item()) == 0.0:
        return loss_sum
    return loss_sum / valid


def scale_grads_by_valid_tokens(model: torch.nn.Module, n_valid: float) -> None:
    """Turn accumulated ``∂(Σℓ)/∂θ`` into the token-mean CE gradient."""
    if n_valid <= 0.0:
        return
    inv = 1.0 / n_valid
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(inv)


def training_step(
    model: AudioChartModel,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return masked CE **sum** (for backward) and detached token stats."""
    logits = model(
        batch["audio"],
        batch["cond_vec"],
        batch["token_ids"],
        attn_mask=batch.get("attn_mask"),
        audio_mask=batch.get("audio_mask"),
    )
    targets = batch["token_ids"][:, 1:]
    mask = batch["loss_mask"][:, 1:]
    loss_sum, valid, correct = token_nll_stats(logits, targets, mask)
    n_valid = float(valid.detach().item())
    token_acc = 0.0 if n_valid == 0.0 else float(correct.detach().item()) / n_valid
    return loss_sum, {
        "loss": float((loss_sum / valid).detach().item()) if n_valid else 0.0,
        "token_acc": token_acc,
        "loss_sum": float(loss_sum.detach().item()),
        "valid_tokens": n_valid,
        "correct_tokens": float(correct.detach().item()),
        "audio_ticks": float(batch["audio"].shape[1]),
    }
