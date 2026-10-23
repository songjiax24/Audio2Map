"""Attention windows from §6.2 and §6.5. True means the key is kept."""

from __future__ import annotations

import torch

from audio2map.model.v1.constants import CROSS_FUTURE, CROSS_PAST, DEC_SPAN, ENC_RADIUS


def encoder_allow(length: int, device: torch.device) -> torch.Tensor:
    """Local bidirectional window on the cropped index e, shape ``(S, S)``."""
    query = torch.arange(length, device=device)[:, None]
    key = torch.arange(length, device=device)[None, :]
    return (key >= query - ENC_RADIUS) & (key < query + ENC_RADIUS)


def decoder_allow(length: int, device: torch.device) -> torch.Tensor:
    """Causal window ``0 <= q - k < 1024``, shape ``(L, L)``."""
    query = torch.arange(length, device=device)[:, None]
    key = torch.arange(length, device=device)[None, :]
    gap = query - key
    return (gap >= 0) & (gap < DEC_SPAN)


def cross_allow(g: torch.Tensor, p_q: torch.Tensor) -> torch.Tensor:
    """``k ∈ [floor(p_q) - 128, floor(p_q) + 384)``. ``g`` is ``(B, S)``, ``p_q`` is ``(B, Q)``."""
    left = torch.floor(p_q) - CROSS_PAST
    right = torch.floor(p_q) + CROSS_FUTURE
    return (g[:, None, :] >= left[:, :, None]) & (g[:, None, :] < right[:, :, None])
