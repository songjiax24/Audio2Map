"""RoPE for integer g and for a fractional query position. Base is 10000."""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Rotary(nn.Module):
    def __init__(self, head_dim: int, *, base: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``x`` is ``(B, H, T, D)``. ``positions`` is ``(T,)`` or ``(B, T)``."""
        freqs = positions.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        if emb.ndim == 2:
            cos = emb.cos()[None, None, :, :]
            sin = emb.sin()[None, None, :, :]
        else:
            cos = emb.cos()[:, None, :, :]
            sin = emb.sin()[:, None, :, :]
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        return x * cos + rotate_half(x) * sin
