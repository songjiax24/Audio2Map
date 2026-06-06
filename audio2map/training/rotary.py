"""Rotary position embeddings (RoPE) for sequence self-attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to ``q``/``k`` with shape ``(B, H, T, D)``."""
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    """Precompute cos/sin tables up to ``max_seq_len``."""

    def __init__(self, dim: int, *, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        cos, sin = self._build_cache(max_seq_len)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _build_cache(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # (1, 1, T, D) broadcasts with (B, H, T, D)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            cos, sin = self._build_cache(seq_len)
            return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)
        return (
            self.cos_cached[:, :, :seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:, :, :seq_len].to(device=device, dtype=dtype),
        )


class RoPEMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(t, x.device, dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.embed_dim)
        return self.out_proj(out)


class RoPETransformerEncoderLayer(nn.Module):
    """Pre-norm encoder layer with RoPE self-attention."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.self_attn = RoPEMultiheadSelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        *,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = src
        x = x + self.dropout1(
            self.self_attn(self.norm1(x), key_padding_mask=src_key_padding_mask)
        )
        x = x + self.dropout2(self._ff_block(self.norm2(x)))
        return x

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class RoPETransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RoPETransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        src: torch.Tensor,
        *,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = src
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=src_key_padding_mask)
        return out
