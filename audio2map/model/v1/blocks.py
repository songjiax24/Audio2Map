"""Encoder and decoder blocks. Condition enters only through decoder AdaLN."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from audio2map.model.v1.constants import D_MODEL, FFN_HIDDEN, HEAD_DIM, N_HEADS, ROPE_BASE
from audio2map.model.v1.rope import Rotary


class SwiGLU(nn.Module):
    """Hidden width is the gated dimension. Biases stay off."""

    def __init__(self, d_model: int = D_MODEL, hidden: int = FFN_HIDDEN) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class AdaLN(nn.Module):
    """LayerNorm without its own scale, then γ, β, α from c. The projection starts at 0."""

    def __init__(self, d_model: int = D_MODEL, cond_dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 3 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta, alpha = self.proj(cond).chunk(3, dim=-1)
        hidden = self.norm(x) * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        return hidden, alpha[:, None, :]


def _sdpa_mask(allow: torch.Tensor) -> torch.Tensor:
    """Boolean SDPA mask. True keeps the key. A query with no key is zeroed later."""
    return allow.unsqueeze(1)


def _fill_empty(allow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a mask SDPA can softmax, and which queries had no real key."""
    empty = ~allow.any(dim=-1)
    if not bool(empty.any()):
        return allow, empty
    return allow.masked_fill(empty.unsqueeze(-1), True), empty


def _zero_empty(out: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
    if not bool(empty.any()):
        return out
    return out.masked_fill(empty[:, None, :, None], 0.0)


class SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL)
        self.rotary = Rotary(HEAD_DIM, base=ROPE_BASE)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        allow: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.rotary.apply(q, positions)
        k = self.rotary.apply(k, positions)
        usable, empty = _fill_empty(allow)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=_sdpa_mask(usable))
        out = _zero_empty(out, empty)
        out = out.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL)
        self.rotary = Rotary(HEAD_DIM, base=ROPE_BASE)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        allow: torch.Tensor,
    ) -> torch.Tensor:
        batch, q_len, _ = query.shape
        s_len = memory.shape[1]
        q = self.q_proj(query).view(batch, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(batch, s_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(batch, s_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.rotary.apply(q, q_pos)
        k = self.rotary.apply(k, k_pos)
        usable, empty = _fill_empty(allow)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=_sdpa_mask(usable))
        out = _zero_empty(out, empty)
        out = out.transpose(1, 2).contiguous().view(batch, q_len, D_MODEL)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    """Pre-norm local self-attention, then SwiGLU. No condition and no φ/ψ."""

    def __init__(self) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(D_MODEL)
        self.attn = SelfAttention()
        self.norm_ff = nn.LayerNorm(D_MODEL)
        self.ff = SwiGLU()

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        allow: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), positions, allow)
        return x + self.ff(self.norm_ff(x))


class DecoderLayer(nn.Module):
    """AdaLN-Zero on self-attention, cross-attention, and the FFN."""

    def __init__(self, cond_dim: int) -> None:
        super().__init__()
        self.sa_mod = AdaLN(cond_dim=cond_dim)
        self.self_attn = SelfAttention()
        self.ca_mod = AdaLN(cond_dim=cond_dim)
        self.cross_attn = CrossAttention()
        self.ff_mod = AdaLN(cond_dim=cond_dim)
        self.ff = SwiGLU()

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        positions: torch.Tensor,
        self_allow: torch.Tensor,
        memory: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        cross_ok: torch.Tensor,
    ) -> torch.Tensor:
        hidden, alpha = self.sa_mod(x, cond)
        x = x + alpha * self.self_attn(hidden, positions, self_allow)
        hidden, alpha = self.ca_mod(x, cond)
        x = x + alpha * self.cross_attn(hidden, memory, q_pos, k_pos, cross_ok)
        hidden, alpha = self.ff_mod(x, cond)
        return x + alpha * self.ff(hidden)
