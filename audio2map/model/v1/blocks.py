"""Encoder and decoder blocks. Condition enters only through decoder AdaLN."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from audio2map.model.v1.constants import D_MODEL, FFN_HIDDEN, HEAD_DIM, N_HEADS, ROPE_BASE
from audio2map.model.v1.cross import CA_TRAIN_FLEX, FlexPlan, flex_cross
from audio2map.model.v1.rope import Rotary
from audio2map.model.v1.windows import DECODER_WINDOW, ENCODER_WINDOW


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


def _prefix_lengths(key_keep: torch.Tensor) -> torch.Tensor:
    """Valid-prefix lengths. A hole is not a sliding window."""
    if key_keep.ndim != 2 or key_keep.dtype != torch.bool:
        raise ValueError("self-attention visibility must be a bool prefix of shape (B, T)")
    lengths = key_keep.sum(dim=-1)
    index = torch.arange(key_keep.shape[-1], device=key_keep.device)
    if not torch.equal(key_keep, index < lengths[:, None]):
        raise ValueError("self-attention visibility must be a valid prefix, not a dense mask")
    return lengths.to(torch.int32)


def _flash_window(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
    window: tuple[int, int],
) -> torch.Tensor:
    """Flash sliding window via ``varlen_attn``. No attention mask is passed."""
    from torch.nn.attention.varlen import varlen_attn

    batch, heads, length, dim = q.shape
    if int(lengths.min()) == length:
        packed_q = q.transpose(1, 2).contiguous().view(batch * length, heads, dim)
        packed_k = k.transpose(1, 2).contiguous().view(batch * length, heads, dim)
        packed_v = v.transpose(1, 2).contiguous().view(batch * length, heads, dim)
        steps = torch.arange(0, (batch + 1) * length, length, device=q.device, dtype=torch.int32)
        out = varlen_attn(
            packed_q, packed_k, packed_v, steps, steps, length, length, window_size=window
        )
        return out.view(batch, length, heads, dim).transpose(1, 2).contiguous()
    if int(lengths.max()) == 0:
        return q.new_zeros(batch, heads, length, dim)
    visible = torch.arange(length, device=q.device)[None, :] < lengths[:, None]
    packed_q = q.permute(0, 2, 1, 3).contiguous()[visible]
    packed_k = k.permute(0, 2, 1, 3).contiguous()[visible]
    packed_v = v.permute(0, 2, 1, 3).contiguous()[visible]
    steps = torch.zeros(batch + 1, device=q.device, dtype=torch.int32)
    steps[1:] = torch.cumsum(lengths, 0, dtype=torch.int32)
    max_len = int(lengths.max())
    out = varlen_attn(
        packed_q, packed_k, packed_v, steps, steps, max_len, max_len, window_size=window
    )
    full = q.new_zeros(batch, length, heads, dim)
    full[visible] = out
    return full.permute(0, 2, 1, 3).contiguous()


def _reference_window(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
    window: tuple[int, int],
) -> torch.Tensor:
    """Same band on CPU / fp32. The mask covers the window width, not the sequence."""
    batch, heads, length, dim = q.shape
    left, right = window
    # The band is 1024 keys, but a shorter sequence has no keys outside it.
    span = min(left + right + 1, length)
    query_index = torch.arange(length, device=q.device)
    start = (query_index - left).clamp(min=0, max=max(length - span, 0))
    index = start[:, None] + torch.arange(span, device=q.device)[None, :]
    low = (query_index - left)[:, None]
    high = (query_index + right)[:, None]
    in_band = (index >= low) & (index <= high)
    query_ok = query_index[None, :] < lengths[:, None]
    valid = in_band[None] & (index[None] < lengths[:, None, None]) & query_ok[:, :, None]
    empty = ~valid.any(dim=-1)
    usable = valid.masked_fill(empty.unsqueeze(-1), True)
    safe = index.clamp(min=0, max=max(length - 1, 0))
    gather_index = safe.view(1, 1, length, span, 1).expand(batch, heads, length, span, dim)
    gathered = k.unsqueeze(3).expand(batch, heads, length, span, dim)
    k_win = torch.gather(gathered, 2, gather_index)
    v_win = torch.gather(v.unsqueeze(3).expand(batch, heads, length, span, dim), 2, gather_index)
    flat = batch * length
    out = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3).reshape(flat, heads, 1, dim),
        k_win.permute(0, 2, 1, 3, 4).reshape(flat, heads, span, dim),
        v_win.permute(0, 2, 1, 3, 4).reshape(flat, heads, span, dim),
        attn_mask=usable.reshape(flat, 1, 1, span),
    )
    out = out.view(batch, length, heads, dim).permute(0, 2, 1, 3)
    return out.masked_fill(~query_ok[:, None, :, None], 0.0)


class SelfAttention(nn.Module):
    """Fixed-band self-attention. CUDA fp16/bf16 uses Flash sliding window."""

    def __init__(self, window: tuple[int, int]) -> None:
        super().__init__()
        if window[0] < 0 or window[1] < 0:
            raise ValueError(f"window must be inclusive (left, right), got {window}")
        self.window = window
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
        key_keep: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        lengths = _prefix_lengths(key_keep)
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.rotary.apply(q, positions)
        k = self.rotary.apply(k, positions)
        flash = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
        out = (
            _flash_window(q, k, v, lengths, self.window)
            if flash
            else _reference_window(q, k, v, lengths, self.window)
        )
        out = out.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Training default is Flex. ``train_mode="legacy"`` keeps the dense bool mask."""

    def __init__(self) -> None:
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL)
        self.rotary = Rotary(HEAD_DIM, base=ROPE_BASE)
        self.train_mode = CA_TRAIN_FLEX

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        allow: torch.Tensor | FlexPlan,
    ) -> torch.Tensor:
        if isinstance(allow, FlexPlan):
            if self.train_mode != CA_TRAIN_FLEX:
                raise TypeError("FlexPlan was passed while ca_train is not flex")
            return flex_cross(self, query, memory, q_pos, k_pos, allow)
        if not isinstance(allow, torch.Tensor):
            raise TypeError("cross-attention expected a FlexPlan or a bool mask")
        return self._dense(query, memory, q_pos, k_pos, allow)

    def _dense(
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
        self.attn = SelfAttention(ENCODER_WINDOW)
        self.norm_ff = nn.LayerNorm(D_MODEL)
        self.ff = SwiGLU()

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        key_keep: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), positions, key_keep)
        return x + self.ff(self.norm_ff(x))


class DecoderLayer(nn.Module):
    """AdaLN-Zero on self-attention, cross-attention, and the FFN."""

    def __init__(self, cond_dim: int) -> None:
        super().__init__()
        self.sa_mod = AdaLN(cond_dim=cond_dim)
        self.self_attn = SelfAttention(DECODER_WINDOW)
        self.ca_mod = AdaLN(cond_dim=cond_dim)
        self.cross_attn = CrossAttention()
        self.ff_mod = AdaLN(cond_dim=cond_dim)
        self.ff = SwiGLU()

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        positions: torch.Tensor,
        key_keep: torch.Tensor,
        memory: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        cross_ok: torch.Tensor | FlexPlan,
    ) -> torch.Tensor:
        hidden, alpha = self.sa_mod(x, cond)
        x = x + alpha * self.self_attn(hidden, positions, key_keep)
        hidden, alpha = self.ca_mod(x, cond)
        x = x + alpha * self.cross_attn(hidden, memory, q_pos, k_pos, cross_ok)
        hidden, alpha = self.ff_mod(x, cond)
        return x + alpha * self.ff(hidden)
