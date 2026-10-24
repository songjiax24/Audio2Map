"""Inference cross-attention: one K/V projection per layer, then the exact window.

Self-attention stays the decoder's Flash window and is recomputed on the prefix.
A contiguous window is a slice of the cached K/V. Gather is only for a window
that visibility has broken into a non-contiguous set. SDPA gets no bool mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import logging

import torch
import torch.nn.functional as F

from audio2map.model.v1.constants import D_MODEL, N_LAYERS
from audio2map.model.v1.cross import window_slots
from audio2map.model.v1.model import V1Batch, V1ChartModel, phase_fourier
from audio2map.model.v1.windows import cross_positions

logger = logging.getLogger(__name__)

_BACKEND = {0: "MATH", 1: "FLASH", 2: "EFFICIENT", 3: "CUDNN", 4: "OVERRIDEABLE"}


@dataclass
class DecodeResult:
    logits: torch.Tensor
    backend: str


def _backend_name(codes: list[int]) -> str:
    names: list[str] = []
    for code in codes:
        name = _BACKEND.get(code, str(code))
        if name not in names:
            names.append(name)
    return ",".join(names) if names else "EMPTY"


def project_cross_cache(
    model: V1ChartModel,
    context: torch.Tensor,
    g: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """``k`` already has RoPE at integer ``g``. ``v`` is unrotated."""
    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in model.decoder:
        attn = layer.cross_attn
        k = attn.k_proj(context).view(context.shape[0], -1, attn.n_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(context).view(context.shape[0], -1, attn.n_heads, attn.head_dim).transpose(1, 2)
        layers.append((attn.rotary.apply(k, g), v))
    return layers


def _read_window(cache: torch.Tensor, row: int, taken: torch.Tensor) -> torch.Tensor:
    """``cache`` is ``(B, H, S, D)``. ``taken`` is an ascending key index."""
    lo = int(taken[0])
    hi = int(taken[-1]) + 1
    if hi - lo == int(taken.shape[0]):
        return cache[row, :, lo:hi]
    pick = taken.view(1, -1, 1).expand(cache.shape[1], -1, cache.shape[-1])
    return torch.gather(cache[row], 1, pick)


def gather_cross(
    attn,
    query: torch.Tensor,
    c_t: torch.Tensor,
    g: torch.Tensor,
    audio_mask: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    """Exact per-query windows, then SDPA. No bool mask is passed."""
    batch, q_len, _ = query.shape
    index, valid = window_slots(c_t, g, audio_mask)
    counts = valid.sum(dim=-1).reshape(-1)
    dtype = query.dtype
    if query.is_cuda and torch.is_autocast_enabled():
        dtype = torch.get_autocast_dtype("cuda")
    out = query.new_zeros(batch * q_len, D_MODEL, dtype=dtype)
    codes: list[int] = []
    query_flat = query.reshape(batch * q_len, D_MODEL)
    c_flat = c_t.reshape(batch * q_len)
    index_flat = index.reshape(batch * q_len, index.shape[-1])
    valid_flat = valid.reshape(batch * q_len, valid.shape[-1])
    empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
    if empty.numel():
        zeros = query.new_zeros(empty.shape[0], D_MODEL, dtype=dtype)
        out = out.index_copy(0, empty, attn.out_proj(zeros))
    for width in torch.unique(counts).tolist():
        width = int(width)
        if width == 0:
            continue
        chosen = torch.nonzero(counts == width, as_tuple=False).flatten()
        rows = torch.div(chosen, q_len, rounding_mode="floor")
        keys = []
        values = []
        for slot, row in zip(chosen.tolist(), rows.tolist(), strict=True):
            taken = index_flat[slot][valid_flat[slot]]
            keys.append(_read_window(key, int(row), taken))
            values.append(_read_window(value, int(row), taken))
        count = chosen.shape[0]
        q = attn.q_proj(query_flat[chosen]).view(count, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
        q = attn.rotary.apply(q, c_flat[chosen].reshape(count, 1)).contiguous()
        k = keys[0].unsqueeze(0) if count == 1 else torch.stack(keys, 0)
        v = values[0].unsqueeze(0) if count == 1 else torch.stack(values, 0)
        k = k.contiguous()
        v = v.contiguous()
        codes.append(int(torch._fused_sdp_choice(q, k, v, None, 0.0, False)))
        mixed = F.scaled_dot_product_attention(q, k, v)
        mixed = attn.out_proj(mixed.transpose(1, 2).contiguous().view(count, D_MODEL))
        out = out.index_copy(0, chosen, mixed)
    return out.view(batch, q_len, D_MODEL), codes


def teacher_decode(model: V1ChartModel, batch: V1Batch) -> DecodeResult:
    """Gold-token steps. Cross-attention reads the preprojected cache."""
    if model.training:
        raise RuntimeError("teacher_decode is inference only")
    audio = model._audio_rows(batch)
    device = audio.device
    tokens = batch.token_ids[:, :-1].to(device)
    steps = tokens.shape[1]
    audio_mask = (
        torch.ones(audio.shape[:2], dtype=torch.bool, device=device)
        if batch.audio_mask is None
        else batch.audio_mask.to(device)
    )
    memory = model.encode(audio, batch.frame_lo.to(device), audio_mask)
    context = memory + model.phase(
        phase_fourier(batch.phi.to(device=device, dtype=memory.dtype), batch.beat_b.to(device))
    )
    g, p_q = cross_positions(batch.frame_lo, context.shape[1], batch.cursor_time_ms.to(device))
    cache = project_cross_cache(model, context, g)
    cond = model.cond(
        batch.cond_x.to(device=device, dtype=memory.dtype),
        batch.cond_m.to(device=device, dtype=memory.dtype),
    )
    seqs: list[torch.Tensor | None] = [None] * (N_LAYERS + 1)
    codes: list[int] = []
    for step in range(steps):
        token = tokens[:, step : step + 1]
        cursor = p_q[:, step : step + 1]
        seqs[0] = model.token_emb(token) if seqs[0] is None else torch.cat((seqs[0], model.token_emb(token)), dim=1)
        for index, layer in enumerate(model.decoder):
            hidden = seqs[index]
            assert hidden is not None
            length = hidden.shape[1]
            positions = torch.arange(length, device=device)
            keep = torch.ones(1, length, dtype=torch.bool, device=device).expand(hidden.shape[0], -1)
            modulated, alpha = layer.sa_mod(hidden, cond)
            hidden = hidden + alpha * layer.self_attn(modulated, positions, keep)
            last = hidden[:, -1:, :]
            modulated, alpha = layer.ca_mod(last, cond)
            mixed, seen = gather_cross(
                layer.cross_attn,
                modulated,
                cursor,
                g,
                audio_mask,
                cache[index][0],
                cache[index][1],
            )
            codes.extend(seen)
            last = last + alpha * mixed
            modulated, alpha = layer.ff_mod(last, cond)
            last = last + alpha * layer.ff(modulated)
            seqs[index + 1] = last if seqs[index + 1] is None else torch.cat((seqs[index + 1], last), dim=1)
    assert seqs[-1] is not None
    backend = _backend_name(codes)
    logger.info("cross backend=%s", backend)
    return DecodeResult(model.lm_head(seqs[-1]), backend)
