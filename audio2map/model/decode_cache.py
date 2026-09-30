"""Inference cache: one encoder pass per audio window, then decoder KV steps.

Training ``model.forward`` is unchanged. This path is eval-only and reuses the
same weights. Dropout modules still run, so they are no-ops under ``eval``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from audio2map.model.model import AudioChartModel


@dataclass
class WindowDecodeState:
    """Encoder memory for one window, plus decoder keys and values."""

    cond_h: torch.Tensor
    cross_k: list[torch.Tensor]
    cross_v: list[torch.Tensor]
    self_k: list[torch.Tensor]
    self_v: list[torch.Tensor]
    length: int
    key_keep: torch.Tensor | None


def start_window_decode(
    model: AudioChartModel,
    audio: torch.Tensor,
    cond_vec: torch.Tensor,
    *,
    audio_mask: torch.Tensor | None = None,
) -> WindowDecodeState:
    """Encode ``audio`` once and project cross-attention keys and values."""
    if model.training:
        raise RuntimeError("window decode cache is inference only")
    memory, cond_h = model.encode_audio(audio, cond_vec, audio_mask=audio_mask)
    cross_k: list[torch.Tensor] = []
    cross_v: list[torch.Tensor] = []
    for layer in model.chart_decoder.layers:
        k, v = _cross_kv(layer.multihead_attn, memory)
        cross_k.append(k)
        cross_v.append(v)
    key_keep = None
    if audio_mask is not None and not bool(audio_mask.all()):
        key_keep = audio_mask
    return WindowDecodeState(
        cond_h=cond_h,
        cross_k=cross_k,
        cross_v=cross_v,
        self_k=[],
        self_v=[],
        length=0,
        key_keep=key_keep,
    )


def window_next_logits(
    model: AudioChartModel,
    state: WindowDecodeState,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Logits for the token after ``token_ids``.

    The first call prefills the whole prefix. Later calls must append exactly
    one id and only that position is decoded.
    """
    if model.training:
        raise RuntimeError("window decode cache is inference only")
    length = int(token_ids.shape[1])
    if length < 1:
        raise ValueError("token_ids is empty")
    if state.length == 0:
        hidden = _run_prefix(model, state, token_ids)
    elif length == state.length + 1:
        hidden = _run_step(model, state, token_ids[:, -1:])
    else:
        raise ValueError(
            f"decode cache has {state.length} tokens, got a sequence of {length}"
        )
    return model.lm_head(hidden)


def _run_prefix(
    model: AudioChartModel,
    state: WindowDecodeState,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    length = int(token_ids.shape[1])
    pos = torch.arange(length, device=token_ids.device).unsqueeze(0)
    x = _embed(model, token_ids, state.cond_h, pos)
    self_k: list[torch.Tensor] = []
    self_v: list[torch.Tensor] = []
    for index, layer in enumerate(model.chart_decoder.layers):
        normed = layer.norm1(x)
        q, k, v = _self_qkv(layer.self_attn, normed)
        self_k.append(k)
        self_v.append(v)
        sa = _attend(layer.self_attn, q, k, v, causal=True, key_keep=None)
        x = x + layer.dropout1(sa)
        cross_q = _query(layer.multihead_attn, layer.norm2(x))
        ca = _attend(
            layer.multihead_attn,
            cross_q,
            state.cross_k[index],
            state.cross_v[index],
            causal=False,
            key_keep=state.key_keep,
        )
        x = x + layer.dropout2(ca)
        x = x + layer._ff_block(layer.norm3(x))
    state.self_k = self_k
    state.self_v = self_v
    state.length = length
    return x[:, -1]

def _run_step(
    model: AudioChartModel,
    state: WindowDecodeState,
    token_id: torch.Tensor,
) -> torch.Tensor:
    pos = torch.tensor([[state.length]], device=token_id.device)
    x = _embed(model, token_id, state.cond_h, pos)
    for index, layer in enumerate(model.chart_decoder.layers):
        normed = layer.norm1(x)
        q, k, v = _self_qkv(layer.self_attn, normed)
        k = torch.cat((state.self_k[index], k), dim=2)
        v = torch.cat((state.self_v[index], v), dim=2)
        state.self_k[index] = k
        state.self_v[index] = v
        sa = _attend(layer.self_attn, q, k, v, causal=False, key_keep=None)
        x = x + layer.dropout1(sa)
        cross_q = _query(layer.multihead_attn, layer.norm2(x))
        ca = _attend(
            layer.multihead_attn,
            cross_q,
            state.cross_k[index],
            state.cross_v[index],
            causal=False,
            key_keep=state.key_keep,
        )
        x = x + layer.dropout2(ca)
        x = x + layer._ff_block(layer.norm3(x))
    state.length += 1
    return x[:, -1]


def _embed(
    model: AudioChartModel,
    token_ids: torch.Tensor,
    cond_h: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor:
    return model.token_emb(token_ids) + model.decoder_pos(pos) + cond_h


def _heads(value: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, length, embed = value.shape
    head_dim = embed // num_heads
    return value.view(batch, length, num_heads, head_dim).transpose(1, 2)


def _self_qkv(
    attn: nn.MultiheadAttention,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query, key, value = F._in_projection_packed(x, x, x, attn.in_proj_weight, attn.in_proj_bias)
    return _heads(query, attn.num_heads), _heads(key, attn.num_heads), _heads(value, attn.num_heads)


def _cross_kv(
    attn: nn.MultiheadAttention,
    memory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    embed = attn.embed_dim
    weight = attn.in_proj_weight[embed:]
    bias = None if attn.in_proj_bias is None else attn.in_proj_bias[embed:]
    key, value = F.linear(memory, weight, bias).chunk(2, dim=-1)
    return _heads(key, attn.num_heads), _heads(value, attn.num_heads)


def _query(attn: nn.MultiheadAttention, x: torch.Tensor) -> torch.Tensor:
    embed = attn.embed_dim
    weight = attn.in_proj_weight[:embed]
    bias = None if attn.in_proj_bias is None else attn.in_proj_bias[:embed]
    return _heads(F.linear(x, weight, bias), attn.num_heads)


def _attend(
    attn: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    key_keep: torch.Tensor | None,
) -> torch.Tensor:
    mask = None
    if key_keep is not None:
        if causal:
            raise ValueError("causal self-attention does not take an audio key mask")
        mask = key_keep[:, None, None, :]
    heads = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=causal,
    )
    batch, num_heads, length, head_dim = heads.shape
    merged = heads.transpose(1, 2).contiguous().view(batch, length, num_heads * head_dim)
    return attn.out_proj(merged)
