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

from audio2map.model.v1.constants import CROSS_FUTURE, CROSS_PAST, D_MODEL, N_LAYERS
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


class CrossWindow:
    """K/V for ``[floor(c_t)-128, floor(c_t)+384)`` only. Not the whole song.

    ``g`` is ``frame_lo + arange(n_frames)`` and every frame is visible.
    The same bounds are reused until the cursor floor moves. A full span on
    CUDA is replayed from one graph. ``teacher_decode`` does not use this; it
    still projects the whole song.
    """

    def __init__(self, frame_lo: int, n_frames: int) -> None:
        if n_frames < 0:
            raise ValueError(f"n_frames={n_frames}")
        self.frame_lo = int(frame_lo)
        self.n_frames = int(n_frames)
        self.rel_lo = -1
        self.rel_hi = -1
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        # A full 512-frame span stays in this storage so a CUDA graph can replay it.
        self._full_keys: list[torch.Tensor] | None = None
        self._full_values: list[torch.Tensor] | None = None
        self._proj_graph: torch.cuda.CUDAGraph | None = None
        self._proj_context: torch.Tensor | None = None
        self._proj_base: torch.Tensor | None = None
        self._proj_iota: torch.Tensor | None = None

    def prepare(self, model: V1ChartModel, context: torch.Tensor, floor_c: int) -> None:
        """Project the cursor span. A repeated floor does not run the linears."""
        if context.shape[1] != self.n_frames:
            raise ValueError(f"context length {context.shape[1]} != {self.n_frames}")
        abs_lo = max(self.frame_lo, int(floor_c) - CROSS_PAST)
        abs_hi = min(self.frame_lo + self.n_frames, int(floor_c) + CROSS_FUTURE)
        rel_lo = abs_lo - self.frame_lo
        rel_hi = abs_hi - self.frame_lo
        if rel_lo == self.rel_lo and rel_hi == self.rel_hi and self.keys:
            return
        if rel_hi <= rel_lo:
            self.keys = []
            self.values = []
            self.rel_lo = rel_lo
            self.rel_hi = rel_hi
            return
        full = rel_hi - rel_lo == CROSS_PAST + CROSS_FUTURE
        if (
            full
            and context.is_cuda
            and self._proj_graph is not None
            and self._proj_context is context
        ):
            assert self._proj_base is not None
            assert self._full_keys is not None and self._full_values is not None
            self._proj_base.fill_(abs_lo)
            self._proj_graph.replay()
            self.keys = self._full_keys
            self.values = self._full_values
            self.rel_lo = rel_lo
            self.rel_hi = rel_hi
            return
        self._project_slice(model, context, abs_lo, abs_hi, rel_lo, rel_hi, full)
        if full and context.is_cuda:
            self._capture_proj(model, context, abs_lo)

    def _project_slice(
        self,
        model: V1ChartModel,
        context: torch.Tensor,
        abs_lo: int,
        abs_hi: int,
        rel_lo: int,
        rel_hi: int,
        full: bool,
    ) -> None:
        span = context[:, rel_lo:rel_hi, :]
        positions = torch.arange(abs_lo, abs_hi, device=context.device)
        full_keys = self._full_keys
        full_values = self._full_values
        pinned = full and full_keys is not None and full_values is not None
        into = (full_keys, full_values) if pinned else None
        keys, values = self._project_layers(model, span, positions, into)
        if pinned:
            self.keys = full_keys
            self.values = full_values
        else:
            self.keys = keys
            self.values = values
            if full:
                self._full_keys = keys
                self._full_values = values
        self.rel_lo = rel_lo
        self.rel_hi = rel_hi

    def _capture_proj(self, model: V1ChartModel, context: torch.Tensor, abs_lo: int) -> None:
        """Record the full-span projection. Later floors only refill ``abs_lo``."""
        span = CROSS_PAST + CROSS_FUTURE
        if self._proj_iota is None or self._proj_iota.device != context.device:
            self._proj_iota = torch.arange(span, device=context.device)
            self._proj_base = torch.zeros((), dtype=torch.long, device=context.device)
        assert self._proj_base is not None
        self._proj_base.fill_(abs_lo)
        self._proj_context = context
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._project_full(model, context)
        torch.cuda.current_stream().wait_stream(stream)
        self._proj_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._proj_graph):
            self._project_full(model, context)

    def _project_full(self, model: V1ChartModel, context: torch.Tensor) -> None:
        """Gather ``[abs_lo, abs_lo+512)`` and write the pinned K/V. Graph-safe."""
        assert self._proj_base is not None and self._proj_iota is not None
        assert self._full_keys is not None and self._full_values is not None
        index = self._proj_iota + (self._proj_base - self.frame_lo)
        span = context.index_select(1, index)
        positions = self._proj_base + self._proj_iota
        self._project_layers(model, span, positions, (self._full_keys, self._full_values))

    def _project_layers(
        self,
        model: V1ChartModel,
        span: torch.Tensor,
        positions: torch.Tensor,
        into: tuple[list[torch.Tensor], list[torch.Tensor]] | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        width = span.shape[1]
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for index, layer in enumerate(model.decoder):
            attn = layer.cross_attn
            flat_k = attn.k_proj(span).view(1, width, attn.n_heads, attn.head_dim)
            flat_v = attn.v_proj(span).view(1, width, attn.n_heads, attn.head_dim)
            key = attn.rotary.apply(flat_k.transpose(1, 2), positions)
            value = flat_v.transpose(1, 2).contiguous()
            if into is None:
                keys.append(key)
                values.append(value)
            else:
                into[0][index].copy_(key)
                into[1][index].copy_(value)
        return keys, values

    def attend(
        self,
        attn: torch.nn.Module,
        query: torch.Tensor,
        c_t: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """One query, the prepared span, no index rebuild and no backend read."""
        if not self.keys or self.rel_hi <= self.rel_lo:
            zeros = query.new_zeros(query.shape[0], query.shape[1], D_MODEL)
            return attn.out_proj(zeros)
        key = self.keys[layer]
        value = self.values[layer]
        batch, q_len, _ = query.shape
        q = attn.q_proj(query).view(batch, q_len, attn.n_heads, attn.head_dim).transpose(1, 2)
        q = attn.rotary.apply(q, c_t)
        mixed = F.scaled_dot_product_attention(q, key, value)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, q_len, D_MODEL)
        return attn.out_proj(mixed)


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
    if batch.mel is None or batch.n_frames is None:
        raise ValueError("teacher_decode needs mel and n_frames")
    device = model.encoder.stem.conv1.weight.device
    tokens = batch.token_ids[:, :-1].to(device)
    steps = tokens.shape[1]
    if batch.audio_mask is None:
        audio_mask = torch.ones(
            batch.mel.shape[0],
            int(batch.n_frames.max()),
            dtype=torch.bool,
            device=device,
        )
    else:
        audio_mask = batch.audio_mask.to(device)
    memory = model.encoder(
        batch.mel,
        batch.frame_lo.to(device),
        audio_mask,
        n_frames=batch.n_frames,
    )
    context = memory + model.phase(
        phase_fourier(
            batch.phi.to(device=device, dtype=memory.dtype),
            batch.beat_b.to(device),
            batch.log_bpm.to(device=device),
            batch.rel_u.to(device=device),
        )
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
