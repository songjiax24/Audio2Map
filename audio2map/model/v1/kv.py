"""Formal decoder step: a fixed self-attention ring, then the cross window.

Self-attention stores one token per slot of ``DEC_SPAN`` and reads
``0 <= q - k < 1024``. RoPE uses the absolute decoder position at write time.
Sampling cross-attention reads a cursor window of H_audio. Once that window
and the ring are both full, ``WindowStep`` replays the step as a CUDA graph.
``teacher_decode`` still projects the whole song and gathers. It does not use
this ring or the graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from audio2map.model.v1.constants import CROSS_FUTURE, CROSS_PAST, DEC_SPAN, D_MODEL
from audio2map.model.v1.decode import CrossWindow, gather_cross
from audio2map.model.v1.model import V1ChartModel


class SelfKV:
    """One rotated K and V per fed token, in a ring of ``DEC_SPAN`` slots.

    The ring does not grow with the song. A step reads the same suffix a
    growing buffer would have sliced. The copy exists only when the next write
    would overwrite a key this window still holds.
    """

    def __init__(self, n_layers: int) -> None:
        self.keys: list[torch.Tensor | None] = [None] * n_layers
        self.values: list[torch.Tensor | None] = [None] * n_layers
        self.length = 0

    def write(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store the current token at ``length % DEC_SPAN``. ``(1, H, 1, D)``."""
        if key.shape[2] != 1:
            raise ValueError(f"self-attention write expects one token, got {key.shape[2]}")
        store = self.keys[layer]
        if store is None:
            shape = (1, key.shape[1], DEC_SPAN, key.shape[-1])
            self.keys[layer] = key.new_empty(shape)
            self.values[layer] = key.new_empty(shape)
            store = self.keys[layer]
        assert store is not None and self.values[layer] is not None
        slot = self.length % DEC_SPAN
        store[:, :, slot : slot + 1] = key
        self.values[layer][:, :, slot : slot + 1] = value

    def commit(self) -> None:
        self.length += 1

    def window(self, layer: int, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys ``[position-1023, position]``, oldest first. RoPE stays as stored."""
        key = self.keys[layer]
        value = self.values[layer]
        if key is None or value is None or position != self.length or position < 0:
            raise IndexError(f"decoder position {position} is not the token just written")
        return _ring_window(key, position), _ring_window(value, position)


def _ring_window(store: torch.Tensor, position: int) -> torch.Tensor:
    """Time order. A view is safe until the next write lands inside it."""
    if position < DEC_SPAN - 1:
        return store[:, :, : position + 1]
    if position < DEC_SPAN:
        return store[:, :, : position + 1].clone()
    start = (position + 1) % DEC_SPAN
    return torch.cat((store[:, :, start:], store[:, :, :start]), dim=2)


def cached_logits(
    model: V1ChartModel,
    token: torch.Tensor,
    position: int,
    kv: SelfKV,
    cond: torch.Tensor,
    c_t: torch.Tensor,
    *,
    window: CrossWindow | None = None,
    cross: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    g: torch.Tensor | None = None,
    audio_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Logits for the token after ``token``. ``token`` is ``(1, 1)``.

    Pass ``window`` for sampling. Pass ``cross`` with ``g`` and ``audio_mask``
    for a full-song cache. Exactly one of those paths is used.
    """
    if model.training:
        raise RuntimeError("cached decode is inference only")
    if position != kv.length:
        raise RuntimeError(f"position {position} does not match the cache length {kv.length}")
    if (window is None) == (cross is None):
        raise ValueError("pass a cursor window or a full cross cache, not both")
    if cross is not None and (g is None or audio_mask is None):
        raise ValueError("a full cross cache needs g and audio_mask")
    x = model.token_emb(token)
    pos = torch.tensor([position], device=x.device)
    for index, layer in enumerate(model.decoder):
        hidden, alpha = layer.sa_mod(x, cond)
        attn = layer.self_attn
        query = attn.q_proj(hidden).view(1, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
        key = attn.k_proj(hidden).view(1, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
        value = attn.v_proj(hidden).view(1, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
        query = attn.rotary.apply(query, pos)
        key = attn.rotary.apply(key, pos)
        kv.write(index, key, value)
        key_w, value_w = kv.window(index, position)
        mixed = F.scaled_dot_product_attention(query, key_w, value_w)
        mixed = mixed.transpose(1, 2).contiguous().view(1, 1, D_MODEL)
        x = x + alpha * attn.out_proj(mixed)
        hidden, alpha = layer.ca_mod(x, cond)
        if window is not None:
            mixed = window.attend(layer.cross_attn, hidden, c_t, index)
        else:
            assert cross is not None and g is not None and audio_mask is not None
            mixed, _seen = gather_cross(
                layer.cross_attn,
                hidden,
                c_t,
                g,
                audio_mask,
                cross[index][0],
                cross[index][1],
            )
        x = x + alpha * mixed
        hidden, alpha = layer.ff_mod(x, cond)
        x = x + alpha * layer.ff(hidden)
    kv.commit()
    return model.lm_head(x)


class WindowStep:
    """CUDA graph of one sampling step after the self window is full.

    The ring slot and the time order are tensors, so a replay can move them.
    Cross K/V stay in the storage ``CrossWindow`` pins for a 512-frame span.
    Shorter spans, and the first ``DEC_SPAN`` tokens, stay on ``cached_logits``.
    """

    def __init__(
        self,
        model: V1ChartModel,
        kv: SelfKV,
        window: CrossWindow,
        cond: torch.Tensor,
    ) -> None:
        self.model = model
        self.kv = kv
        self.window = window
        self.cond = cond
        self.graph: torch.cuda.CUDAGraph | None = None
        self.out: torch.Tensor | None = None
        self._key_ptr: int | None = None
        self.token: torch.Tensor | None = None
        self.c_t: torch.Tensor | None = None
        self.pos: torch.Tensor | None = None
        self.slot: torch.Tensor | None = None
        self.order: torch.Tensor | None = None
        self._iota: torch.Tensor | None = None

    def covers(self, position: int) -> bool:
        """True when this step can replay, or can capture on its first call."""
        width = self.window.rel_hi - self.window.rel_lo
        if position < DEC_SPAN or width != CROSS_PAST + CROSS_FUTURE or not self.window.keys:
            return False
        if self._key_ptr is None:
            return True
        return self.window.keys[0].data_ptr() == self._key_ptr

    def run(self, token: torch.Tensor, position: int, c_t: torch.Tensor) -> torch.Tensor:
        """Logits for this token. Commits the ring, same as ``cached_logits``."""
        if position != self.kv.length:
            raise RuntimeError(
                f"position {position} does not match the cache length {self.kv.length}"
            )
        self._load(token, position, c_t)
        if self.graph is None:
            self._capture()
        assert self.graph is not None and self.out is not None
        self.graph.replay()
        self.kv.commit()
        return self.out

    def _load(self, token: torch.Tensor, position: int, c_t: torch.Tensor) -> None:
        slot = position % DEC_SPAN
        start = (position + 1) % DEC_SPAN
        if self.token is None or self.c_t is None or self.pos is None or self.slot is None:
            self.token = token.detach().clone()
            self.c_t = c_t.detach().clone()
            self.pos = torch.tensor([position], device=token.device)
            self.slot = torch.tensor([slot], device=token.device)
            self._iota = torch.arange(DEC_SPAN, device=token.device)
            self.order = torch.empty(DEC_SPAN, dtype=torch.long, device=token.device)
        else:
            self.token.copy_(token)
            self.c_t.copy_(c_t)
            self.pos.fill_(position)
            self.slot.fill_(slot)
        assert self.order is not None and self._iota is not None
        self.order.copy_(self._iota)
        self.order.add_(start)
        self.order.remainder_(DEC_SPAN)

    def _capture(self) -> None:
        saved_k = [key.clone() for key in self.kv.keys if key is not None]
        saved_v = [value.clone() for value in self.kv.values if value is not None]
        if len(saved_k) != len(self.kv.keys) or len(saved_v) != len(self.kv.values):
            raise RuntimeError("the self-attention ring is not full yet")

        def restore() -> None:
            for index, (key, value) in enumerate(zip(saved_k, saved_v, strict=True)):
                store_k = self.kv.keys[index]
                store_v = self.kv.values[index]
                assert store_k is not None and store_v is not None
                store_k.copy_(key)
                store_v.copy_(value)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(stream)
        restore()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._forward()
        restore()
        self._key_ptr = self.window.keys[0].data_ptr()

    def _forward(self) -> torch.Tensor:
        """Same math as the ``window`` path of ``cached_logits``, without commit."""
        assert self.token is not None and self.pos is not None and self.slot is not None
        assert self.order is not None and self.c_t is not None
        x = self.model.token_emb(self.token)
        for index, layer in enumerate(self.model.decoder):
            hidden, alpha = layer.sa_mod(x, self.cond)
            attn = layer.self_attn
            query = attn.q_proj(hidden).view(1, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
            key = attn.k_proj(hidden).view(1, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
            value = attn.v_proj(hidden).view(1, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
            query = attn.rotary.apply(query, self.pos)
            key = attn.rotary.apply(key, self.pos)
            store_k = self.kv.keys[index]
            store_v = self.kv.values[index]
            assert store_k is not None and store_v is not None
            store_k.index_copy_(2, self.slot, key)
            store_v.index_copy_(2, self.slot, value)
            key_w = store_k.index_select(2, self.order)
            value_w = store_v.index_select(2, self.order)
            mixed = F.scaled_dot_product_attention(query, key_w, value_w)
            mixed = mixed.transpose(1, 2).contiguous().view(1, 1, D_MODEL)
            x = x + alpha * attn.out_proj(mixed)
            hidden, alpha = layer.ca_mod(x, self.cond)
            mixed = self.window.attend(layer.cross_attn, hidden, self.c_t, index)
            x = x + alpha * mixed
            hidden, alpha = layer.ff_mod(x, self.cond)
            x = x + alpha * layer.ff(hidden)
        return self.model.lm_head(x)
