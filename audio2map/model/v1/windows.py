"""Attention windows from §6.2 and §6.5. True means the key is kept."""

from __future__ import annotations

import torch

from audio2map.features.audio.v1.constants import DT_MS
from audio2map.model.v1.constants import CROSS_FUTURE, CROSS_PAST, DEC_SPAN, ENC_RADIUS

_LOW_PRECISION = (torch.float16, torch.bfloat16)


def encoder_keeps(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """True when ``key`` lies in ``[query - 512, query + 512)``."""
    return (key >= query - ENC_RADIUS) & (key < query + ENC_RADIUS)


def decoder_keeps(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """True when ``0 <= query - key < 1024``."""
    gap = query - key
    return (gap >= 0) & (gap < DEC_SPAN)


def inclusive_window(keeps) -> tuple[int, int]:
    """Flash ``window_size`` as inclusive ``(left, right)``, from ``keeps`` only."""
    origin = 8192
    query = torch.tensor([origin])
    offsets = torch.arange(-4096, 4096)
    chosen = offsets[keeps(query, query + offsets).squeeze(0)]
    if chosen.numel() == 0 or int(chosen.max() - chosen.min() + 1) != int(chosen.numel()):
        raise RuntimeError("self-attention window is not one contiguous band")
    return int(-chosen.min()), int(chosen.max())


# Inclusive on both sides. ``encoder_keeps`` is the half-open band of 1024 keys.
ENCODER_WINDOW = inclusive_window(encoder_keeps)
DECODER_WINDOW = inclusive_window(decoder_keeps)


def encoder_allow(length: int, device: torch.device) -> torch.Tensor:
    """Local bidirectional window on the cropped index e, shape ``(S, S)``."""
    query = torch.arange(length, device=device)[:, None]
    key = torch.arange(length, device=device)[None, :]
    return encoder_keeps(query, key)


def decoder_allow(length: int, device: torch.device) -> torch.Tensor:
    """Causal window ``0 <= q - k < 1024``, shape ``(L, L)``."""
    query = torch.arange(length, device=device)[:, None]
    key = torch.arange(length, device=device)[None, :]
    return decoder_keeps(query, key)


def reject_low_precision(name: str, value: torch.Tensor) -> None:
    """Frame indexes must stay int or fp32. bf16 cannot represent them past 256."""
    if value.dtype in _LOW_PRECISION:
        raise TypeError(f"{name} must be an integer or fp32 frame index, got {value.dtype}")


def cross_positions(
    frame_lo: torch.Tensor,
    n_frames: int,
    cursor_time_ms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer key index ``g`` and fp32 query position ``c_t``.

    ``c_t = cursor_time_ms / DT_MS``. Window edges use ``floor(c_t)``.
    """
    device = cursor_time_ms.device
    g = frame_lo.to(device=device, dtype=torch.long).unsqueeze(1)
    g = g + torch.arange(n_frames, device=device)
    c_t = cursor_time_ms.to(device=device, dtype=torch.float32) / DT_MS
    return g, c_t


def _as_long_index(name: str, value: torch.Tensor) -> torch.Tensor:
    reject_low_precision(name, value)
    if value.dtype == torch.long:
        return value
    return value.to(dtype=torch.long)


def _floor_index(name: str, value: torch.Tensor) -> torch.Tensor:
    reject_low_precision(name, value)
    if value.dtype == torch.long:
        return value
    return torch.floor(value.to(dtype=torch.float32)).to(dtype=torch.long)


def cross_allow(g: torch.Tensor, p_q: torch.Tensor) -> torch.Tensor:
    """``k ∈ [floor(p_q) - 128, floor(p_q) + 384)``. ``g`` is ``(B, S)``, ``p_q`` is ``(B, Q)``."""
    keys = _as_long_index("g", g)
    floor_c = _floor_index("p_q", p_q)
    left = floor_c - CROSS_PAST
    right = floor_c + CROSS_FUTURE
    return (keys[:, None, :] >= left[:, :, None]) & (keys[:, None, :] < right[:, :, None])
