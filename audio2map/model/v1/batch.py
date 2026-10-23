"""Teacher-forcing batch. The dataset builds it and the chart model reads it.

A trainable example carries normalized mel. ``precomputed=True`` carries
detached stem rows and does not update ``AudioStem``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class V1Batch:
    """``cursor_time_ms`` is the state before each target (``token_ids[:, :-1]``)."""

    frame_lo: torch.Tensor
    cond_x: torch.Tensor
    cond_m: torch.Tensor
    token_ids: torch.Tensor
    cursor_time_ms: torch.Tensor
    phi: torch.Tensor
    beat_b: torch.Tensor
    audio_mask: torch.Tensor | None = None
    token_mask: torch.Tensor | None = None
    # (B, n_mels, 2S+3) on the trainable path. None when ``precomputed``.
    mel: torch.Tensor | None = None
    # Real stem-frame count per example, before batch padding.
    n_frames: torch.Tensor | None = None
    # (B, S, 512) only when ``precomputed`` is true.
    audio: torch.Tensor | None = None
    precomputed: bool = False
