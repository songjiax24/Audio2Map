"""Teacher-forcing batch. The dataset builds it and the chart model reads it.

A trainable example carries normalized mel. The encoder stems it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
    # (ln(BPM) - ln(180)) / 0.5, and u = (t - t0) / (t1 - t0).
    log_bpm: torch.Tensor
    rel_u: torch.Tensor
    # Size-1 examples keep these so collate can fill right-pad frames.
    # A collated batch has already written log_bpm and rel_u, and leaves
    # the redline rows empty.
    coverage_start_ms: torch.Tensor | None = None
    coverage_end_ms: torch.Tensor | None = None
    redline_offset_ms: torch.Tensor | None = None
    redline_beat_ms: torch.Tensor | None = None
    audio_mask: torch.Tensor | None = None
    token_mask: torch.Tensor | None = None
    # (B, n_mels, 2S+3). The encoder reads this.
    mel: torch.Tensor | None = None
    # Real stem-frame count per example, before batch padding.
    n_frames: torch.Tensor | None = None
    # One int16 array of legal ids per non-PAD target. Validation only.
    legal_ids: tuple[np.ndarray, ...] | None = None
