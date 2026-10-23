"""Whisper-style audio stem. Mel comes from ``features.audio.v1``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from audio2map.features.audio.v1.constants import (
    MEL_NORM_MU,
    MEL_NORM_SIGMA,
    N_MELS,
    SAMPLE_RATE,
)
from audio2map.features.audio.v1.mel import mel_span

STEM_CHANNELS = 512


@dataclass(frozen=True, slots=True)
class StemBatch:
    """Stem rows for mel frames ``frame_lo .. frame_hi-1``."""

    features: np.ndarray
    frame_lo: int
    frame_hi: int


class AudioStem(nn.Module):
    """Conv k=3 s=1, then conv k=3 s=2. No LayerNorm and no absolute positions."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(N_MELS, STEM_CHANNELS, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(STEM_CHANNELS, STEM_CHANNELS, kernel_size=3, stride=2, padding=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """``mel`` is ``(batch, n_mels, frames)``. Returns ``(batch, S, channels)``."""
        hidden = nn.functional.gelu(self.conv1(mel))
        hidden = nn.functional.gelu(self.conv2(hidden))
        if hidden.shape[-1] < 3:
            return hidden.new_zeros(hidden.shape[0], 0, STEM_CHANNELS)
        return hidden[:, :, 1:-1].transpose(1, 2)

    def rows(self, mel: np.ndarray) -> np.ndarray:
        """Detached CPU rows for offline checks. Training does not call this."""
        if int(mel.shape[1]) < 3:
            return np.zeros((0, STEM_CHANNELS), dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(mel, dtype=np.float32)).unsqueeze(0)
        return self.forward(tensor)[0].detach().cpu().numpy().copy()


def stem_span(
    y: np.ndarray,
    t_lo_ms: float,
    t_hi_ms: float,
    stem: AudioStem,
    *,
    sample_rate: int = SAMPLE_RATE,
    mu: float = MEL_NORM_MU,
    sigma: float = MEL_NORM_SIGMA,
) -> StemBatch:
    """Offline stem rows. The copy is detached and is not a training batch."""
    span = mel_span(y, t_lo_ms, t_hi_ms, sample_rate=sample_rate, mu=mu, sigma=sigma)
    if span.mel.shape[1] == 0:
        empty = np.zeros((0, STEM_CHANNELS), dtype=np.float32)
        return StemBatch(empty, span.frame_lo, span.frame_hi)
    return StemBatch(stem.rows(span.mel), span.frame_lo, span.frame_hi)
