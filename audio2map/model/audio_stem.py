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
from audio2map.features.audio.v1.mel import (
    buffer_sample_range,
    frame_index,
    mel_power,
    normalize_mel,
    slice_pcm,
)

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
        """``mel`` is ``(n_mels, frames)``. Returns ``(S, channels)`` float32."""
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
    """Stem tokens for ``[t_lo, t_hi)``. The same ``g`` matches a full-song run."""
    if t_hi_ms <= t_lo_ms:
        frame_lo = frame_index(t_lo_ms)
        empty = np.zeros((0, STEM_CHANNELS), dtype=np.float32)
        return StemBatch(empty, frame_lo, frame_lo)
    frame_lo = frame_index(t_lo_ms)
    frame_hi = frame_index(t_hi_ms)
    if frame_hi < frame_lo:
        frame_hi = frame_lo
    start, end = buffer_sample_range(frame_lo, frame_hi)
    normed = normalize_mel(mel_power(slice_pcm(y, start, end), sample_rate), mu=mu, sigma=sigma)
    return StemBatch(stem.rows(normed), frame_lo, frame_hi)
