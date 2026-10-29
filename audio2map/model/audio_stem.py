"""Conv stem. The first two layers of ``AudioEncoder``.

Mel comes from ``features.audio.v1``. This module does not build it.
"""

from __future__ import annotations

import torch
from torch import nn

from audio2map.features.audio.v1.constants import N_MELS

STEM_CHANNELS = 512


class AudioStem(nn.Module):
    """Conv k=3 s=1, then conv k=3 s=2. No LayerNorm and no absolute positions.

    ``S`` stem frames take ``2S+3`` mel frames. Conv2 emits ``S+2``. The kept
    rows are ``q = 1..S``, aligned with global frame ``g``.
    """

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
