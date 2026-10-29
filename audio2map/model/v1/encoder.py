"""Stem plus the eight local transformer layers. Phase is added outside."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from audio2map.model.audio_stem import STEM_CHANNELS, AudioStem
from audio2map.model.v1.blocks import EncoderLayer
from audio2map.model.v1.constants import D_MODEL, N_LAYERS


def run_layer(layer: nn.Module, *args: object, recompute: bool) -> torch.Tensor:
    """Run one encoder or decoder layer. Training can recompute it on the way back."""
    if recompute:
        return checkpoint(layer, *args, use_reentrant=False)
    return layer(*args)


class AudioEncoder(nn.Module):
    """Conv stem and eight local self-attention layers.

    ``forward`` takes normalized mel. ``forward_rows`` is the layer stack for
    a tensor that is already ``(B, S, 512)``. It is not a dataset format.
    """

    def __init__(self, stem: AudioStem | None = None) -> None:
        super().__init__()
        self.stem = stem if stem is not None else AudioStem()
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(N_LAYERS))

    def forward(
        self,
        mel: torch.Tensor,
        frame_lo: torch.Tensor,
        audio_mask: torch.Tensor | None,
        *,
        n_frames: torch.Tensor | None = None,
        recompute: bool = False,
    ) -> torch.Tensor:
        """Mel ``(B, n_mels, T)`` in, encoder memory ``(B, S, D)`` out."""
        rows = self.stem_mel(mel, n_frames)
        if audio_mask is None:
            mask = torch.ones(rows.shape[:2], dtype=torch.bool, device=rows.device)
        else:
            mask = audio_mask.to(device=rows.device)
        return self.forward_rows(rows, frame_lo, mask, recompute=recompute)

    def stem_mel(self, mel: torch.Tensor, n_frames: torch.Tensor | None) -> torch.Tensor:
        """Conv each example on its own ``2 * count + 3`` mel frames.

        Batch padding stays out of the convolution. A single unpadded window
        passes ``n_frames=None``.
        """
        if n_frames is None:
            return self.stem(mel)
        device = self.stem.conv1.weight.device
        spec = mel.to(device=device)
        counts = n_frames.to(device=device)
        width = int(counts.max())
        parts: list[torch.Tensor] = []
        for index in range(spec.shape[0]):
            count = int(counts[index])
            if count <= 0:
                raise ValueError("stem span is empty")
            produced = self.stem(spec[index : index + 1, :, : 2 * count + 3])
            if produced.shape[1] != count:
                raise ValueError(f"stem length {produced.shape[1]} != {count}")
            if count < width:
                pad = produced.new_zeros(1, width - count, STEM_CHANNELS)
                produced = torch.cat((produced, pad), dim=1)
            parts.append(produced)
        return torch.cat(parts, dim=0)

    def forward_rows(
        self,
        audio: torch.Tensor,
        frame_lo: torch.Tensor,
        audio_mask: torch.Tensor,
        *,
        recompute: bool = False,
    ) -> torch.Tensor:
        """Eight layers on stem rows ``(B, S, D)``."""
        _batch, length, channels = audio.shape
        if channels != D_MODEL:
            raise ValueError(f"stem width {channels} != {D_MODEL}")
        if audio_mask.shape[:2] != audio.shape[:2]:
            raise ValueError(f"audio mask {tuple(audio_mask.shape)} != {tuple(audio.shape[:2])}")
        local = torch.arange(length, device=audio.device)
        positions = frame_lo.to(device=audio.device).unsqueeze(1) + local
        hidden = audio
        for layer in self.layers:
            hidden = run_layer(layer, hidden, positions, audio_mask, recompute=recompute)
        return hidden
