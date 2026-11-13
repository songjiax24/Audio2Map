"""Small diagnostic readouts; these are not part of V1ChartModel or its checkpoint."""

from __future__ import annotations

import torch
from torch import nn


class OnsetProbe(nn.Module):
    """One frame's features to an onset logit and a within-frame fraction.

    A zero hidden width selects a linear readout. The second output uses a
    sigmoid so that decoded times remain inside their assigned frame.
    """

    def __init__(self, channels: int, hidden: int = 0) -> None:
        super().__init__()
        if channels <= 0 or hidden < 0:
            raise ValueError("channels must be positive and hidden nonnegative")
        self.readout = (
            nn.Linear(channels, 2)
            if hidden == 0
            else nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, 2))
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.readout(features)
        return output[..., 0], output[..., 1].sigmoid()
