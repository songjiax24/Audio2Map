"""A diagnostic readout must expose trainable subframe timing without a chart decoder."""

import pytest
import torch

from audio2map.model.v1.onset import OnsetProbe


@pytest.mark.parametrize("hidden", [0, 64])
def test_probe_returns_presence_and_bounded_subframe_predictions_with_gradients(hidden):
    probe = OnsetProbe(8, hidden)
    logit, fraction = probe(torch.randn(7, 8))
    assert logit.shape == fraction.shape == (7,)
    assert torch.all((fraction >= 0) & (fraction <= 1))
    (logit.square().mean() + (fraction - 0.25).square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
