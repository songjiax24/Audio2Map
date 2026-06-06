"""Tests for RoPE audio encoder padding mask semantics."""

from __future__ import annotations

import torch

from audio2map.training.model import AudioChartModel


def test_audio_encoder_padding_mask_ignores_padded_keys() -> None:
    """Padded ticks must not affect valid tick outputs (mask on keys, not zero output)."""
    model = AudioChartModel(d_model=32, n_heads=4, encoder_layers=1, decoder_layers=1)
    model.eval()

    b, t, feat = 1, 4, model.audio_dim
    cond = torch.randn(b, model.cond_dim)
    audio_mask = torch.tensor([[True, True, False, False]])

    base = torch.randn(b, t, feat)
    alt = base.clone()
    alt[:, 2:, :] = torch.randn(b, 2, feat) * 100.0

    with torch.no_grad():
        mem_base, _ = model.encode_audio(base, cond, audio_mask=audio_mask)
        mem_alt, _ = model.encode_audio(alt, cond, audio_mask=audio_mask)

    assert torch.isfinite(mem_base).all()
    assert torch.isfinite(mem_alt).all()
    torch.testing.assert_close(mem_base[:, :2, :], mem_alt[:, :2, :], rtol=1e-5, atol=1e-5)
