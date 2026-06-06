"""Tests for encoder-decoder AudioChartModel."""

from __future__ import annotations

import torch

from audio2map.data.cond_vec import COND_VEC_DIM
from audio2map.osu.grid_config import TICKS_PER_BAR
from audio2map.training.model import AudioChartModel, build_model, load_checkpoint


def test_enc_dec_forward_shapes() -> None:
    b, bars, feat = 2, 8, 128
    t_audio = bars * TICKS_PER_BAR
    seq = 64
    model = AudioChartModel(d_model=64, n_heads=4, encoder_layers=2, decoder_layers=2)
    audio = torch.randn(b, t_audio, feat)
    cond = torch.randn(b, COND_VEC_DIM)
    token_ids = torch.randint(0, model.vocab_size, (b, seq))
    attn = torch.ones(b, seq, dtype=torch.bool)
    audio_mask = torch.ones(b, t_audio, dtype=torch.bool)

    logits = model(audio, cond, token_ids, attn_mask=attn, audio_mask=audio_mask)
    assert logits.shape == (b, seq - 1, model.vocab_size)

    nxt = model.next_token_logits(audio[:1], cond[:1], token_ids[:1, :10])
    assert nxt.shape == (1, model.vocab_size)


def test_build_model_defaults() -> None:
    m = build_model(d_model=32, layers=2, n_heads=4)
    assert isinstance(m, AudioChartModel)
    assert m.architecture == "enc_dec"
    assert m.audio_pooling == "tick"


def test_compute_loss_empty_mask_returns_zero() -> None:
    b, l, v = 2, 8, 32
    logits = torch.randn(b, l, v, requires_grad=True)
    targets = torch.zeros(b, l, dtype=torch.long)
    mask = torch.zeros(b, l, dtype=torch.float32)
    loss = AudioChartModel.compute_loss(logits, targets, mask)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() == 0.0


def test_checkpoint_roundtrip_enc_dec(tmp_path) -> None:
    model = AudioChartModel(d_model=32, n_heads=4, encoder_layers=2, decoder_layers=2)
    path = tmp_path / "enc_dec.pt"
    model.save_checkpoint(path)
    loaded = load_checkpoint(path, torch.device("cpu"))
    assert isinstance(loaded, AudioChartModel)
    assert loaded.d_model == 32
