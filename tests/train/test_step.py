"""Tests for batch padding and masked CE."""

from __future__ import annotations

import torch

from audio2map.features.cond import COND_VEC_DIM
from audio2map.grid import TICKS_PER_BAR
from audio2map.model.model import AudioChartModel
from audio2map.train.data import pad_batch
from audio2map.train.step import compute_loss, training_step


def test_compute_loss_empty_mask_returns_zero() -> None:
    b, l, v = 2, 8, 32
    logits = torch.randn(b, l, v, requires_grad=True)
    targets = torch.zeros(b, l, dtype=torch.long)
    mask = torch.zeros(b, l, dtype=torch.float32)
    loss = compute_loss(logits, targets, mask)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() == 0.0


def test_pad_batch_aligns_tokens_and_audio() -> None:
    short = {
        "token_ids": torch.tensor([1, 2]),
        "loss_mask": torch.tensor([0.0, 1.0]),
        "cond_vec": torch.ones(COND_VEC_DIM),
        "audio": torch.ones(2, 8),
    }
    long = {
        "token_ids": torch.tensor([1, 2, 3]),
        "loss_mask": torch.tensor([0.0, 1.0, 1.0]),
        "cond_vec": torch.zeros(COND_VEC_DIM),
        "audio": torch.ones(4, 8),
    }
    out = pad_batch([short, long])
    assert out["token_ids"].shape == (2, 3)
    assert out["attn_mask"][0].tolist() == [True, True, False]
    assert out["loss_mask"][0, 2].item() == 0.0
    assert out["audio"].shape == (2, 4, 8)
    assert out["audio_mask"][0].tolist() == [True, True, False, False]


def test_training_step_masked_shift() -> None:
    b, bars, feat, seq = 2, 2, 128, 16
    model = AudioChartModel(d_model=32, n_heads=4, encoder_layers=1, decoder_layers=1)
    batch = {
        "audio": torch.randn(b, bars * TICKS_PER_BAR, feat),
        "cond_vec": torch.randn(b, COND_VEC_DIM),
        "token_ids": torch.randint(0, model.vocab_size, (b, seq)),
        "loss_mask": torch.ones(b, seq),
        "attn_mask": torch.ones(b, seq, dtype=torch.bool),
        "audio_mask": torch.ones(b, bars * TICKS_PER_BAR, dtype=torch.bool),
    }
    loss, stats = training_step(model, batch)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert "token_acc" in stats
    assert stats["audio_ticks"] == float(bars * TICKS_PER_BAR)
