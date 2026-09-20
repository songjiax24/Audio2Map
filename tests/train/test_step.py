"""Tests for batch padding and masked CE."""

from __future__ import annotations

import torch

from audio2map.features.cond import COND_VEC_DIM
from audio2map.grid import TICKS_PER_BAR
from audio2map.model.model import AudioChartModel
from audio2map.train.data import pad_batch
from audio2map.train.step import (
    compute_loss,
    scale_grads_by_valid_tokens,
    token_nll_stats,
    training_step,
)


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
    assert stats["valid_tokens"] > 0
    assert 0.0 <= stats["token_acc"] <= 1.0


def test_token_nll_stats_accum_is_token_weighted() -> None:
    v = 8
    hit = torch.zeros(1, 1, v)
    hit[0, 0, 0] = 5.0
    miss = torch.zeros(1, 10, v)
    miss[0, :, 3] = 5.0
    targets_hit = torch.zeros(1, 1, dtype=torch.long)
    targets_miss = torch.ones(1, 10, dtype=torch.long)
    sum_hit, n_hit, c_hit = token_nll_stats(hit, targets_hit, torch.ones(1, 1))
    sum_miss, n_miss, c_miss = token_nll_stats(miss, targets_miss, torch.ones(1, 10))
    token_acc = (c_hit + c_miss) / (n_hit + n_miss)
    last_only = c_miss / n_miss
    assert token_acc.item() != last_only.item()
    token_loss = (sum_hit + sum_miss) / (n_hit + n_miss)
    last_loss = sum_miss / n_miss
    assert token_loss.item() != last_loss.item()
    assert n_hit.item() == 1.0
    assert n_miss.item() == 10.0


def test_scale_grads_matches_concatenated_token_mean() -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 8, bias=False)
    x1 = torch.randn(1, 4)
    x2 = torch.randn(10, 4)
    y1 = torch.zeros(1, dtype=torch.long)
    y2 = torch.ones(10, dtype=torch.long)

    model.zero_grad(set_to_none=True)
    loss1 = torch.nn.functional.cross_entropy(model(x1), y1, reduction="sum")
    loss2 = torch.nn.functional.cross_entropy(model(x2), y2, reduction="sum")
    (loss1 + loss2).backward()
    scale_grads_by_valid_tokens(model, 11.0)
    got = model.weight.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    logits = model(torch.cat([x1, x2], dim=0))
    targets = torch.cat([y1, y2], dim=0)
    mean = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
    mean.backward()
    assert torch.allclose(got, model.weight.grad, atol=1e-6, rtol=1e-5)
