"""Window rules and one fake-batch backward through the 8/8 model."""

from __future__ import annotations

import torch

from audio2map.model.v1.blocks import SelfAttention
from audio2map.model.v1.constants import D_MODEL, N_LAYERS
from audio2map.model.v1.loss import plain_ce
from audio2map.model.v1.model import V1Batch, V1ChartModel
from audio2map.model.v1.windows import cross_allow, decoder_allow, encoder_allow
from audio2map.tokens.v1.vocab import TOKEN_PAD, build_vocab


def test_windows_follow_the_locked_ranges() -> None:
    enc = encoder_allow(2000, torch.device("cpu"))
    assert bool(enc[600, 600 - 512])
    assert not bool(enc[600, 600 - 513])
    assert bool(enc[600, 600 + 511])
    assert not bool(enc[600, 600 + 512])
    dec = decoder_allow(1030, torch.device("cpu"))
    assert bool(dec[1024, 1024])
    assert bool(dec[1024, 1])
    assert not bool(dec[1024, 0])
    assert not bool(dec[5, 6])


def test_cross_window_drops_keys_outside_the_past_and_future() -> None:
    g = torch.tensor([[0.0, 71.0, 72.0, 200.0, 583.0, 584.0]])
    p_q = torch.tensor([[200.4]])
    allowed = cross_allow(g, p_q)[0, 0].tolist()
    assert allowed == [False, False, True, True, True, False]


def test_true_mask_keeps_the_key() -> None:
    torch.manual_seed(0)
    attn = SelfAttention()
    x = torch.randn(1, 2, D_MODEL)
    positions = torch.arange(2)
    both = torch.ones(1, 2, 2, dtype=torch.bool)
    only_self = torch.tensor([[[True, False], [False, True]]])
    mixed = attn(x, positions, both)
    kept = attn(x, positions, only_self)
    assert not torch.allclose(mixed, kept)


def test_fake_batch_backward() -> None:
    torch.manual_seed(0)
    model = V1ChartModel()
    assert len(model.encoder) == N_LAYERS == 8
    assert len(model.decoder) == 8
    for layer in model.decoder:
        assert torch.count_nonzero(layer.sa_mod.proj.weight) == 0
        assert torch.count_nonzero(layer.ca_mod.proj.bias) == 0
        assert torch.count_nonzero(layer.ff_mod.proj.weight) == 0
    vocab = build_vocab()
    batch_size, frames, length = 2, 12, 6
    tokens = torch.randint(0, len(vocab), (batch_size, length))
    tokens[:, 0] = vocab["<BOS>"]
    tokens[:, -1] = vocab[TOKEN_PAD]
    cond_m = torch.ones(batch_size, 18)
    cond_m[:, 1] = 0
    batch = V1Batch(
        frame_lo=torch.tensor([0, 40]),
        cond_x=torch.rand(batch_size, 18),
        cond_m=cond_m,
        token_ids=tokens,
        cursor_time_ms=torch.linspace(-20.0, 80.0, length - 1).repeat(batch_size, 1),
        phi=torch.rand(batch_size, frames),
        beat_b=torch.tensor([-1.0, 0.0, 2.0]).repeat(batch_size, frames // 3),
        audio=torch.randn(batch_size, frames, D_MODEL),
        precomputed=True,
    )
    model.train()
    first = model(batch)
    other = model(
        V1Batch(
            frame_lo=batch.frame_lo,
            cond_x=torch.rand(batch_size, 18),
            cond_m=torch.ones(batch_size, 18),
            token_ids=batch.token_ids,
            cursor_time_ms=batch.cursor_time_ms,
            phi=batch.phi,
            beat_b=batch.beat_b,
            audio=batch.audio,
            precomputed=True,
        )
    )
    assert torch.allclose(first, other)
    loss = plain_ce(first, tokens[:, 1:])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.token_emb.weight.grad is not None
    assert torch.count_nonzero(model.decoder[0].sa_mod.proj.weight.grad) > 0
