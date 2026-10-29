"""Window rules and one fake-batch backward through the 8/8 model."""

from __future__ import annotations

import pytest
import torch

import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from audio2map.features.audio.v1.constants import N_MELS
from audio2map.model.v1.blocks import SelfAttention, _fill_empty, _sdpa_mask, _zero_empty
from audio2map.model.v1.constants import D_MODEL, N_LAYERS
from audio2map.model.v1.loss import plain_ce
from audio2map.model.v1.model import V1Batch, V1ChartModel
from audio2map.model.v1.windows import (
    DECODER_WINDOW,
    ENCODER_WINDOW,
    cross_allow,
    decoder_allow,
    decoder_keeps,
    encoder_allow,
    encoder_keeps,
)
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


def _band(length: int, query: int, window: tuple[int, int]) -> torch.Tensor:
    left, right = window
    key = torch.arange(length)
    return (key >= query - left) & (key <= query + right)


def test_flash_window_is_the_allow_set() -> None:
    assert ENCODER_WINDOW == (512, 511)
    assert DECODER_WINDOW == (1023, 0)
    enc = encoder_allow(2000, torch.device("cpu"))
    for query in (0, 10, 600, 1999):
        got = _band(2000, query, ENCODER_WINDOW)
        assert torch.equal(got, enc[query])
        assert torch.equal(got, encoder_keeps(torch.tensor(query), torch.arange(2000)))
    assert int(_band(2000, 600, ENCODER_WINDOW).sum()) == 1024
    dec = decoder_allow(1030, torch.device("cpu"))
    for query in (0, 5, 1024, 1029):
        got = _band(1030, query, DECODER_WINDOW)
        assert torch.equal(got, dec[query])
        assert torch.equal(got, decoder_keeps(torch.tensor(query), torch.arange(1030)))
        assert int(got.sum()) <= 1024
    assert int(_band(1030, 1024, DECODER_WINDOW).sum()) == 1024


def test_prefix_visibility_changes_the_output_and_holes_fail() -> None:
    torch.manual_seed(0)
    attn = SelfAttention(ENCODER_WINDOW)
    x = torch.randn(1, 8, D_MODEL)
    positions = torch.arange(8)
    full = torch.ones(1, 8, dtype=torch.bool)
    tail = torch.ones(1, 8, dtype=torch.bool)
    tail[0, 6:] = False
    assert not torch.allclose(attn(x, positions, full), attn(x, positions, tail))
    hole = full.clone()
    hole[0, 3] = False
    with pytest.raises(ValueError, match="valid prefix"):
        attn(x, positions, hole)


def _dense_bool_forward(attn: SelfAttention, x: torch.Tensor, positions: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
    """Retired full-sequence bool mask, kept only as the numeric control."""
    batch, length, _ = x.shape
    q = attn.q_proj(x).view(batch, length, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(batch, length, attn.n_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(batch, length, attn.n_heads, attn.head_dim).transpose(1, 2)
    q = attn.rotary.apply(q, positions)
    k = attn.rotary.apply(k, positions)
    usable, empty = _fill_empty(allow)
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=_sdpa_mask(usable))
    out = _zero_empty(out, empty)
    out = out.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
    return attn.out_proj(out)


@pytest.mark.parametrize(
    ("window", "keeps"),
    [(ENCODER_WINDOW, encoder_keeps), (DECODER_WINDOW, decoder_keeps)],
)
def test_sliding_window_matches_dense_bool_mask(window: tuple[int, int], keeps) -> None:
    torch.manual_seed(0)
    length = 48
    lengths = torch.tensor([30, length])
    query = torch.arange(length)[:, None]
    key = torch.arange(length)[None, :]
    visible = (key[None] < lengths[:, None, None]) & (query[None] < lengths[:, None, None])
    allow = keeps(query, key)[None] & visible
    attn = SelfAttention(window)
    x = torch.randn(2, length, D_MODEL)
    positions = torch.arange(length)
    keep = torch.arange(length)[None, :] < lengths[:, None]
    left = x.detach().requires_grad_(True)
    right = x.detach().requires_grad_(True)
    got = attn(left, positions, keep)
    attn.zero_grad(set_to_none=True)
    got.sum().backward()
    got_weight = attn.q_proj.weight.grad.detach().clone()
    attn.zero_grad(set_to_none=True)
    ref = _dense_bool_forward(attn, right, positions, allow)
    ref.sum().backward()
    assert (got - ref).abs().max() < 1e-4
    assert (left.grad - right.grad).abs().max() < 1e-4
    assert (got_weight - attn.q_proj.weight.grad).abs().max() < 1e-4


def test_fake_batch_backward() -> None:
    torch.manual_seed(0)
    model = V1ChartModel()
    assert len(model.encoder.layers) == N_LAYERS == 8
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
        log_bpm=torch.zeros(batch_size, frames),
        rel_u=torch.zeros(batch_size, frames),
        mel=torch.randn(batch_size, N_MELS, 2 * frames + 3),
        n_frames=torch.full((batch_size,), frames, dtype=torch.long),
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
            log_bpm=batch.log_bpm,
            rel_u=batch.rel_u,
            mel=batch.mel,
            n_frames=batch.n_frames,
        )
    )
    assert torch.allclose(first, other)
    loss = plain_ce(first, tokens[:, 1:])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.token_emb.weight.grad is not None
    assert torch.count_nonzero(model.decoder[0].sa_mod.proj.weight.grad) > 0
