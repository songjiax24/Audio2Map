"""Tests for RoPE audio encoder components."""

from __future__ import annotations

import torch

from audio2map.grid import TICKS_PER_BAR
from audio2map.model.rotary import RoPETransformerEncoder, apply_rotary_pos_emb, rotate_half


def test_rotate_half_involution() -> None:
    x = torch.randn(2, 4, 8)
    y = rotate_half(rotate_half(x))
    torch.testing.assert_close(y, -x)


def test_rope_encoder_forward_shape() -> None:
    b, t, d = 2, TICKS_PER_BAR * 2, 64
    enc = RoPETransformerEncoder(
        d_model=d,
        nhead=4,
        dim_feedforward=128,
        dropout=0.0,
        max_seq_len=t,
        num_layers=2,
    )
    x = torch.randn(b, t, d)
    mask = torch.ones(b, t, dtype=torch.bool)
    mask[0, -10:] = False
    out = enc(x, src_key_padding_mask=~mask)
    assert out.shape == (b, t, d)


def test_apply_rotary_changes_qk() -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    cos = torch.ones(1, 1, 4, 8)
    sin = torch.zeros(1, 1, 4, 8)
    q2, k2 = apply_rotary_pos_emb(q, k, cos, sin)
    torch.testing.assert_close(q2, q)
    torch.testing.assert_close(k2, k)
