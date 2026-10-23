"""Locked §6 widths. RoPE base stays at the standard 10000."""

from __future__ import annotations

D_MODEL = 512
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS
N_LAYERS = 8
FFN_HIDDEN = 1536
ROPE_BASE = 10000.0
# Encoder key window is [e - ENC_RADIUS, e + ENC_RADIUS).
ENC_RADIUS = 512
# Decoder self-attention keeps 0 <= q - k < DEC_SPAN.
DEC_SPAN = 1024
CROSS_PAST = 128
CROSS_FUTURE = 384
PHI_HARMONICS = 8
PHASE_DIM = PHI_HARMONICS * 2 + 2
