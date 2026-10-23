"""Plain next-token cross-entropy. PAD is ignored. BOS is only a prefix."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from audio2map.tokens.v1.vocab import TOKEN_PAD, build_vocab

_PAD_ID = build_vocab()[TOKEN_PAD]


def plain_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """``logits`` is ``(B, L, V)`` and ``targets`` is ``(B, L)``."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=_PAD_ID,
    )
