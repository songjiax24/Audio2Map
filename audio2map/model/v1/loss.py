"""Plain next-token cross-entropy. PAD is ignored. BOS is only a prefix."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from audio2map.tokens.v1.vocab import TOKEN_BEAT, TOKEN_EOS, TOKEN_PAD, build_vocab

_VOCAB = build_vocab()
_PAD_ID = _VOCAB[TOKEN_PAD]
_EOS_ID = _VOCAB[TOKEN_EOS]
_BEAT_ID = _VOCAB[TOKEN_BEAT]
_ACC_TYPES = ("all", "pos", "row", "beat", "eos")


def _id_span(prefix: str) -> tuple[int, int]:
    ids = [index for token, index in _VOCAB.items() if token.startswith(prefix)]
    if not ids:
        raise RuntimeError(f"vocab has no {prefix} tokens")
    lo, hi = min(ids), max(ids)
    if hi - lo + 1 != len(ids):
        raise RuntimeError(f"{prefix} ids are not contiguous")
    return lo, hi + 1


_POS_LO, _POS_HI = _id_span("<POS_")
_ROW_LO, _ROW_HI = _id_span("<ROW_")


def plain_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """``logits`` is ``(B, L, V)`` and ``targets`` is ``(B, L)``."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=_PAD_ID,
    )


def nonpad_count(targets: torch.Tensor) -> int:
    """How many targets ``plain_ce`` averages. PAD is ignored."""
    return int((targets != _PAD_ID).sum().item())


def accuracy_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, tuple[int, int]]:
    """``(correct, total)`` over non-PAD targets, and over each gold type.

    A hit is ``argmax`` equal to the target. ``pos`` / ``row`` / ``beat`` /
    ``eos`` count only targets of that type. PAD is in none of them.
    """
    pred = logits.argmax(dim=-1)
    hit = pred.eq(targets)
    valid = targets != _PAD_ID
    masks = torch.stack(
        (
            valid,
            valid & (targets >= _POS_LO) & (targets < _POS_HI),
            valid & (targets >= _ROW_LO) & (targets < _ROW_HI),
            valid & targets.eq(_BEAT_ID),
            valid & targets.eq(_EOS_ID),
        )
    )
    correct = (hit & masks).flatten(1).sum(dim=1)
    total = masks.flatten(1).sum(dim=1)
    pairs = torch.stack((correct, total)).detach().cpu()
    return {
        name: (int(pairs[0, index]), int(pairs[1, index]))
        for index, name in enumerate(_ACC_TYPES)
    }
