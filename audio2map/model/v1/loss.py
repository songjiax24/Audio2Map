"""Plain next-token cross-entropy. PAD is ignored. BOS is only a prefix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


def ce_by_type(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, tuple[float, int]]:
    """``(sum, count)`` of per-token cross-entropy for each gold type.

    The mean is ``sum / count``. PAD is ignored. A non-PAD target outside
    ``POS`` / ``ROW`` / ``BEAT`` / ``EOS`` is an error. Sums are fp32.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_targets = targets.reshape(-1)
    ce = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="none",
        ignore_index=_PAD_ID,
    )
    valid = flat_targets != _PAD_ID
    type_masks = (
        valid & (flat_targets >= _POS_LO) & (flat_targets < _POS_HI),
        valid & (flat_targets >= _ROW_LO) & (flat_targets < _ROW_HI),
        valid & flat_targets.eq(_BEAT_ID),
        valid & flat_targets.eq(_EOS_ID),
    )
    covered = type_masks[0] | type_masks[1] | type_masks[2] | type_masks[3]
    if int((valid & ~covered).sum()) != 0:
        raise RuntimeError("a non-PAD target is not POS, ROW, BEAT, or EOS")
    out: dict[str, tuple[float, int]] = {}
    for name, mask in zip(_LEGAL_TYPES, type_masks, strict=True):
        count = int(mask.sum())
        total = float((ce * mask).sum()) if count else 0.0
        out[name] = (total, count)
    return out


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


_LEGAL_TYPES = ("pos", "row", "beat", "eos")


@dataclass(frozen=True, slots=True)
class LegalTally:
    """One chart. ``by_type`` is ``(hits, probability sum, count)`` per gold type."""

    hits: int
    mass: float
    count: int
    by_type: dict[str, tuple[int, float, int]]


def _gold_type(token_id: int) -> str:
    if _POS_LO <= token_id < _POS_HI:
        return "pos"
    if _ROW_LO <= token_id < _ROW_HI:
        return "row"
    if token_id == _BEAT_ID:
        return "beat"
    if token_id == _EOS_ID:
        return "eos"
    raise RuntimeError(f"target id {token_id} is not POS, ROW, BEAT, or EOS")


def legal_tally(
    logits: torch.Tensor,
    targets: torch.Tensor,
    rows: tuple[np.ndarray, ...],
) -> LegalTally:
    """Hits and legal-probability mass for one chart, also split by gold type.

    ``logits`` predicts ``targets``. ``rows[i]`` is the legal ids for the
    i-th non-PAD target. Top-1 hits when argmax is in that row. The
    probability is the softmax mass on those ids. The type is the gold id.
    """
    if logits.shape[0] != 1 or targets.shape[0] != 1:
        raise ValueError("legal tally is one chart")
    work = logits[0].detach().float()
    gold = targets[0]
    keep = gold != _PAD_ID
    if int(keep.sum()) != len(rows):
        raise ValueError("legal rows do not match non-PAD targets")
    probs = torch.softmax(work, dim=-1).cpu().numpy()
    pred = work.argmax(dim=-1).cpu().numpy()
    gold_cpu = gold.detach().cpu().numpy()
    hits = 0
    mass = 0.0
    type_hits = {name: 0 for name in _LEGAL_TYPES}
    type_mass = {name: 0.0 for name in _LEGAL_TYPES}
    type_count = {name: 0 for name in _LEGAL_TYPES}
    row_index = 0
    seen = np.zeros(probs.shape[-1], dtype=bool)
    for index, token_id in enumerate(gold_cpu):
        if token_id == _PAD_ID:
            continue
        allowed = rows[row_index]
        row_index += 1
        seen[allowed] = True
        hit = int(seen[pred[index]])
        seen[allowed] = False
        piece = float(probs[index, allowed].sum())
        hits += hit
        mass += piece
        kind = _gold_type(int(token_id))
        type_hits[kind] += hit
        type_mass[kind] += piece
        type_count[kind] += 1
    by_type = {
        name: (type_hits[name], type_mass[name], type_count[name])
        for name in _LEGAL_TYPES
    }
    return LegalTally(hits, mass, len(rows), by_type)
