"""Pad full-chart batches. Song silence stays a real frame; pad does not."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from audio2map.features.audio.v1.constants import N_MELS
from audio2map.model.v1.batch import V1Batch
from audio2map.tokens.v1.vocab import TOKEN_PAD, build_vocab

_PAD_ID = build_vocab()[TOKEN_PAD]


def collate_v1(examples: Sequence[V1Batch]) -> V1Batch:
    """Stack batch-size-1 charts. Right-pad mel (or precomputed rows) and tokens.

    ``audio_mask`` / ``token_mask`` are false only on the padded tail.
    A silent frame that belongs to the chart stays true. Token pad ids come
    from the v1 vocabulary, not a literal.
    """
    if not examples:
        raise ValueError("collate_v1 needs at least one example")
    precomputed = examples[0].precomputed
    for item in examples:
        if item.token_ids.shape[0] != 1:
            raise ValueError("collate_v1 expects one chart per example")
        if item.cursor_time_ms.shape[1] != item.token_ids.shape[1] - 1:
            raise ValueError("cursor_time_ms must align with token_ids[:, :-1]")
        if item.precomputed != precomputed:
            raise ValueError("cannot collate precomputed rows with mel")
        if item.n_frames is None:
            raise ValueError("collate_v1 needs n_frames")
    counts = [int(item.n_frames[0]) for item in examples]
    max_frames = max(counts)
    max_tokens = max(int(item.token_ids.shape[1]) for item in examples)
    batch = len(examples)
    phi = examples[0].phi.new_zeros(batch, max_frames)
    beat_b = examples[0].beat_b.new_zeros(batch, max_frames)
    token_ids = examples[0].token_ids.new_full((batch, max_tokens), _PAD_ID)
    cursor = examples[0].cursor_time_ms.new_zeros(batch, max_tokens - 1)
    audio_mask = torch.zeros(batch, max_frames, dtype=torch.bool)
    token_mask = torch.zeros(batch, max_tokens - 1, dtype=torch.bool)
    frame_lo = torch.empty(batch, dtype=torch.long)
    n_frames = torch.tensor(counts, dtype=torch.long)
    cond_x = torch.stack([item.cond_x[0] for item in examples], dim=0)
    cond_m = torch.stack([item.cond_m[0] for item in examples], dim=0)
    mel = None
    audio = None
    if precomputed:
        width = examples[0].audio.shape[-1]
        audio = examples[0].audio.new_zeros(batch, max_frames, width)
    else:
        mel = examples[0].mel.new_zeros(batch, N_MELS, 2 * max_frames + 3)
    for index, item in enumerate(examples):
        frames = counts[index]
        tokens = int(item.token_ids.shape[1])
        phi[index, :frames] = item.phi[0, :frames]
        beat_b[index, :frames] = item.beat_b[0, :frames]
        token_ids[index, :tokens] = item.token_ids[0]
        cursor[index, : tokens - 1] = item.cursor_time_ms[0]
        audio_mask[index, :frames] = _kept(item.audio_mask, frames)
        token_mask[index, : tokens - 1] = _kept(item.token_mask, tokens - 1)
        frame_lo[index] = int(item.frame_lo[0])
        if precomputed:
            audio[index, :frames] = item.audio[0, :frames]
        else:
            mel_len = 2 * frames + 3
            mel[index, :, :mel_len] = item.mel[0, :, :mel_len]
    return V1Batch(
        frame_lo=frame_lo,
        cond_x=cond_x,
        cond_m=cond_m,
        token_ids=token_ids,
        cursor_time_ms=cursor,
        phi=phi,
        beat_b=beat_b,
        audio_mask=audio_mask,
        token_mask=token_mask,
        mel=mel,
        n_frames=n_frames,
        audio=audio,
        precomputed=precomputed,
    )


def _kept(mask: torch.Tensor | None, length: int) -> torch.Tensor:
    if mask is None:
        return torch.ones(length, dtype=torch.bool)
    return mask[0, :length].to(dtype=torch.bool)
