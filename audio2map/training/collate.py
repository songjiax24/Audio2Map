"""Batch padding for variable-length token windows."""

from __future__ import annotations

import torch

from audio2map.osu.row_tokens import TOKEN_PAD, build_vocab

_PAD_ID = build_vocab()[TOKEN_PAD]


def pad_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["token_ids"].shape[0] for item in batch)
    max_audio = max(item["audio"].shape[0] for item in batch)
    feat_dim = batch[0]["audio"].shape[1]

    token_ids = torch.full((len(batch), max_len), _PAD_ID, dtype=torch.long)
    loss_mask = torch.zeros(len(batch), max_len, dtype=torch.float32)
    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    cond_vec = torch.stack([item["cond_vec"] for item in batch], dim=0)
    audio = torch.zeros(len(batch), max_audio, feat_dim, dtype=torch.float32)

    for i, item in enumerate(batch):
        n = item["token_ids"].shape[0]
        token_ids[i, :n] = item["token_ids"]
        loss_mask[i, :n] = item["loss_mask"]
        attn_mask[i, :n] = True
        a = item["audio"].shape[0]
        audio[i, :a] = item["audio"]

    return {
        "token_ids": token_ids,
        "loss_mask": loss_mask,
        "attn_mask": attn_mask,
        "cond_vec": cond_vec,
        "audio": audio,
    }
