"""Minimal audio-conditioned chart decoder for debug overfit."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from audio2map.audio.tick_features import V2_FEATURE_DIM
from audio2map.data.cond_vec import COND_VEC_DIM
from audio2map.osu.grid_config import TICKS_PER_BAR
from audio2map.osu.row_tokens import build_vocab


class AudioChartModel(nn.Module):
    """Bar-pooled audio prefix + causal token decoder."""

    def __init__(
        self,
        *,
        vocab_size: int | None = None,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        audio_dim: int = V2_FEATURE_DIM,
        cond_dim: int = COND_VEC_DIM,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        vocab_size = vocab_size or len(build_vocab())
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.max_seq_len = max_seq_len

    def _pool_audio_by_bar(self, audio: torch.Tensor, window_bars: int) -> torch.Tensor:
        """``(B, T, F)`` → ``(B, window_bars, d_model)``."""
        b, t, _ = audio.shape
        expected = window_bars * TICKS_PER_BAR
        if t < expected:
            pad = audio.new_zeros(b, expected - t, audio.shape[-1])
            audio = torch.cat([audio, pad], dim=1)
        elif t > expected:
            audio = audio[:, :expected]
        bars = audio.view(b, window_bars, TICKS_PER_BAR, -1).mean(dim=2)
        return self.audio_proj(bars)

    def forward(
        self,
        audio: torch.Tensor,
        cond_vec: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits ``(B, L-1, vocab)`` for next-token prediction."""
        b, seq_len = token_ids.shape
        window_bars = max(1, audio.shape[1] // TICKS_PER_BAR)

        audio_h = self._pool_audio_by_bar(audio, window_bars)
        cond_h = self.cond_proj(cond_vec).unsqueeze(1)
        prefix = torch.cat([cond_h, audio_h], dim=1)
        prefix_len = prefix.shape[1]

        tok_in = token_ids[:, :-1]
        if tok_in.numel() and int(tok_in.max()) >= self.vocab_size:
            raise ValueError(
                f"token id {int(tok_in.max())} >= vocab_size {self.vocab_size}"
            )
        if tok_in.shape[1] > self.max_seq_len:
            raise ValueError(
                f"token sequence length {tok_in.shape[1]} > max_seq_len {self.max_seq_len}"
            )
        tok_h = self.token_emb(tok_in)
        pos = torch.arange(tok_h.shape[1], device=token_ids.device).unsqueeze(0)
        tok_h = tok_h + self.pos_emb(pos)

        seq = torch.cat([prefix, tok_h], dim=1)
        total_len = seq.shape[1]

        if attn_mask is not None:
            valid = torch.cat(
                [
                    torch.ones(b, prefix_len, dtype=torch.bool, device=token_ids.device),
                    attn_mask[:, :-1],
                ],
                dim=1,
            )
            key_pad = ~valid
        else:
            key_pad = None

        attn_bias = torch.zeros(total_len, total_len, device=token_ids.device)
        attn_bias = attn_bias.masked_fill(
            torch.triu(torch.ones(total_len, total_len, device=token_ids.device, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )
        h = self.transformer(
            seq,
            mask=attn_bias,
            src_key_padding_mask=key_pad,
        )
        h = h[:, prefix_len:]
        return self.lm_head(h)

    def next_token_logits(
        self,
        audio: torch.Tensor,
        cond_vec: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits for the token following ``token_ids`` (``(B, vocab)``)."""
        pad = torch.zeros(
            token_ids.shape[0],
            1,
            dtype=token_ids.dtype,
            device=token_ids.device,
        )
        padded = torch.cat([token_ids, pad], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat(
                [attn_mask, torch.ones(token_ids.shape[0], 1, dtype=torch.bool, device=token_ids.device)],
                dim=1,
            )
        return self.forward(audio, cond_vec, padded, attn_mask=attn_mask)[:, -1, :]

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked CE; ``loss_mask`` aligned with ``targets`` positions."""
        b, l, v = logits.shape
        logits = logits.reshape(b * l, v)
        targets = targets.reshape(b * l)
        mask = loss_mask.reshape(b * l)
        ce = F.cross_entropy(logits, targets, reduction="none")
        if mask.sum() == 0:
            return ce.mean()
        return (ce * mask).sum() / mask.sum()

    def training_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self.forward(
            batch["audio"],
            batch["cond_vec"],
            batch["token_ids"],
            attn_mask=batch["attn_mask"],
        )
        targets = batch["token_ids"][:, 1:]
        mask = batch["loss_mask"][:, 1:]
        loss = self.compute_loss(logits, targets, mask)
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = ((pred == targets) & mask.bool()).float().sum() / mask.sum().clamp(min=1)
        return loss, {"loss": float(loss.item()), "token_acc": float(acc.item())}

    def save_checkpoint(self, path: Path, *, extra: dict | None = None) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "model_cfg": {
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "n_heads": self.transformer.layers[0].self_attn.num_heads,
                "n_layers": len(self.transformer.layers),
            },
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: Path, device: torch.device) -> AudioChartModel:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(**ckpt.get("model_cfg", {}))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
