"""Tick audio encoder + causal chart decoder.

log-mel ticks — Linear → RoPE encoder + ``pos_in_bar`` + cond → memory
→ causal ``TransformerDecoder`` (learned position + cond + cross-attention).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from audio2map.features.audio.tick_features import AUDIO_FEATURE_DIM
from audio2map.features.cond import COND_VEC_DIM
from audio2map.grid import TICKS_PER_BAR
from audio2map.model.config import MAX_AUDIO_TICKS, MAX_DECODER_LEN, model_data_versions
from audio2map.model.rotary import RoPETransformerEncoder
from audio2map.tokens import build_vocab

_MODEL_INIT_KEYS = frozenset(
    {
        "vocab_size",
        "d_model",
        "n_heads",
        "encoder_layers",
        "decoder_layers",
        "dropout",
        "audio_dim",
        "cond_dim",
        "max_audio_ticks",
        "max_decoder_len",
    }
)


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool),
        diagonal=1,
    )


class AudioChartModel(nn.Module):
    """Tick-level audio encoder + autoregressive chart decoder."""

    architecture = "enc_dec"
    audio_pos_encoding = "rope+pos_in_bar"
    audio_frontend = "linear"

    def __init__(
        self,
        *,
        vocab_size: int | None = None,
        d_model: int = 256,
        n_heads: int = 4,
        encoder_layers: int = 4,
        decoder_layers: int = 6,
        dropout: float = 0.1,
        audio_dim: int = AUDIO_FEATURE_DIM,
        cond_dim: int = COND_VEC_DIM,
        max_audio_ticks: int = MAX_AUDIO_TICKS,
        max_decoder_len: int = MAX_DECODER_LEN,
    ) -> None:
        super().__init__()
        vocab_size = vocab_size or len(build_vocab())
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.dropout_p = dropout
        self.audio_dim = audio_dim
        self.cond_dim = cond_dim
        self.max_audio_ticks = max_audio_ticks
        self.max_decoder_len = max_decoder_len

        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.pos_in_bar = nn.Embedding(TICKS_PER_BAR, d_model)

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.audio_encoder = RoPETransformerEncoder(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            max_seq_len=max_audio_ticks,
            num_layers=encoder_layers,
        )

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.decoder_pos = nn.Embedding(max_decoder_len, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.chart_decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def _audio_bar_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Learned within-bar phase; windows must start on a bar boundary."""
        return self.pos_in_bar(t % TICKS_PER_BAR)

    def encode_audio(
        self,
        audio: torch.Tensor,
        cond_vec: torch.Tensor,
        *,
        audio_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(memory, cond_h)`` with shapes ``(B, T, d)``, ``(B, 1, d)``."""
        b, t, _ = audio.shape
        if t > self.max_audio_ticks:
            raise ValueError(f"audio length {t} > max_audio_ticks {self.max_audio_ticks}")

        tick_idx = torch.arange(t, device=audio.device)
        x = self.audio_proj(audio)
        x = x + self._audio_bar_embedding(tick_idx).unsqueeze(0)
        cond_h = self.cond_proj(cond_vec).unsqueeze(1)
        x = x + cond_h

        if audio_mask is None:
            audio_mask = torch.ones(b, t, dtype=torch.bool, device=audio.device)
        memory = self.audio_encoder(x, src_key_padding_mask=~audio_mask)
        return memory, cond_h

    def forward(
        self,
        audio: torch.Tensor,
        cond_vec: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits ``(B, L-1, vocab)`` for next-token prediction."""
        memory, cond_h = self.encode_audio(audio, cond_vec, audio_mask=audio_mask)

        tok_in = token_ids[:, :-1]
        if tok_in.numel() and int(tok_in.max()) >= self.vocab_size:
            raise ValueError(
                f"token id {int(tok_in.max())} >= vocab_size {self.vocab_size}"
            )
        dec_len = tok_in.shape[1]
        if dec_len > self.max_decoder_len:
            raise ValueError(
                f"decoder length {dec_len} > max_decoder_len {self.max_decoder_len}"
            )

        pos = torch.arange(dec_len, device=token_ids.device).unsqueeze(0)
        dec_x = self.token_emb(tok_in) + self.decoder_pos(pos) + cond_h

        h = self.chart_decoder(
            dec_x,
            memory,
            tgt_mask=_causal_mask(dec_len, token_ids.device),
            tgt_key_padding_mask=None if attn_mask is None else ~attn_mask[:, :-1],
            memory_key_padding_mask=None if audio_mask is None else ~audio_mask,
        )
        return self.lm_head(h)

    def next_token_logits(
        self,
        audio: torch.Tensor,
        cond_vec: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits for the token after ``token_ids``.

        ``forward`` always predicts ``token_ids[:, 1:]`` from ``token_ids[:, :-1]``,
        so a dummy last column is appended and discarded. Its value is unused.
        """
        pad = torch.zeros(
            token_ids.shape[0],
            1,
            dtype=token_ids.dtype,
            device=token_ids.device,
        )
        padded = torch.cat([token_ids, pad], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat(
                [
                    attn_mask,
                    torch.ones(
                        token_ids.shape[0], 1, dtype=torch.bool, device=token_ids.device
                    ),
                ],
                dim=1,
            )
        return self.forward(
            audio,
            cond_vec,
            padded,
            attn_mask=attn_mask,
            audio_mask=audio_mask,
        )[:, -1, :]

    def checkpoint_model_cfg(self) -> dict:
        return {
            "architecture": self.architecture,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "encoder_layers": len(self.audio_encoder.layers),
            "decoder_layers": len(self.chart_decoder.layers),
            "dropout": self.dropout_p,
            "audio_dim": self.audio_dim,
            "cond_dim": self.cond_dim,
            "max_audio_ticks": self.max_audio_ticks,
            "max_decoder_len": self.max_decoder_len,
            "audio_pos_encoding": self.audio_pos_encoding,
            "audio_frontend": self.audio_frontend,
            **model_data_versions(),
        }

    def save_checkpoint(self, path: Path, *, extra: dict | None = None) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "model_cfg": self.checkpoint_model_cfg(),
        }
        if extra:
            payload.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.part")
        torch.save(payload, tmp)
        tmp.replace(path)

    @classmethod
    def load_checkpoint(cls, path: Path, device: torch.device) -> AudioChartModel:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        raw = dict(ckpt.get("model_cfg", {}))
        cfg = {key: raw[key] for key in _MODEL_INIT_KEYS if key in raw}
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model


def build_model(**kwargs) -> AudioChartModel:
    return AudioChartModel(**kwargs)


def load_checkpoint(path: Path, device: torch.device) -> AudioChartModel:
    return AudioChartModel.load_checkpoint(path, device)
