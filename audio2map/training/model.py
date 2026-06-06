"""Audio-conditioned chart models for v2.

``AudioChartModel`` (encoder-decoder):

    tick log-mel (Linear frontend, not Conv1D) -> RoPE audio encoder + learned pos_in_bar + cond
        -> memory
        -> causal chart decoder (learned position + cond + cross-attention)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from audio2map.audio.tick_features import V2_FEATURE_DIM
from audio2map.data.cond_vec import COND_VEC_DIM
from audio2map.osu.grid_config import TICKS_PER_BAR
from audio2map.osu.row_tokens import build_vocab
from audio2map.training.config import MAX_AUDIO_TICKS, MAX_DECODER_LEN, model_data_versions
from audio2map.training.rotary import RoPETransformerEncoder

_CHECKPOINT_META_KEYS = frozenset(
    {
        "architecture",
        "audio_pos_encoding",
        "tokenizer_version",
        "cond_vec_version",
        "cond_vec_dim",
        "audio_feature_spec_version",
        "ticks_per_bar",
        "window_bars",
    }
)


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool),
        diagonal=1,
    )


class AudioChartModel(nn.Module):
    """Tick-level audio encoder + autoregressive chart decoder."""

    def __init__(
        self,
        *,
        vocab_size: int | None = None,
        d_model: int = 256,
        n_heads: int = 4,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        dropout: float = 0.1,
        audio_dim: int = V2_FEATURE_DIM,
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
        self.architecture = "enc_dec"
        self.audio_pos_encoding = "rope+pos_in_bar"
        self.audio_frontend = "linear"

        # log-Mel[t] -> d_model (no Conv1D local frontend in v1 sanity-check config).
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

    @property
    def audio_pooling(self) -> str:
        return "tick"

    @property
    def max_seq_len(self) -> int:
        return self.max_decoder_len

    def _audio_bar_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Learned within-bar phase; requires bar-aligned window start (see window_sampler)."""
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
        src_key_padding_mask = ~audio_mask
        memory = self.audio_encoder(x, src_key_padding_mask=src_key_padding_mask)
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

        tgt_mask = _causal_mask(dec_len, token_ids.device)
        tgt_key_padding_mask = None if attn_mask is None else ~attn_mask[:, :-1]
        mem_key_padding_mask = None if audio_mask is None else ~audio_mask

        h = self.chart_decoder(
            dec_x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=mem_key_padding_mask,
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
        """Return logits for the next token after ``token_ids``.

        Appends a dummy column so ``forward()`` predicts one step ahead; the dummy
        is stripped by ``forward()`` via ``token_ids[:, :-1]`` and is **not** read
        as decoder input. The value ``0`` is not assumed to be ``<PAD>``.
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
                    torch.ones(token_ids.shape[0], 1, dtype=torch.bool, device=token_ids.device),
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

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, l, v = logits.shape
        logits = logits.reshape(b * l, v)
        targets = targets.reshape(b * l)
        mask = loss_mask.reshape(b * l)
        ce = F.cross_entropy(logits, targets, reduction="none")
        weighted = ce * mask
        if mask.sum() == 0:
            return weighted.sum()
        return weighted.sum() / mask.sum()

    def training_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self.forward(
            batch["audio"],
            batch["cond_vec"],
            batch["token_ids"],
            attn_mask=batch.get("attn_mask"),
            audio_mask=batch.get("audio_mask"),
        )
        targets = batch["token_ids"][:, 1:]
        mask = batch["loss_mask"][:, 1:]
        loss = self.compute_loss(logits, targets, mask)
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = ((pred == targets) & mask.bool()).float().sum() / mask.sum().clamp(min=1)
        audio_ticks = batch["audio"].shape[1]
        return loss, {
            "loss": float(loss.item()),
            "token_acc": float(acc.item()),
            "audio_ticks": float(audio_ticks),
            "encoder_audio_ticks": float(audio_ticks),
        }

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
        cfg = dict(ckpt.get("model_cfg", {}))
        arch = cfg.pop("architecture", "enc_dec")
        if arch != "enc_dec":
            raise ValueError(
                f"checkpoint {path} uses removed architecture {arch!r}; "
                "only enc_dec (AudioChartModel) is supported"
            )
        for key in _CHECKPOINT_META_KEYS | {"audio_frontend"}:
            cfg.pop(key, None)
        cfg.pop("audio_pooling", None)
        if "encoder_layers" not in cfg and "n_layers" in cfg:
            raise ValueError(
                f"checkpoint {path} looks like a removed prefix_lm model; "
                "re-train with AudioChartModel (enc_dec)"
            )
        cfg.pop("n_layers", None)
        dropout = cfg.pop("dropout", 0.1)
        model = cls(dropout=dropout, **cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model


def build_model(
    *,
    d_model: int = 256,
    n_heads: int = 4,
    layers: int | None = None,
    encoder_layers: int = 4,
    decoder_layers: int = 6,
    **kwargs,
) -> AudioChartModel:
    enc = layers if layers is not None else encoder_layers
    dec = layers if layers is not None else decoder_layers
    return AudioChartModel(
        d_model=d_model,
        n_heads=n_heads,
        encoder_layers=enc,
        decoder_layers=dec,
        **kwargs,
    )


def load_checkpoint(path: Path, device: torch.device) -> AudioChartModel:
    """Load an ``enc_dec`` checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("model_cfg", {}).get("architecture")
    if arch in (None, "prefix_lm"):
        raise ValueError(
            f"checkpoint {path} uses removed prefix_lm architecture; "
            "re-train with AudioChartModel (enc_dec)"
        )
    return AudioChartModel.load_checkpoint(path, device)
