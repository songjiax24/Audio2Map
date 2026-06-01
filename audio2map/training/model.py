"""Audio-conditioned chart models for v2.

Formal architecture (``AudioChartModel``, ``architecture=enc_dec``):

    tick audio -> non-causal Audio Encoder (+ time emb + cond)
        -> memory
        -> causal Chart Decoder (cross-attention)

Legacy ablation (``PrefixLMAudioChartModel``, ``architecture=prefix_lm``):

    [cond ; audio ; chart] -> causal TransformerEncoder (prefix-LM)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from audio2map.audio.tick_features import V2_FEATURE_DIM
from audio2map.data.cond_vec import COND_VEC_DIM
from audio2map.osu.grid_config import TICKS_PER_BAR, TICKS_PER_BEAT
from audio2map.osu.row_tokens import build_vocab

# Max window: 16 bars x 192 ticks (formal variable-window target).
MAX_AUDIO_TICKS = 16 * TICKS_PER_BAR
MAX_DECODER_LEN = 1024


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool),
        diagonal=1,
    )


class AudioChartModel(nn.Module):
    """Tick-level audio encoder + autoregressive chart decoder (formal v2)."""

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
        self.max_audio_ticks = max_audio_ticks
        self.max_decoder_len = max_decoder_len
        self.architecture = "enc_dec"

        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.audio_abs_pos = nn.Embedding(max_audio_ticks, d_model)
        self.pos_in_bar = nn.Embedding(TICKS_PER_BAR, d_model)
        self.pos_in_beat = nn.Embedding(TICKS_PER_BEAT, d_model)
        self.beat_in_bar = nn.Embedding(TICKS_PER_BAR // TICKS_PER_BEAT, d_model)

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.audio_encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)

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
        """Always tick-level for enc_dec (compat with logging / infer reports)."""
        return "tick"

    @property
    def max_seq_len(self) -> int:
        """Alias for inference/decode loops (``PrefixLMAudioChartModel`` uses the same name)."""
        return self.max_decoder_len

    def _audio_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """``t``: ``(T,)`` tick indices within window -> ``(T, d_model)``."""
        pos_bar = t % TICKS_PER_BAR
        pos_beat = t % TICKS_PER_BEAT
        beat_bar = pos_bar // TICKS_PER_BEAT
        emb = self.audio_abs_pos(t)
        emb = emb + self.pos_in_bar(pos_bar)
        emb = emb + self.pos_in_beat(pos_beat)
        emb = emb + self.beat_in_bar(beat_bar)
        return emb

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
        x = x + self._audio_time_embedding(tick_idx).unsqueeze(0)
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
        if mask.sum() == 0:
            return ce.mean()
        return (ce * mask).sum() / mask.sum()

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
            "architecture": 1.0,
        }

    def save_checkpoint(self, path: Path, *, extra: dict | None = None) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "model_cfg": {
                "architecture": "enc_dec",
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "n_heads": self.audio_encoder.layers[0].self_attn.num_heads,
                "encoder_layers": len(self.audio_encoder.layers),
                "decoder_layers": len(self.chart_decoder.layers),
                "max_audio_ticks": self.max_audio_ticks,
                "max_decoder_len": self.max_decoder_len,
            },
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
        cfg.pop("architecture", None)
        cfg.pop("audio_pooling", None)
        # Backward compat: old checkpoints used n_layers for a single stack.
        if "encoder_layers" not in cfg and "n_layers" in cfg:
            cfg["encoder_layers"] = cfg.pop("n_layers")
            cfg["decoder_layers"] = cfg.get("decoder_layers", cfg["encoder_layers"])
        cfg.pop("n_layers", None)
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model


class PrefixLMAudioChartModel(nn.Module):
    """Legacy prefix-LM ablation — not the formal v2 architecture."""

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
        max_seq_len: int = MAX_DECODER_LEN,
        audio_pooling: str = "tick",
    ) -> None:
        super().__init__()
        if audio_pooling not in ("tick", "bar"):
            raise ValueError(f"audio_pooling must be 'tick' or 'bar', got {audio_pooling!r}")
        vocab_size = vocab_size or len(build_vocab())
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.audio_pooling = audio_pooling
        self.max_seq_len = max_seq_len
        self.architecture = "prefix_lm"

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
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

    def encode_audio_prefix(self, audio: torch.Tensor, *, window_bars: int) -> torch.Tensor:
        if self.audio_pooling == "bar":
            b, t, _ = audio.shape
            expected = window_bars * TICKS_PER_BAR
            if t < expected:
                pad = audio.new_zeros(b, expected - t, audio.shape[-1])
                audio = torch.cat([audio, pad], dim=1)
            elif t > expected:
                audio = audio[:, :expected]
            bars = audio.view(b, window_bars, TICKS_PER_BAR, -1).mean(dim=2)
            return self.audio_proj(bars)
        return self.audio_proj(audio)

    @property
    def audio_stride_ticks(self) -> int:
        return TICKS_PER_BAR if self.audio_pooling == "bar" else 1

    def forward(
        self,
        audio: torch.Tensor,
        cond_vec: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del audio_mask  # prefix-LM does not use separate audio padding mask
        b, _ = token_ids.shape
        window_bars = max(1, audio.shape[1] // TICKS_PER_BAR)

        audio_h = self.encode_audio_prefix(audio, window_bars=window_bars)
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
            torch.triu(
                torch.ones(total_len, total_len, device=token_ids.device, dtype=torch.bool),
                diagonal=1,
            ),
            float("-inf"),
        )
        h = self.transformer(seq, mask=attn_bias, src_key_padding_mask=key_pad)
        h = h[:, prefix_len:]
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
        return AudioChartModel.compute_loss(logits, targets, loss_mask)

    def training_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self.forward(
            batch["audio"],
            batch["cond_vec"],
            batch["token_ids"],
            attn_mask=batch.get("attn_mask"),
        )
        targets = batch["token_ids"][:, 1:]
        mask = batch["loss_mask"][:, 1:]
        loss = self.compute_loss(logits, targets, mask)
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = ((pred == targets) & mask.bool()).float().sum() / mask.sum().clamp(min=1)
        audio_ticks = batch["audio"].shape[1]
        enc_len = 1 + (
            audio_ticks // TICKS_PER_BAR
            if self.audio_pooling == "bar"
            else audio_ticks
        )
        return loss, {
            "loss": float(loss.item()),
            "token_acc": float(acc.item()),
            "audio_ticks": float(audio_ticks),
            "encoder_prefix_len": float(enc_len),
            "architecture": 0.0,
        }

    def save_checkpoint(self, path: Path, *, extra: dict | None = None) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "model_cfg": {
                "architecture": "prefix_lm",
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "n_heads": self.transformer.layers[0].self_attn.num_heads,
                "n_layers": len(self.transformer.layers),
                "audio_pooling": self.audio_pooling,
            },
        }
        if extra:
            payload.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.part")
        torch.save(payload, tmp)
        tmp.replace(path)

    @classmethod
    def load_checkpoint(cls, path: Path, device: torch.device) -> PrefixLMAudioChartModel:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = dict(ckpt.get("model_cfg", {}))
        cfg.pop("architecture", None)
        if "audio_pooling" not in cfg:
            cfg["audio_pooling"] = "bar"
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model


def build_model(
    *,
    architecture: str = "enc_dec",
    d_model: int = 256,
    n_heads: int = 4,
    layers: int | None = None,
    encoder_layers: int = 4,
    decoder_layers: int = 6,
    audio_pooling: str = "tick",
    **kwargs,
) -> nn.Module:
    if architecture == "prefix_lm":
        n_layers = layers if layers is not None else encoder_layers
        return PrefixLMAudioChartModel(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            audio_pooling=audio_pooling,
            **kwargs,
        )
    if architecture != "enc_dec":
        raise ValueError(f"unknown architecture {architecture!r}")
    enc = layers if layers is not None else encoder_layers
    dec = layers if layers is not None else decoder_layers
    return AudioChartModel(
        d_model=d_model,
        n_heads=n_heads,
        encoder_layers=enc,
        decoder_layers=dec,
        **kwargs,
    )


def load_checkpoint(path: Path, device: torch.device) -> nn.Module:
    """Load ``enc_dec`` or legacy ``prefix_lm`` checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_cfg", {})
    arch = cfg.get("architecture")
    if arch is None:
        # Checkpoints before architecture field used prefix-LM.
        arch = "prefix_lm"
    if arch == "prefix_lm":
        return PrefixLMAudioChartModel.load_checkpoint(path, device)
    return AudioChartModel.load_checkpoint(path, device)
