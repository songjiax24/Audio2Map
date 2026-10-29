"""§6 chart model. The encoder's forward takes normalized mel.

The conv stem is the front of ``AudioEncoder``. Phase is added to its output.
"""

from __future__ import annotations

import torch
from torch import nn

from audio2map.features.cond.v1.vec import COND_MLP_HIDDEN, COND_MLP_IN, COND_VEC_DIM
from audio2map.model.audio_stem import AudioStem
from audio2map.model.v1.batch import V1Batch
from audio2map.model.v1.blocks import DecoderLayer
from audio2map.model.v1.constants import D_MODEL, N_LAYERS, PHASE_DIM, PHI_HARMONICS
from audio2map.model.v1.encoder import AudioEncoder, run_layer
from audio2map.model.v1.cross import (
    CA_TRAIN_FLEX,
    CA_TRAIN_LEGACY,
    CrossLog,
    bucket_length,
    flex_compiled,
    normalize_ca_train,
    prepare_flex,
)
from audio2map.model.v1.windows import cross_allow, cross_positions
from audio2map.tokens.v1.vocab import build_vocab


def phase_fourier(
    phi: torch.Tensor,
    beat_b: torch.Tensor,
    log_bpm: torch.Tensor,
    rel_u: torch.Tensor,
) -> torch.Tensor:
    """φ k=1..8, ψ k=1, then log BPM and u. Last dim is 20.

    ``log_bpm`` and ``rel_u`` are already sampled at the same frame time as
    ``phi``. This layer does not look up a redline.
    """
    harmonics = torch.arange(1, PHI_HARMONICS + 1, device=phi.device, dtype=phi.dtype)
    angle = phi.unsqueeze(-1) * harmonics * (2.0 * torch.pi)
    phi_feat = torch.cat((angle.sin(), angle.cos()), dim=-1)
    psi = (beat_b.to(phi.dtype) + phi) / 2.0
    psi_angle = psi * (2.0 * torch.pi)
    psi_feat = torch.stack((psi_angle.sin(), psi_angle.cos()), dim=-1)
    extra = torch.stack(
        (log_bpm.to(dtype=phi.dtype), rel_u.to(dtype=phi.dtype)),
        dim=-1,
    )
    return torch.cat((phi_feat, psi_feat, extra), dim=-1)


class ConditionMLP(nn.Module):
    """``z = [x ⊙ m; m]`` then 36 → 256 → 256."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(COND_MLP_IN, COND_MLP_HIDDEN),
            nn.GELU(),
            nn.Linear(COND_MLP_HIDDEN, COND_MLP_HIDDEN),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((values * mask, mask), dim=-1)
        return self.net(joined)


class V1ChartModel(nn.Module):
    """8 encoder layers and 8 decoder layers. The LM head shares the token embedding.

    ``stem`` is the encoder's conv front. Pass one instance to share it.
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        stem: AudioStem | None = None,
        ca_train: str = CA_TRAIN_FLEX,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size if vocab_size is not None else len(build_vocab())
        self.encoder = AudioEncoder(stem)
        self.token_emb = nn.Embedding(self.vocab_size, D_MODEL)
        self.phase = nn.Linear(PHASE_DIM, D_MODEL)
        self.cond = ConditionMLP()
        self.decoder = nn.ModuleList(DecoderLayer(COND_MLP_HIDDEN) for _ in range(N_LAYERS))
        self.lm_head = nn.Linear(D_MODEL, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.ca_train = CA_TRAIN_FLEX
        # Off until the train loop turns it on. Eval never recomputes.
        self.grad_checkpoint = False
        self.last_cross: CrossLog | None = None
        self.cross_history: list[CrossLog] = []
        self._cross_buckets: set[tuple[int, int]] = set()
        self.set_ca_train(ca_train)

    def set_ca_train(self, mode: str) -> None:
        """``flex`` (default) or ``legacy`` dense bool mask. Not a checkpoint field."""
        self.ca_train = normalize_ca_train(mode)
        for layer in self.decoder:
            layer.cross_attn.train_mode = self.ca_train

    def set_grad_checkpoint(self, enabled: bool) -> None:
        """Recompute each encoder and decoder layer on the backward pass.

        The weight checkpoint does not store this. The run config does.
        """
        self.grad_checkpoint = bool(enabled)

    def _note_cross(self, log: CrossLog) -> None:
        self.last_cross = log
        self.cross_history.append(log)
        if len(self.cross_history) > 16:
            del self.cross_history[:-16]

    def forward(self, batch: V1Batch) -> torch.Tensor:
        """Logits ``(B, L-1, vocab)`` for ``token_ids[:, 1:]``."""
        if batch.mel is None or batch.n_frames is None:
            raise ValueError("trainable batch needs mel and n_frames")
        if batch.cond_x.shape[-1] != COND_VEC_DIM or batch.cond_m.shape[-1] != COND_VEC_DIM:
            raise ValueError("condition must be 18 values and 18 mask dims")
        tokens = batch.token_ids[:, :-1]
        if tokens.shape[1] != batch.cursor_time_ms.shape[1]:
            raise ValueError("cursor_time_ms must align with token_ids[:, :-1]")
        if int(tokens.max()) >= self.vocab_size:
            raise ValueError(f"token id {int(tokens.max())} >= vocab {self.vocab_size}")

        recompute = self.training and self.grad_checkpoint
        memory = self.encoder(
            batch.mel,
            batch.frame_lo,
            batch.audio_mask,
            n_frames=batch.n_frames,
            recompute=recompute,
        )
        device = memory.device
        batch_size, length, _ = memory.shape
        audio_mask = (
            torch.ones(batch_size, length, dtype=torch.bool, device=device)
            if batch.audio_mask is None
            else batch.audio_mask.to(device=device)
        )
        phi = batch.phi.to(device=device, dtype=memory.dtype)
        beat_b = batch.beat_b.to(device=device)
        context = memory + self.phase(
            phase_fourier(
                phi,
                beat_b,
                batch.log_bpm.to(device=device),
                batch.rel_u.to(device=device),
            )
        )

        g, p_q = cross_positions(batch.frame_lo, length, batch.cursor_time_ms.to(device=device))
        q_len = tokens.shape[1]
        positions = torch.arange(q_len, device=device)
        if batch.token_mask is not None:
            token_keep = batch.token_mask.to(device=device)
        else:
            token_keep = torch.ones(batch_size, q_len, dtype=torch.bool, device=device)
        if self.ca_train == CA_TRAIN_FLEX and context.is_cuda:
            bucket = (bucket_length(q_len), bucket_length(length))
            plan, log = prepare_flex(
                p_q,
                g,
                audio_mask,
                compiled=flex_compiled(context),
                bucket_hit=bucket in self._cross_buckets,
            )
            self._cross_buckets.add(bucket)
            cross_arg: torch.Tensor | object = plan
        elif self.ca_train == CA_TRAIN_FLEX:
            bucket = (bucket_length(q_len), bucket_length(length))
            hit = bucket in self._cross_buckets
            self._cross_buckets.add(bucket)
            cross_arg = cross_allow(g, p_q) & audio_mask[:, None, :]
            log = CrossLog(
                impl=CA_TRAIN_FLEX,
                compiled=False,
                q_len=q_len,
                kv_len=length,
                q_bucket=bucket[0],
                kv_bucket=bucket[1],
                bucket_hit=hit,
                fallback="cpu-dense",
            )
        elif self.ca_train == CA_TRAIN_LEGACY:
            cross_arg = cross_allow(g, p_q) & audio_mask[:, None, :]
            log = CrossLog(
                impl=CA_TRAIN_LEGACY,
                compiled=False,
                q_len=q_len,
                kv_len=length,
                q_bucket=q_len,
                kv_bucket=length,
                bucket_hit=False,
                fallback="",
            )
        else:
            raise ValueError(f"unknown ca_train {self.ca_train!r}")
        self._note_cross(log)

        hidden = self.token_emb(tokens.to(device=device))
        cond = self.cond(
            batch.cond_x.to(device=device, dtype=memory.dtype),
            batch.cond_m.to(device=device, dtype=memory.dtype),
        )
        k_pos = g
        recompute = self.training and self.grad_checkpoint
        for layer in self.decoder:
            hidden = run_layer(
                layer,
                hidden,
                cond,
                positions,
                token_keep,
                context,
                p_q,
                k_pos,
                cross_arg,
                recompute=recompute,
            )
        return self.lm_head(hidden)
