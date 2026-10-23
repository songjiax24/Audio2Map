"""§6 chart model. Training stems mel inside this module.

``precomputed=True`` skips the stem and reads detached rows. That path does
not train ``AudioStem``.
"""

from __future__ import annotations

import torch
from torch import nn

from audio2map.features.audio.v1.constants import DT_MS
from audio2map.features.cond.v1.vec import COND_MLP_HIDDEN, COND_MLP_IN, COND_VEC_DIM
from audio2map.model.audio_stem import STEM_CHANNELS, AudioStem
from audio2map.model.v1.batch import V1Batch
from audio2map.model.v1.blocks import DecoderLayer, EncoderLayer
from audio2map.model.v1.constants import D_MODEL, N_LAYERS, PHASE_DIM, PHI_HARMONICS
from audio2map.model.v1.windows import cross_allow, decoder_allow, encoder_allow
from audio2map.tokens.v1.vocab import build_vocab


def phase_fourier(phi: torch.Tensor, beat_b: torch.Tensor) -> torch.Tensor:
    """φ harmonics k=1..8 and ψ = (b + φ) / 2 at k=1. Last dim is 18."""
    harmonics = torch.arange(1, PHI_HARMONICS + 1, device=phi.device, dtype=phi.dtype)
    angle = phi.unsqueeze(-1) * harmonics * (2.0 * torch.pi)
    phi_feat = torch.cat((angle.sin(), angle.cos()), dim=-1)
    psi = (beat_b.to(phi.dtype) + phi) / 2.0
    psi_angle = psi * (2.0 * torch.pi)
    psi_feat = torch.stack((psi_angle.sin(), psi_angle.cos()), dim=-1)
    return torch.cat((phi_feat, psi_feat), dim=-1)


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

    ``stem`` is a submodule. Pass one instance to share it. Its weights train
    when the batch carries mel.
    """

    def __init__(self, vocab_size: int | None = None, stem: AudioStem | None = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size if vocab_size is not None else len(build_vocab())
        self.stem = stem if stem is not None else AudioStem()
        self.token_emb = nn.Embedding(self.vocab_size, D_MODEL)
        self.encoder = nn.ModuleList(EncoderLayer() for _ in range(N_LAYERS))
        self.phase = nn.Linear(PHASE_DIM, D_MODEL)
        self.cond = ConditionMLP()
        self.decoder = nn.ModuleList(DecoderLayer(COND_MLP_HIDDEN) for _ in range(N_LAYERS))
        self.lm_head = nn.Linear(D_MODEL, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def encode(
        self,
        audio: torch.Tensor,
        frame_lo: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> torch.Tensor:
        _batch, length, channels = audio.shape
        if channels != D_MODEL:
            raise ValueError(f"stem width {channels} != {D_MODEL}")
        local = torch.arange(length, device=audio.device)
        positions = frame_lo.to(device=audio.device).unsqueeze(1) + local
        allow = encoder_allow(length, audio.device).unsqueeze(0)
        allow = allow & audio_mask[:, None, :]
        hidden = audio
        for layer in self.encoder:
            hidden = layer(hidden, positions, allow)
        return hidden

    def _audio_rows(self, batch: V1Batch) -> torch.Tensor:
        """Stem mel without detaching, or return an explicitly precomputed cache."""
        if batch.precomputed:
            if batch.audio is None:
                raise ValueError("precomputed batch is missing stem rows")
            return batch.audio
        if batch.audio is not None and batch.mel is None:
            raise ValueError("stem rows require precomputed=True")
        if batch.mel is None or batch.n_frames is None:
            raise ValueError("trainable batch needs mel and n_frames")
        device = self.stem.conv1.weight.device
        mel = batch.mel.to(device=device)
        counts = batch.n_frames.to(device=device)
        width = int(counts.max())
        parts: list[torch.Tensor] = []
        for index in range(mel.shape[0]):
            count = int(counts[index])
            if count <= 0:
                raise ValueError("stem span is empty")
            produced = self.stem(mel[index : index + 1, :, : 2 * count + 3])
            if produced.shape[1] != count:
                raise ValueError(f"stem length {produced.shape[1]} != {count}")
            if count < width:
                pad = produced.new_zeros(1, width - count, STEM_CHANNELS)
                produced = torch.cat((produced, pad), dim=1)
            parts.append(produced)
        return torch.cat(parts, dim=0)

    def forward(self, batch: V1Batch) -> torch.Tensor:
        """Logits ``(B, L-1, vocab)`` for ``token_ids[:, 1:]``."""
        audio = self._audio_rows(batch)
        if audio.shape[-1] != D_MODEL:
            raise ValueError(f"stem width {audio.shape[-1]} != {D_MODEL}")
        if batch.cond_x.shape[-1] != COND_VEC_DIM or batch.cond_m.shape[-1] != COND_VEC_DIM:
            raise ValueError("condition must be 18 values and 18 mask dims")
        tokens = batch.token_ids[:, :-1]
        if tokens.shape[1] != batch.cursor_time_ms.shape[1]:
            raise ValueError("cursor_time_ms must align with token_ids[:, :-1]")
        if int(tokens.max()) >= self.vocab_size:
            raise ValueError(f"token id {int(tokens.max())} >= vocab {self.vocab_size}")

        batch_size, length, _ = audio.shape
        device = audio.device
        if batch.audio_mask is None:
            audio_mask = torch.ones(batch_size, length, dtype=torch.bool, device=device)
        else:
            audio_mask = batch.audio_mask.to(device=device)
        memory = self.encode(audio, batch.frame_lo.to(device=device), audio_mask)
        phi = batch.phi.to(device=device, dtype=audio.dtype)
        beat_b = batch.beat_b.to(device=device)
        context = memory + self.phase(phase_fourier(phi, beat_b))

        g = batch.frame_lo.to(device=device).unsqueeze(1) + torch.arange(length, device=device)
        g = g.to(dtype=context.dtype)
        q_len = tokens.shape[1]
        positions = torch.arange(q_len, device=device)
        self_allow = decoder_allow(q_len, device).unsqueeze(0).expand(batch_size, q_len, q_len)
        if batch.token_mask is not None:
            self_allow = self_allow & batch.token_mask.to(device=device)[:, None, :]
        p_q = batch.cursor_time_ms.to(device=device, dtype=context.dtype) / DT_MS
        cross_ok = cross_allow(g, p_q) & audio_mask[:, None, :]

        hidden = self.token_emb(tokens.to(device=device))
        cond = self.cond(
            batch.cond_x.to(device=device, dtype=audio.dtype),
            batch.cond_m.to(device=device, dtype=audio.dtype),
        )
        k_pos = g
        for layer in self.decoder:
            hidden = layer(hidden, cond, positions, self_allow, context, p_q, k_pos, cross_ok)
        return self.lm_head(hidden)
