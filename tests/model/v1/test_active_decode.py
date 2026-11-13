"""Cached v1 decoding agrees with full logits after the AdaLN gates become active."""

import pytest
import torch

from audio2map.features.audio.v1.constants import N_MELS
from audio2map.features.cond.v1.vec import COND_VEC_DIM
from audio2map.model.v1.batch import V1Batch
from audio2map.model.v1.constants import D_MODEL
from audio2map.model.v1.decode import project_cross_cache
from audio2map.model.v1.kv import SelfKV, cached_logits
from audio2map.model.v1.model import V1ChartModel, phase_fourier
from audio2map.model.v1.windows import cross_positions
from audio2map.tokens.v1.vocab import build_vocab


@pytest.mark.parametrize("position_offset", [0, 2048])
def test_full_and_cached_logits_match_with_nonzero_gates_and_fractional_cursor(
    monkeypatch, position_offset,
):
    torch.manual_seed(3)
    model = V1ChartModel().eval()
    model.set_ca_train("legacy")
    with torch.no_grad():
        for layer in model.decoder:
            for modulation in (layer.sa_mod, layer.ca_mod, layer.ff_mod):
                modulation.proj.bias[2 * D_MODEL :].fill_(0.1)
    frames = 8
    vocab = build_vocab()
    batch = V1Batch(
        frame_lo=torch.tensor([-2]),
        cond_x=torch.ones(1, COND_VEC_DIM),
        cond_m=torch.ones(1, COND_VEC_DIM),
        token_ids=torch.tensor(
            [[vocab[name] for name in ("<BOS>", "<POS_0/1>", "<ROW_1000>", "<BEAT>", "<EOS>")]]
        ),
        cursor_time_ms=torch.tensor([[-10.0, 0.0, 15.5, 30.0]]),
        phi=torch.rand(1, frames),
        beat_b=torch.zeros(1, frames),
        log_bpm=torch.zeros(1, frames),
        rel_u=torch.linspace(0, 1, frames).unsqueeze(0),
        mel=torch.randn(1, N_MELS, 2 * frames + 3),
        n_frames=torch.tensor([frames]),
    )
    mask = torch.ones(1, frames, dtype=torch.bool)
    with torch.no_grad():
        full = model(batch)
        memory = model.encoder(batch.mel, batch.frame_lo, mask, n_frames=batch.n_frames)
        context = memory + model.phase(
            phase_fourier(batch.phi, batch.beat_b, batch.log_bpm, batch.rel_u)
        )
        g, query = cross_positions(batch.frame_lo, frames, batch.cursor_time_ms)
        cross = project_cross_cache(model, context, g)
        cond = model.cond(batch.cond_x, batch.cond_m)
        kv = SelfKV(len(model.decoder), position_offset=position_offset)
        positions_seen = []
        rotary = model.decoder[0].self_attn.rotary
        apply = rotary.apply

        def record_positions(tensor, positions):
            positions_seen.extend(positions.tolist())
            return apply(tensor, positions)

        monkeypatch.setattr(rotary, "apply", record_positions)
        cached = torch.cat(
            [
                cached_logits(
                    model,
                    batch.token_ids[:, index : index + 1],
                    index,
                    kv,
                    cond,
                    query[:, index : index + 1],
                    cross=cross,
                    g=g,
                    audio_mask=mask,
                )
                for index in range(4)
            ],
            dim=1,
        )
        copied_input = model.lm_head(model.token_emb(batch.token_ids[:, :-1]))
    assert not torch.allclose(full, copied_input)
    assert torch.allclose(full, cached, atol=1e-4, rtol=1e-5)
    assert positions_seen == [position_offset + i for i in range(4) for _ in range(2)]
