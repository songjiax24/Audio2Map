"""Cached window decode matches full next_token_logits and encodes once."""

from __future__ import annotations

import torch

from audio2map.model.model import AudioChartModel


def test_cached_logits_match_full_decode_and_encode_once() -> None:
    torch.manual_seed(0)
    model = AudioChartModel(
        d_model=64,
        n_heads=4,
        encoder_layers=2,
        decoder_layers=2,
        dropout=0.1,
    )
    model.eval()
    audio = torch.randn(1, 64, 128)
    cond = torch.randn(1, model.cond_dim)
    token_ids = torch.randint(0, model.vocab_size, (1, 4))

    encodes = {"n": 0}
    real_encode = model.encode_audio

    def counted(audio_in, cond_in, audio_mask=None):
        encodes["n"] += 1
        return real_encode(audio_in, cond_in, audio_mask=audio_mask)

    model.encode_audio = counted
    state = model.start_window_decode(audio, cond)
    assert encodes["n"] == 1

    diffs = []
    for _ in range(12):
        before = encodes["n"]
        with torch.no_grad():
            cached = model.window_next_logits(state, token_ids)
        assert encodes["n"] == before
        with torch.no_grad():
            full = model.next_token_logits(audio, cond, token_ids)
        diffs.append((cached - full).abs().max().item())
        assert cached.argmax(dim=-1).item() == full.argmax(dim=-1).item()
        nxt = int(cached.argmax(dim=-1).item())
        token_ids = torch.cat([token_ids, torch.tensor([[nxt]])], dim=1)

    assert encodes["n"] == 13
    assert max(diffs) < 1e-4
