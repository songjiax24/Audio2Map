"""Tests for formal encoder-decoder AudioChartModel."""

from __future__ import annotations

import torch

from audio2map.osu.grid_config import TICKS_PER_BAR
from audio2map.training.model import AudioChartModel, PrefixLMAudioChartModel, build_model, load_checkpoint


def test_enc_dec_forward_shapes() -> None:
    b, bars, feat = 2, 8, 142
    t_audio = bars * TICKS_PER_BAR
    seq = 64
    model = AudioChartModel(d_model=64, n_heads=4, encoder_layers=2, decoder_layers=2)
    audio = torch.randn(b, t_audio, feat)
    cond = torch.randn(b, 23)
    token_ids = torch.randint(0, model.vocab_size, (b, seq))
    attn = torch.ones(b, seq, dtype=torch.bool)
    audio_mask = torch.ones(b, t_audio, dtype=torch.bool)

    logits = model(audio, cond, token_ids, attn_mask=attn, audio_mask=audio_mask)
    assert logits.shape == (b, seq - 1, model.vocab_size)

    nxt = model.next_token_logits(audio[:1], cond[:1], token_ids[:1, :10])
    assert nxt.shape == (1, model.vocab_size)


def test_build_model_defaults_enc_dec() -> None:
    m = build_model(architecture="enc_dec", d_model=32, layers=2, n_heads=4)
    assert isinstance(m, AudioChartModel)
    assert m.architecture == "enc_dec"
    assert m.audio_pooling == "tick"


def test_prefix_lm_ablation_still_available() -> None:
    m = build_model(architecture="prefix_lm", d_model=32, layers=2, n_heads=4)
    assert isinstance(m, PrefixLMAudioChartModel)


def test_checkpoint_roundtrip_enc_dec(tmp_path) -> None:
    model = AudioChartModel(d_model=32, n_heads=4, encoder_layers=2, decoder_layers=2)
    path = tmp_path / "enc_dec.pt"
    model.save_checkpoint(path)
    loaded = load_checkpoint(path, torch.device("cpu"))
    assert isinstance(loaded, AudioChartModel)
    assert loaded.d_model == 32
