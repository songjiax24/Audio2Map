"""Activation checkpoint recomputes each layer and matches the plain backward."""

from __future__ import annotations

import pytest
import torch

from audio2map.cli.train_v1 import build_parser
from audio2map.model.v1.batch import V1Batch
from audio2map.features.audio.v1.constants import N_MELS
from audio2map.model.v1.constants import D_MODEL
from audio2map.model.v1.loss import plain_ce
from audio2map.model.v1.model import V1ChartModel
from audio2map.tokens.v1.vocab import TOKEN_PAD, build_vocab
from audio2map.train.v1.loop import V1TrainConfig


def _batch(frames: int = 8, length: int = 4) -> V1Batch:
    vocab = build_vocab()
    tokens = torch.randint(0, len(vocab), (2, length))
    tokens[:, 0] = vocab["<BOS>"]
    tokens[:, -1] = vocab[TOKEN_PAD]
    return V1Batch(
        frame_lo=torch.tensor([0, 4]),
        cond_x=torch.rand(2, 18),
        cond_m=torch.ones(2, 18),
        token_ids=tokens,
        cursor_time_ms=torch.linspace(0.0, 40.0, length - 1).repeat(2, 1),
        phi=torch.rand(2, frames),
        beat_b=torch.zeros(2, frames),
        log_bpm=torch.zeros(2, frames),
        rel_u=torch.linspace(0.0, 1.0, frames).repeat(2, 1),
        mel=torch.randn(2, N_MELS, 2 * frames + 3),
        n_frames=torch.full((2,), frames, dtype=torch.long),
    )


def _open_gates(model: V1ChartModel) -> None:
    """AdaLN-Zero would otherwise multiply every residual branch by 0."""
    with torch.no_grad():
        for layer in model.decoder:
            for mod in (layer.sa_mod, layer.ca_mod, layer.ff_mod):
                mod.proj.bias[2 * D_MODEL :] = 0.1


def _watch(layer: torch.nn.Module) -> list[int]:
    seen = [0]
    original = layer.forward

    def wrapped(*args, **kwargs):
        seen[0] += 1
        return original(*args, **kwargs)

    layer.forward = wrapped
    return seen


def test_train_config_and_cli_default_on() -> None:
    assert V1TrainConfig().grad_checkpoint is True
    assert build_parser().parse_args([]).grad_checkpoint is True
    assert build_parser().parse_args(["--no-grad-checkpoint"]).grad_checkpoint is False
    assert V1ChartModel().grad_checkpoint is False


def test_checkpoint_matches_plain_backward_and_reruns_the_layer() -> None:
    torch.manual_seed(0)
    plain = V1ChartModel()
    _open_gates(plain)
    recomputed = V1ChartModel()
    recomputed.load_state_dict(plain.state_dict())
    recomputed.set_grad_checkpoint(True)
    batch = _batch()
    enc = _watch(recomputed.encoder.layers[0])
    dec = _watch(recomputed.decoder[0])

    def backward(model: V1ChartModel) -> torch.Tensor:
        model.train()
        logits = model(batch)
        loss = plain_ce(logits, batch.token_ids[:, 1:])
        loss.backward()
        return loss.detach()

    left = backward(plain)
    right = backward(recomputed)
    assert enc[0] == 2
    assert dec[0] == 2
    assert torch.allclose(left, right)
    pairs = (
        (
            plain.encoder.layers[0].attn.q_proj.weight,
            recomputed.encoder.layers[0].attn.q_proj.weight,
        ),
        (plain.decoder[0].self_attn.q_proj.weight, recomputed.decoder[0].self_attn.q_proj.weight),
        (plain.decoder[0].sa_mod.proj.weight, recomputed.decoder[0].sa_mod.proj.weight),
    )
    for src, dst in pairs:
        assert src.grad is not None and dst.grad is not None
        assert torch.allclose(src.grad, dst.grad, atol=1e-5, rtol=1e-4)


def test_eval_does_not_recompute() -> None:
    torch.manual_seed(1)
    model = V1ChartModel()
    model.set_grad_checkpoint(True)
    model.eval()
    seen = _watch(model.encoder.layers[0])
    with torch.no_grad():
        model(_batch())
    assert seen[0] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_flex_checkpoint_backward_is_finite() -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    model = V1ChartModel().to(device)
    _open_gates(model)
    model.train()
    model.set_grad_checkpoint(True)
    batch = _batch(frames=32, length=8)
    moved = V1Batch(
        frame_lo=batch.frame_lo.to(device),
        cond_x=batch.cond_x.to(device),
        cond_m=batch.cond_m.to(device),
        token_ids=batch.token_ids.to(device),
        cursor_time_ms=batch.cursor_time_ms.to(device),
        phi=batch.phi.to(device),
        beat_b=batch.beat_b.to(device),
        log_bpm=batch.log_bpm.to(device),
        rel_u=batch.rel_u.to(device),
        mel=batch.mel.to(device),
        n_frames=batch.n_frames.to(device),
    )
    seen = _watch(model.decoder[0])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(moved)
        loss = plain_ce(logits, moved.token_ids[:, 1:])
    loss.backward()
    assert model.last_cross is not None
    assert model.last_cross.impl == "flex"
    assert model.last_cross.compiled is True
    assert seen[0] == 2
    grad = model.decoder[0].cross_attn.q_proj.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
