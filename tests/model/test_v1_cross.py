"""Flex training cross-attention and gather decode, against the legacy dense mask."""

from __future__ import annotations

import logging
import warnings

import pytest
import torch

from audio2map.features.audio.v1.constants import DT_MS, N_MELS
from audio2map.features.cond.v1.vec import COND_VEC_DIM
from audio2map.model.v1.blocks import CrossAttention
from audio2map.model.v1.constants import D_MODEL
from audio2map.model.v1.cross import BUCKET, _mask_mod, bucket_length, prepare_flex, window_slots
from audio2map.model.v1.decode import _read_window, gather_cross, teacher_decode
from audio2map.model.v1.loss import plain_ce
from audio2map.model.v1.model import V1Batch, V1ChartModel
from audio2map.model.v1.windows import cross_allow, cross_positions
from audio2map.tokens.v1.vocab import build_vocab


def _batch(frames: int, steps: int, batch: int = 1, frame_lo: int = 0) -> V1Batch:
    vocab = build_vocab()
    tokens = torch.randint(1, len(vocab), (batch, steps + 1))
    tokens[:, 0] = vocab["<BOS>"]
    cursor = torch.linspace(0.0, (frames - 1) * 11.61, steps).unsqueeze(0).expand(batch, -1).contiguous()
    return V1Batch(
        frame_lo=torch.full((batch,), frame_lo, dtype=torch.long),
        cond_x=torch.rand(batch, COND_VEC_DIM),
        cond_m=torch.ones(batch, COND_VEC_DIM),
        token_ids=tokens,
        cursor_time_ms=cursor,
        phi=torch.rand(batch, frames),
        beat_b=torch.zeros(batch, frames),
        log_bpm=torch.zeros(batch, frames),
        rel_u=torch.zeros(batch, frames),
        mel=torch.randn(batch, N_MELS, 2 * frames + 3),
        n_frames=torch.full((batch,), frames, dtype=torch.long),
    )


def test_default_training_cross_is_flex() -> None:
    model = V1ChartModel()
    assert model.ca_train == "flex"
    assert model.decoder[0].cross_attn.train_mode == "flex"
    with pytest.raises(ValueError, match="ca_train"):
        model.set_ca_train("gather")


def test_window_slots_are_each_query_not_a_union() -> None:
    g = (40 + torch.arange(80)).to(torch.float32).unsqueeze(0)
    c_t = torch.tensor([[40.2, 90.0, 118.5]])
    audio_mask = torch.ones(1, 80, dtype=torch.bool)
    audio_mask[0, 12] = False
    index, valid = window_slots(c_t, g, audio_mask)
    allow = cross_allow(g, c_t) & audio_mask[:, None, :]
    for query in range(c_t.shape[1]):
        got = g[0, index[0, query][valid[0, query]]]
        ref = g[0, allow[0, query]]
        assert sorted(got.tolist()) == sorted(ref.tolist())
        assert int(valid[0, query].sum()) <= 512
    wide = torch.arange(900, dtype=torch.float32).unsqueeze(0)
    far = torch.tensor([[200.0, 700.0]])
    far_index, far_valid = window_slots(far, wide, torch.ones(1, 900, dtype=torch.bool))
    left = set(wide[0, far_index[0, 0][far_valid[0, 0]]].tolist())
    right = set(wide[0, far_index[0, 1][far_valid[0, 1]]].tolist())
    assert left != right
    assert len(left | right) > max(len(left), len(right))
    assert len(left) <= 512 and len(right) <= 512


def test_flex_matches_legacy_and_records_variable_lengths() -> None:
    torch.manual_seed(0)
    model = V1ChartModel()
    short = _batch(40, 6)
    longer = _batch(200, 20)
    model.set_ca_train("legacy")
    ref_short = model(short)
    ref_long = model(longer)
    model.set_ca_train("flex")
    got_short = model(short)
    got_long = model(longer)
    assert (got_short - ref_short).abs().max() < 1e-4
    assert (got_long - ref_long).abs().max() < 1e-4
    flex_logs = [item for item in model.cross_history if item.impl == "flex"]
    assert [item.q_len for item in flex_logs] == [6, 20]
    assert [item.kv_len for item in flex_logs] == [40, 200]
    assert flex_logs[0].q_bucket == bucket_length(6) == 128
    assert flex_logs[0].kv_bucket == 128
    assert flex_logs[1].kv_bucket == bucket_length(200) == 256
    assert flex_logs[0].bucket_hit is False
    assert flex_logs[1].bucket_hit is False
    assert flex_logs[0].compiled is False
    assert flex_logs[0].fallback == "cpu-dense"
    again = model(short)
    assert torch.allclose(again, got_short)
    assert model.last_cross is not None
    assert model.last_cross.bucket_hit is True
    assert model.last_cross.q_bucket == 128 and model.last_cross.kv_bucket == 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flex_matches_legacy_with_a_visibility_hole() -> None:
    warnings.filterwarnings("ignore", message="flex_attention called without torch.compile.*")
    torch.manual_seed(1)
    device = torch.device("cuda")
    frames, queries = 64, 8
    memory = torch.randn(1, frames, D_MODEL, device=device)
    query = torch.randn(1, queries, D_MODEL, device=device)
    g = torch.arange(frames, device=device).to(torch.float32).unsqueeze(0)
    c_t = torch.linspace(0.0, frames - 1.0, queries, device=device).unsqueeze(0)
    audio_mask = torch.ones(1, frames, dtype=torch.bool, device=device)
    audio_mask[0, 5] = False
    audio_mask[0, 40] = False
    allow = cross_allow(g, c_t) & audio_mask[:, None, :]
    dense = CrossAttention().to(device)
    dense.train_mode = "legacy"
    flex = CrossAttention().to(device)
    flex.load_state_dict(dense.state_dict())
    assert flex.train_mode == "flex"
    from audio2map.model.v1.cross import prepare_flex

    plan, _log = prepare_flex(c_t, g, audio_mask, compiled=False, bucket_hit=False)
    left = query.detach().requires_grad_(True)
    right = query.detach().requires_grad_(True)
    mem_l = memory.detach().requires_grad_(True)
    mem_r = memory.detach().requires_grad_(True)
    got = flex(left, mem_l, c_t, g, plan)
    ref = dense(right, mem_r, c_t, g, allow)
    assert (got - ref).abs().max() < 1e-4
    got.sum().backward()
    ref.sum().backward()
    assert (left.grad - right.grad).abs().max() < 1e-4
    assert (mem_l.grad - mem_r.grad).abs().max() < 1e-4
    assert (flex.k_proj.weight.grad - dense.k_proj.weight.grad).abs().max() < 1e-4


def _window_blocks(plan) -> torch.Tensor:
    batch = plan.floor_c.shape[0]
    queries, keys = plan.q_bucket, plan.kv_bucket
    b = torch.arange(batch)[:, None, None]
    q = torch.arange(queries)[None, :, None]
    k = torch.arange(keys)[None, None, :]
    elements = _mask_mod(b, 0, q, k)
    q_blocks, k_blocks = queries // BUCKET, keys // BUCKET
    reduced = elements.view(batch, q_blocks, BUCKET, k_blocks, BUCKET)
    return reduced.any(dim=(2, 4))[:, None]


def test_block_mask_is_the_tiles_the_window_meets() -> None:
    frames, queries = 700, 40
    origin = torch.tensor([[40], [7]], dtype=torch.long)
    g = origin + torch.arange(frames)
    c_t = torch.linspace(40.0, 40.0 + frames - 1, queries).unsqueeze(0).repeat(2, 1).clone()
    c_t[0, 0] = 295.0
    c_t[0, 3] = -50.0
    c_t[1] = torch.linspace(7.0 + frames - 1, 7.0, queries)
    audio = torch.ones(2, frames, dtype=torch.bool)
    audio[:, 80:90] = False
    audio[1, 400:] = False
    plan, _log = prepare_flex(c_t, g, audio, compiled=False, bucket_hit=False)
    got = plan.block.to_dense().to(dtype=torch.bool)
    assert plan.block.full_kv_num_blocks is None
    assert torch.equal(got, _window_blocks(plan))
    flipped = c_t.flip(1).contiguous()
    flipped[1, 1] = -20.0
    again, _log = prepare_flex(flipped, g, audio, compiled=False, bucket_hit=True)
    assert again.block.full_kv_num_blocks is None
    assert torch.equal(again.block.to_dense().to(dtype=torch.bool), _window_blocks(again))


def test_contiguous_cross_window_is_a_slice() -> None:
    cache = torch.randn(1, 2, 32, 4)
    span = torch.arange(4, 12)
    sliced = _read_window(cache, 0, span)
    assert sliced.shape == (2, 8, 4)
    assert sliced.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert torch.equal(sliced, cache[0, :, 4:12])
    holed = torch.tensor([4, 5, 7, 8])
    gathered = _read_window(cache, 0, holed)
    assert gathered.untyped_storage().data_ptr() != cache.untyped_storage().data_ptr()
    assert torch.equal(gathered, cache[0, :, holed])


def test_long_frame_index_stays_exact() -> None:
    frames = 11060
    frame_lo = torch.tensor([-17])
    cursor = torch.tensor([[0.0, 2048.0, 8000.0, 10000.0, 11000.0]]) * DT_MS
    g, c_t = cross_positions(frame_lo, frames, cursor)
    assert g.dtype == torch.long
    assert c_t.dtype == torch.float32
    assert int(g.unique().numel()) == frames
    allow = cross_allow(g, c_t)
    index, valid = window_slots(c_t, g, torch.ones(1, frames, dtype=torch.bool))
    for query in range(c_t.shape[1]):
        got = set(g[0, index[0, query][valid[0, query]]].tolist())
        ref = set(g[0, allow[0, query]].tolist())
        assert got == ref
    with pytest.raises(TypeError, match="bfloat16"):
        cross_allow(g.to(torch.bfloat16), c_t)


def _decode_against_dense(c_t: torch.Tensor, audio_mask: torch.Tensor) -> None:
    torch.manual_seed(0)
    frames = audio_mask.shape[1]
    attn = CrossAttention()
    memory = torch.randn(1, frames, D_MODEL)
    query = torch.randn(1, c_t.shape[1], D_MODEL)
    g = torch.arange(frames).unsqueeze(0)
    key = attn.k_proj(memory).view(1, frames, attn.n_heads, attn.head_dim).transpose(1, 2)
    value = attn.v_proj(memory).view(1, frames, attn.n_heads, attn.head_dim).transpose(1, 2)
    key = attn.rotary.apply(key, g)
    got, _codes = gather_cross(attn, query, c_t, g, audio_mask, key, value)
    allow = cross_allow(g, c_t) & audio_mask[:, None, :]
    ref = attn._dense(query, memory, c_t, g, allow)
    assert (got - ref).abs().max() < 1e-5


def test_decode_empty_prefix_and_hole_match_legacy() -> None:
    frames = 80
    full = torch.ones(1, frames, dtype=torch.bool)
    _decode_against_dense(torch.tensor([[-40.0, 10.2, 70.0, 400.0]]), full)
    prefix = full.clone()
    prefix[0, 40:] = False
    _decode_against_dense(torch.tensor([[50.0]]), prefix)
    hole = full.clone()
    hole[0, 12] = False
    hole[0, 15] = False
    _decode_against_dense(torch.tensor([[20.0, 30.0]]), hole)


def test_cpu_decode_matches_legacy(caplog: pytest.LogCaptureFixture) -> None:
    torch.manual_seed(2)
    model = V1ChartModel().eval()
    batch = _batch(48, 4, batch=2, frame_lo=3)
    model.set_ca_train("legacy")
    with torch.no_grad():
        reference = model(batch)
    with caplog.at_level(logging.INFO, logger="audio2map.model.v1.decode"):
        decoded = teacher_decode(model, batch)
    assert "cross backend=" in caplog.text
    assert decoded.backend in caplog.text
    assert (decoded.logits - reference).abs().max() < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bf16_training_window_matches_the_integer_index(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = 11060
    queries = torch.tensor([0.0, 2048.0, 8000.0, 10000.0, 11000.0])
    batch = _batch(frames, queries.numel(), frame_lo=-17)
    batch = V1Batch(
        frame_lo=batch.frame_lo,
        cond_x=batch.cond_x,
        cond_m=batch.cond_m,
        token_ids=batch.token_ids,
        cursor_time_ms=queries.unsqueeze(0) * DT_MS,
        phi=batch.phi,
        beat_b=batch.beat_b,
        log_bpm=batch.log_bpm,
        rel_u=batch.rel_u,
        mel=batch.mel,
        n_frames=batch.n_frames,
    )
    device = torch.device("cuda")
    batch = V1Batch(
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
    seen: dict[str, torch.Tensor] = {}
    import audio2map.model.v1.cross as cross_mod
    import audio2map.model.v1.model as model_mod

    real = model_mod.prepare_flex

    def _capture(*args, **kwargs):
        plan, log = real(*args, **kwargs)
        c_t, g = args[0], args[1]
        query = torch.arange(c_t.shape[1], device=c_t.device)[:, None]
        key = torch.arange(g.shape[1], device=g.device)[None, :]
        seen["c"] = c_t.detach()
        seen["g"] = g.detach()
        seen["mask"] = cross_mod._mask_mod(0, 0, query, key).detach()
        return plan, log

    monkeypatch.setattr(model_mod, "prepare_flex", _capture)
    model = V1ChartModel().to(device).train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(batch)
    assert model.last_cross is not None
    assert model.last_cross.impl == "flex"
    assert model.last_cross.compiled is True
    g = seen["g"]
    c_t = seen["c"]
    assert g.dtype == torch.long
    assert c_t.dtype == torch.float32
    assert int(g.unique().numel()) == frames
    allow = cross_allow(g, c_t)
    assert torch.equal(seen["mask"], allow[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_train_step_uses_flex_and_decode_is_flash(caplog: pytest.LogCaptureFixture) -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    model = V1ChartModel().to(device)
    batch = _batch(64, 8)
    batch = V1Batch(
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
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(batch)
        loss = plain_ce(logits, batch.token_ids[:, 1:])
    loss.backward()
    assert model.last_cross is not None
    assert model.last_cross.impl == "flex"
    assert model.last_cross.compiled is True
    assert model.decoder[0].cross_attn.q_proj.weight.grad is not None
    assert torch.isfinite(model.decoder[0].cross_attn.q_proj.weight.grad).all()

    model.eval()
    model.set_ca_train("legacy")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model(batch)
        model.set_ca_train("flex")
        flex_logits = model(batch)
        with caplog.at_level(logging.INFO, logger="audio2map.model.v1.decode"):
            decoded = teacher_decode(model, batch)
    assert "cross backend=FLASH" in caplog.text
    assert decoded.backend == "FLASH"
    flex_gap = (flex_logits.float() - reference.float()).abs().max().item()
    decode_gap = (decoded.logits.float() - reference.float()).abs().max().item()
    assert flex_gap < 2e-2, flex_gap
    assert decode_gap < 2e-2, decode_gap
