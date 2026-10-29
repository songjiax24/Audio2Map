"""Formal KV steps match one full forward. The prefix is not concatenated."""

from __future__ import annotations

import pytest
import torch

from audio2map.features.audio.v1.constants import N_MELS
from audio2map.features.cond.v1.vec import COND_VEC_DIM
from audio2map.model.v1.constants import CROSS_FUTURE, CROSS_PAST, D_MODEL
from audio2map.model.v1.constants import DEC_SPAN
from audio2map.model.v1.decode import CrossWindow, project_cross_cache
from audio2map.model.v1.kv import SelfKV, WindowStep, cached_logits
from audio2map.model.v1.model import V1Batch, V1ChartModel, phase_fourier
from audio2map.model.v1.windows import cross_positions
from audio2map.tokens.v1.vocab import build_vocab


def _batch(frames: int, steps: int) -> V1Batch:
    vocab = build_vocab()
    tokens = torch.randint(1, len(vocab), (1, steps + 1))
    tokens[:, 0] = vocab["<BOS>"]
    cursor = torch.linspace(0.0, (frames - 1) * 11.61, steps).unsqueeze(0)
    return V1Batch(
        frame_lo=torch.zeros(1, dtype=torch.long),
        cond_x=torch.rand(1, COND_VEC_DIM),
        cond_m=torch.ones(1, COND_VEC_DIM),
        token_ids=tokens,
        cursor_time_ms=cursor,
        phi=torch.rand(1, frames),
        beat_b=torch.zeros(1, frames),
        log_bpm=torch.zeros(1, frames),
        rel_u=torch.linspace(0.0, 1.0, frames).unsqueeze(0),
        mel=torch.randn(1, N_MELS, 2 * frames + 3),
        n_frames=torch.tensor([frames], dtype=torch.long),
    )


def test_cached_steps_match_the_full_forward() -> None:
    torch.manual_seed(0)
    frames, steps = 16, 5
    model = V1ChartModel().eval()
    model.set_ca_train("legacy")
    batch = _batch(frames, steps)
    with torch.no_grad():
        reference = model(batch)
        mask = torch.ones(1, frames, dtype=torch.bool)
        memory = model.encoder(batch.mel, batch.frame_lo, mask, n_frames=batch.n_frames)
        context = memory + model.phase(
            phase_fourier(batch.phi, batch.beat_b, batch.log_bpm, batch.rel_u)
        )
        g, p_q = cross_positions(batch.frame_lo, frames, batch.cursor_time_ms)
        cross = project_cross_cache(model, context, g)
        cond = model.cond(batch.cond_x, batch.cond_m)
        kv = SelfKV(len(model.decoder))
        rows = []
        for index in range(steps):
            rows.append(
                cached_logits(
                    model,
                    batch.token_ids[:, index : index + 1],
                    index,
                    kv,
                    cond,
                    p_q[:, index : index + 1],
                    cross=cross,
                    g=g,
                    audio_mask=mask,
                )
            )
        got = torch.cat(rows, dim=1)
    assert kv.length == steps
    assert (got - reference).abs().max() < 1e-4


def test_self_kv_ring_keeps_the_decoder_window() -> None:
    kv = SelfKV(1)
    held = None
    for position in range(DEC_SPAN + 80):
        key = torch.full((1, 2, 1, 4), float(position))
        value = torch.full((1, 2, 1, 4), float(-position))
        kv.write(0, key, value)
        got_k, got_v = kv.window(0, position)
        start = 0 if position < DEC_SPAN else position + 1 - DEC_SPAN
        expect = torch.arange(start, position + 1, dtype=torch.float32)
        assert got_k.shape[2] == expect.shape[0]
        assert torch.equal(got_k[0, 0, :, 0], expect)
        assert torch.equal(got_v[0, 0, :, 0], -expect)
        if position == DEC_SPAN - 1:
            held = got_k
        kv.commit()
    assert kv.keys[0] is not None
    assert kv.keys[0].shape[2] == DEC_SPAN
    assert held is not None
    assert float(held[0, 0, 0, 0]) == 0.0


def test_sliding_cross_matches_the_full_cache() -> None:
    torch.manual_seed(1)
    frames, steps = 40, 6
    model = V1ChartModel().eval()
    batch = _batch(frames, steps)
    with torch.no_grad():
        mask = torch.ones(1, frames, dtype=torch.bool)
        memory = model.encoder(batch.mel, batch.frame_lo, mask, n_frames=batch.n_frames)
        context = memory + model.phase(
            phase_fourier(batch.phi, batch.beat_b, batch.log_bpm, batch.rel_u)
        )
        g, p_q = cross_positions(batch.frame_lo, frames, batch.cursor_time_ms)
        cross = project_cross_cache(model, context, g)
        cond = model.cond(batch.cond_x, batch.cond_m)
        full = SelfKV(len(model.decoder))
        slide = SelfKV(len(model.decoder))
        window = CrossWindow(int(g[0, 0]), frames)
        for index in range(steps):
            cursor = p_q[:, index : index + 1]
            floor_c = int(torch.floor(cursor[0, 0]).item())
            window.prepare(model, context, floor_c)
            token = batch.token_ids[:, index : index + 1]
            full_logits = cached_logits(
                model, token, index, full, cond, cursor, cross=cross, g=g, audio_mask=mask
            )
            slide_logits = cached_logits(
                model, token, index, slide, cond, cursor, window=window
            )
            assert (full_logits - slide_logits).abs().max() < 1e-5
        again = window.keys[0].data_ptr()
        window.prepare(model, context, floor_c)
        assert window.keys[0].data_ptr() == again


def test_full_cross_span_reuses_storage() -> None:
    torch.manual_seed(2)
    frames = 800
    model = V1ChartModel().eval()
    context = torch.randn(1, frames, D_MODEL)
    window = CrossWindow(0, frames)
    with torch.no_grad():
        window.prepare(model, context, 200)
        held = window.keys[0].data_ptr()
        before = window.keys[0].clone()
        window.prepare(model, context, 210)
        assert window.keys[0].data_ptr() == held
        assert not torch.equal(window.keys[0], before)
        fresh = CrossWindow(0, frames)
        fresh.prepare(model, context, 210)
        assert torch.equal(window.keys[0], fresh.keys[0])
        window.prepare(model, context, 0)
        assert window.rel_hi - window.rel_lo < 512
        window.prepare(model, context, 210)
        assert window.keys[0].data_ptr() == held


def _slice_keys(model: V1ChartModel, context: torch.Tensor, frame_lo: int, floor: int):
    frames = context.shape[1]
    abs_lo = max(frame_lo, floor - CROSS_PAST)
    abs_hi = min(frame_lo + frames, floor + CROSS_FUTURE)
    rel_lo = abs_lo - frame_lo
    span = context[:, rel_lo : rel_lo + (abs_hi - abs_lo), :]
    positions = torch.arange(abs_lo, abs_hi, device=context.device)
    keys = []
    for layer in model.decoder:
        attn = layer.cross_attn
        width = span.shape[1]
        flat = attn.k_proj(span).view(1, width, attn.n_heads, attn.head_dim)
        keys.append(attn.rotary.apply(flat.transpose(1, 2), positions))
    return keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graphed_cross_prepare_matches_a_slice() -> None:
    torch.manual_seed(2)
    frames = 800
    frame_lo = -25
    device = torch.device("cuda")
    model = V1ChartModel().to(device).eval()
    context = torch.randn(1, frames, D_MODEL, device=device)
    window = CrossWindow(frame_lo, frames)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        window.prepare(model, context, 200)
        held = window.keys[0].data_ptr()
        for floor in (200, 210, 300, 0, 210):
            window.prepare(model, context, floor)
            width = window.rel_hi - window.rel_lo
            if width != CROSS_PAST + CROSS_FUTURE:
                assert floor == 0
                continue
            assert window.keys[0].data_ptr() == held
            ref = _slice_keys(model, context, frame_lo, floor)
            pairs = zip(window.keys, ref, strict=True)
            gap = max(float((got - expect).abs().max()) for got, expect in pairs)
            assert gap == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_window_step_matches_cached_logits() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    frames = 4096
    model = V1ChartModel().to(device).eval()
    context = torch.randn(1, frames, D_MODEL, device=device)
    zeros = torch.zeros(1, COND_VEC_DIM, device=device)
    cond = model.cond(zeros, zeros)
    eager_window = CrossWindow(0, frames)
    graph_window = CrossWindow(0, frames)
    eager_kv = SelfKV(len(model.decoder))
    graph_kv = SelfKV(len(model.decoder))
    token = torch.tensor([[1]], device=device)
    cursor = torch.tensor([[2000.0]], device=device)
    cases = ((3, 2000.0), (7, 2000.4), (9, 2010.0), (4, 1800.0))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager_window.prepare(model, context, 2000)
        graph_window.prepare(model, context, 2000)
        for index in range(DEC_SPAN):
            cached_logits(model, token, index, eager_kv, cond, cursor, window=eager_window)
        graph_kv.keys = [key.clone() for key in eager_kv.keys]
        graph_kv.values = [value.clone() for value in eager_kv.values]
        graph_kv.length = eager_kv.length
        stepper = WindowStep(model, graph_kv, graph_window, cond)
        for step, (token_id, time_c) in enumerate(cases):
            position = DEC_SPAN + step
            floor_c = int(time_c)
            eager_window.prepare(model, context, floor_c)
            graph_window.prepare(model, context, floor_c)
            assert stepper.covers(position)
            step_token = torch.tensor([[token_id]], device=device)
            step_c = torch.tensor([[time_c]], device=device)
            got = stepper.run(step_token, position, step_c).float().clone()
            ref = cached_logits(
                model, step_token, position, eager_kv, cond, step_c, window=eager_window
            )
            assert float((got - ref.float()).abs().max()) < 1e-5
    assert graph_kv.length == DEC_SPAN + len(cases)
    assert eager_kv.keys[0] is not None
    assert eager_kv.keys[0].shape[2] == DEC_SPAN
