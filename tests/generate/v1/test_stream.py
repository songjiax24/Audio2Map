"""Segment indexes cover the encoder receptive field. A short song matches."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from audio2map.features.audio.v1.constants import DT_MS, SAMPLE_RATE
from audio2map.features.audio.v1.mel import frame_index, mel_span
from audio2map.generate.v1.stream import ENCODER_HALO, encode_context, iter_segments
from audio2map.grid.v1.beatgrid import build_beatgrid
from audio2map.model.v1.model import V1ChartModel, phase_fourier
from audio2map.osu.v1.timing import Redline


def test_each_kept_frame_sees_its_halo() -> None:
    n_frames = 20
    halo = 5
    keep = 6
    covered: list[int] = []
    for encode_lo, encode_hi, keep_lo, keep_hi in iter_segments(n_frames, keep=keep, halo=halo):
        for index in range(keep_lo, keep_hi):
            need_lo = max(0, index - halo)
            need_hi = min(n_frames, index + halo)
            assert encode_lo <= need_lo
            assert encode_hi >= need_hi
        covered.extend(range(keep_lo, keep_hi))
    assert covered == list(range(n_frames))
    assert ENCODER_HALO == 8 * 512


def test_stitched_keeps_match_one_span() -> None:
    torch.manual_seed(2)
    model = V1ChartModel().eval()
    seconds = 0.8
    rng = np.random.default_rng(2)
    pcm = rng.standard_normal(int(SAMPLE_RATE * seconds)).astype(np.float32) * 0.05
    t_audio = float(pcm.shape[0]) / SAMPLE_RATE * 1000.0
    redlines = (Redline(0.0, 500.0, 4, 0),)
    grid = build_beatgrid(redlines, t_audio)
    device = torch.device("cpu")
    parts, g_parts, n_parts = encode_context(
        model, pcm, grid, redlines, device=device, keep_frames=5
    )
    whole, g_whole, n_whole = encode_context(
        model, pcm, grid, redlines, device=device, keep_frames=10**9
    )
    assert n_parts > 1
    assert n_whole == 1
    assert torch.equal(g_parts, g_whole)
    gap = (parts - whole).abs().max().item()
    assert gap < 1e-5, gap


def test_short_song_segment_matches_one_encode() -> None:
    torch.manual_seed(0)
    model = V1ChartModel().eval()
    seconds = 0.4
    rng = np.random.default_rng(0)
    pcm = rng.standard_normal(int(SAMPLE_RATE * seconds)).astype(np.float32) * 0.05
    t_audio = seconds * 1000.0
    redlines = (Redline(0.0, 500.0, 4, 0),)
    grid = build_beatgrid(redlines, t_audio)
    context, g, segments = encode_context(model, pcm, grid, redlines, device=torch.device("cpu"))
    assert segments == 1
    frame_lo = frame_index(grid.coverage_start_ms)
    frame_hi = frame_index(grid.coverage_end_ms)
    span = mel_span(pcm, frame_lo * DT_MS, frame_hi * DT_MS)
    mel = torch.from_numpy(np.ascontiguousarray(span.mel)).unsqueeze(0)
    origin = torch.tensor([frame_lo], dtype=torch.long)
    mask = torch.ones(1, frame_hi - frame_lo, dtype=torch.bool)
    with torch.no_grad():
        memory = model.encoder(mel, origin, mask)
        phase = _phase_like(model, grid, redlines, frame_lo, frame_hi - frame_lo)
        full = memory + model.phase(phase)
    assert int(g[0, 0]) == frame_lo
    assert (context - full).abs().max() < 1e-5


def _phase_like(model, grid, redlines, frame_lo: int, n_frames: int) -> torch.Tensor:
    from audio2map.grid.v1.timefeat import phase_rows

    phi, beat_b, log_bpm, rel_u = phase_rows(grid, redlines, frame_lo, n_frames)
    features = np.stack((phi, beat_b, log_bpm, rel_u), axis=0).astype(np.float32)
    stacked = torch.from_numpy(features).unsqueeze(0)
    return phase_fourier(stacked[:, 0], stacked[:, 1], stacked[:, 2], stacked[:, 3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_halo_matches_the_full_span() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    model = V1ChartModel().eval().to(device)
    # Longer than one halo, so the second segment does not see the song start.
    n_target = ENCODER_HALO + 128
    seconds = (n_target + 8) * DT_MS / 1000.0
    rng = np.random.default_rng(1)
    pcm = rng.standard_normal(int(SAMPLE_RATE * seconds)).astype(np.float32) * 0.05
    t_audio = float(pcm.shape[0]) / SAMPLE_RATE * 1000.0
    redlines = (Redline(0.0, 500.0, 4, 0),)
    grid = build_beatgrid(redlines, t_audio)
    context, g, segments = encode_context(
        model,
        pcm,
        grid,
        redlines,
        device=device,
        keep_frames=256,
    )
    assert segments > 1
    frame_lo = int(g[0, 0])
    frame_hi = frame_lo + int(g.shape[1])
    span = mel_span(pcm, frame_lo * DT_MS, frame_hi * DT_MS)
    mel = torch.from_numpy(np.ascontiguousarray(span.mel)).unsqueeze(0).to(device)
    origin = torch.tensor([frame_lo], dtype=torch.long, device=device)
    mask = torch.ones(1, frame_hi - frame_lo, dtype=torch.bool, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        memory = model.encoder(mel, origin, mask).float()
        phase = _phase_like(model, grid, redlines, frame_lo, frame_hi - frame_lo)
        phase = phase.to(device)
        full = memory + model.phase(phase)
    gap = (context.float() - full.float()).abs().max().item()
    assert gap < 1e-2, gap
