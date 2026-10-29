"""Frame log BPM and u follow the stretched chart, including batch pad."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from audio2map.dataset.v1.augment import AugmentChoice, apply_clock
from audio2map.dataset.v1.collate import collate_v1
from audio2map.dataset.v1.example import ChartAudio, build_v1_example
from audio2map.features.audio.v1.constants import DT_MS, SAMPLE_RATE
from audio2map.features.cond.chart import ChartMeta
from audio2map.grid.v1.timefeat import LOG_BPM_SCALE, bpm_at, log_bpm_norm, relative_u
from audio2map.model.v1.constants import PHASE_DIM
from audio2map.model.v1.model import phase_fourier
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.osu.v1.timing import Redline


def _pcm(t_audio_ms: float) -> np.ndarray:
    n = int(round(t_audio_ms / 1000.0 * SAMPLE_RATE))
    return np.zeros(n, dtype=np.float32)


def _meta() -> ChartMeta:
    return ChartMeta(osu_path="synth.osu", official_sr=5.0, pattern_stream=0.25)


def test_phase_features_are_20_and_keep_the_two_scalars() -> None:
    assert PHASE_DIM == 20
    phi = torch.zeros(1, 4)
    beat = torch.zeros(1, 4)
    log_bpm = torch.arange(4, dtype=torch.float32).unsqueeze(0)
    rel_u = torch.full((1, 4), 1.5)
    feat = phase_fourier(phi, beat, log_bpm, rel_u)
    assert feat.shape[-1] == 20
    assert torch.equal(feat[0, :, -2], log_bpm[0])
    assert torch.equal(feat[0, :, -1], rel_u[0])


def test_built_frames_change_log_bpm_across_a_redline() -> None:
    redlines = (Redline(0.0, 500.0, 4, 0), Redline(1000.0, 250.0, 4, 1))
    chart = ChartAudio(
        notes=(ManiaNote(0, 0, NoteType.TAP),),
        redlines=redlines,
        pcm=_pcm(2000.0),
        t_audio_ms=2000.0,
    )
    batch = build_v1_example(chart, meta=_meta(), cond_mask="none")
    frame_lo = int(batch.frame_lo[0])
    before = after = None
    for index in range(int(batch.n_frames[0])):
        t_ms = (frame_lo + index) * DT_MS
        if t_ms < 1000.0:
            before = index
        elif after is None:
            after = index
    assert before is not None and after is not None
    gap = batch.log_bpm[0, after].item() - batch.log_bpm[0, before].item()
    assert gap == pytest.approx(math.log(2.0) / LOG_BPM_SCALE)
    t0 = float(batch.coverage_start_ms[0])
    t1 = float(batch.coverage_end_ms[0])
    t_after = (frame_lo + after) * DT_MS
    assert batch.rel_u[0, after].item() == pytest.approx(relative_u(t_after, t0, t1))


def test_right_pad_u_can_pass_one_without_masking() -> None:
    short = build_v1_example(_chart(200.0), meta=_meta(), cond_mask="none")
    long = build_v1_example(_chart(800.0), meta=_meta(), cond_mask="none")
    packed = collate_v1([short, long])
    start = int(short.n_frames[0])
    assert start < int(long.n_frames[0])
    frame_lo = int(short.frame_lo[0])
    t0 = float(short.coverage_start_ms[0])
    t1 = float(short.coverage_end_ms[0])
    t_pad = (frame_lo + start) * DT_MS
    expected = relative_u(t_pad, t0, t1)
    assert expected > 1.0
    assert packed.rel_u[0, start].item() == pytest.approx(expected)
    offsets = short.redline_offset_ms[0].tolist()
    beats = short.redline_beat_ms[0].tolist()
    for index in range(start, int(long.n_frames[0])):
        t_ms = (frame_lo + index) * DT_MS
        assert packed.rel_u[0, index].item() == pytest.approx(relative_u(t_ms, t0, t1))
        assert packed.log_bpm[0, index].item() == pytest.approx(
            log_bpm_norm(bpm_at((Redline(offsets[0], beats[0], 4, 0),), t_ms))
        )
    assert not bool(packed.audio_mask[0, start])
    assert bool(packed.audio_mask[0, start - 1])
    left = relative_u((frame_lo - 1) * DT_MS, t0, t1)
    assert left < 0.0


def test_rate_stretch_changes_log_bpm() -> None:
    lines = (Redline(0.0, 500.0, 4, 0),)
    base = log_bpm_norm(bpm_at(lines, 0.0))
    _notes, scaled, _pcm_out = apply_clock(
        [ManiaNote(0, 0, NoteType.TAP)],
        lines,
        _pcm(1000.0),
        AugmentChoice(mirror=False, rate=2.0),
    )
    assert log_bpm_norm(bpm_at(scaled, 0.0)) - base == pytest.approx(
        math.log(2.0) / LOG_BPM_SCALE
    )


def _chart(t_audio_ms: float) -> ChartAudio:
    return ChartAudio(
        notes=(ManiaNote(0, 0, NoteType.TAP),),
        redlines=(Redline(0.0, 500.0, 4, 0),),
        pcm=_pcm(t_audio_ms),
        t_audio_ms=t_audio_ms,
    )

