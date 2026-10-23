"""A synthetic chart becomes a V1Batch the 8/8 model can score."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from audio2map.dataset.v1 import ChartAudio, build_v1_example, collate_v1
from audio2map.features.audio.v1.constants import DT_MS, N_MELS, SAMPLE_RATE
from audio2map.features.cond.chart import ChartMeta, ChartMetaVersionError
from audio2map.features.cond.v1.mask import sample_cond_mask
from audio2map.features.cond.v1.vec import COND_VEC_NAMES, COND_VEC_VERSION, build_cond
from audio2map.model.audio_stem import AudioStem
from audio2map.model.v1.loss import plain_ce
from audio2map.model.v1.model import V1ChartModel
from audio2map.osu.parser import is_mania_4k
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1 import cursor_times_ms
from audio2map.tokens.v1.codec import encode_chart
from audio2map.tokens.v1.vocab import TOKEN_PAD, build_vocab
from audio2map.utils.errors import BAD_AUDIO_DURATION, ChartRejected
from audio2map.utils.paths import raw_dir


def _pcm(t_audio_ms: float) -> np.ndarray:
    n = int(round(t_audio_ms / 1000.0 * SAMPLE_RATE))
    return np.zeros(n, dtype=np.float32)


def _chart(t_audio_ms: float) -> ChartAudio:
    return ChartAudio(
        notes=(ManiaNote(0, 0, NoteType.TAP),),
        redlines=(Redline(0.0, 100.0, 4, 0),),
        pcm=_pcm(t_audio_ms),
        t_audio_ms=t_audio_ms,
    )


def _meta() -> ChartMeta:
    return ChartMeta(
        osu_path="synth.osu",
        official_sr=5.0,
        pattern_stream=0.25,
        error="rework_sr: synthetic diagnostic",
    )


def test_full_chart_batch_matches_the_model_contract() -> None:
    meta = _meta()
    batch = build_v1_example(_chart(200.0), meta=meta)
    encoded = encode_chart(
        [ManiaNote(0, 0, NoteType.TAP)],
        [Redline(0.0, 100.0, 4, 0)],
        200.0,
    )
    frames = int(batch.n_frames[0])
    assert batch.precomputed is False
    assert batch.audio is None
    assert batch.mel is not None
    assert batch.mel.shape == (1, N_MELS, 2 * frames + 3)
    assert batch.token_ids.shape[0] == 1
    assert batch.cursor_time_ms.shape == (1, batch.token_ids.shape[1] - 1)
    assert batch.phi.shape == batch.beat_b.shape == (1, frames)
    assert batch.cond_x.shape == batch.cond_m.shape == (1, 18)
    assert bool(batch.audio_mask.all())
    expected_times = cursor_times_ms(encoded.tokens, encoded.grid)
    assert torch.allclose(
        batch.cursor_time_ms[0],
        torch.tensor(expected_times, dtype=torch.float32),
    )
    g0 = int(batch.frame_lo[0])
    phi0, beat0 = encoded.grid.phase_at(g0 * DT_MS)
    assert batch.phi[0, 0].item() == pytest.approx(phi0)
    assert batch.beat_b[0, 0].item() == pytest.approx(beat0)
    assert torch.all((batch.phi >= 0.0) & (batch.phi < 1.0))
    cond = build_cond(meta)
    assert torch.allclose(batch.cond_x[0], torch.from_numpy(cond.values))
    assert torch.equal(batch.cond_m[0], torch.from_numpy(cond.m_real))
    assert batch.cond_m[0, 1].item() == 0.0
    assert COND_VEC_VERSION == 3


def test_eval_mask_stays_on_m_real() -> None:
    meta = _meta()
    real = torch.from_numpy(build_cond(meta).m_real)
    first = build_v1_example(_chart(200.0), meta=meta, cond_mask="none")
    second = build_v1_example(_chart(200.0), meta=meta)
    assert torch.equal(first.cond_m[0], real)
    assert torch.equal(second.cond_m[0], real)


def test_train_mask_repeats_for_a_fixed_rng() -> None:
    meta = _meta()
    real = build_cond(meta).m_real
    first = build_v1_example(
        _chart(200.0),
        meta=meta,
        cond_mask="train",
        rng=np.random.default_rng(0),
    )
    second = build_v1_example(
        _chart(200.0),
        meta=meta,
        cond_mask="train",
        rng=np.random.default_rng(0),
    )
    expected = sample_cond_mask(real, np.random.default_rng(0))
    assert torch.equal(first.cond_m[0], second.cond_m[0])
    assert torch.equal(first.cond_m[0], torch.from_numpy(expected))
    assert torch.all(first.cond_m[0] <= torch.from_numpy(real))
    assert first.cond_m[0, 1].item() == 0.0


def test_shared_stem_receives_backward() -> None:
    stem = AudioStem()
    model = V1ChartModel(stem=stem)
    assert model.stem is stem
    batch = build_v1_example(_chart(200.0), meta=_meta(), cond_mask="none")
    model.train()
    loss = plain_ce(model(batch), batch.token_ids[:, 1:])
    assert torch.isfinite(loss)
    loss.backward()
    assert stem.conv1.weight.grad is not None


def test_precomputed_rows_do_not_train_the_model_stem() -> None:
    batch = build_v1_example(
        _chart(200.0),
        meta=_meta(),
        precomputed_stem=AudioStem(),
    )
    assert batch.precomputed is True
    assert batch.mel is None
    assert batch.audio is not None
    assert batch.audio.shape[-1] == 512
    model = V1ChartModel()
    model.train()
    loss = plain_ce(model(batch), batch.token_ids[:, 1:])
    loss.backward()
    assert model.stem.conv1.weight.grad is None


def test_tensor_cond_mask_is_anded_with_m_real() -> None:
    meta = _meta()
    real = build_cond(meta).m_real
    supplied = torch.ones(len(COND_VEC_NAMES))
    supplied[COND_VEC_NAMES.index("official_sr_norm")] = 0.0
    supplied[COND_VEC_NAMES.index("rework_sr_norm")] = 1.0
    batch = build_v1_example(_chart(200.0), meta=meta, cond_mask=supplied)
    expected = np.asarray(supplied.numpy() * real, dtype=np.float32)
    assert torch.equal(batch.cond_m[0], torch.from_numpy(expected))
    assert batch.cond_m[0, COND_VEC_NAMES.index("official_sr_norm")].item() == 0.0
    assert batch.cond_m[0, COND_VEC_NAMES.index("rework_sr_norm")].item() == 0.0
    assert batch.cond_m[0, COND_VEC_NAMES.index("pattern_stream")].item() == 1.0


def test_zero_audio_duration_is_rejected() -> None:
    with pytest.raises(ChartRejected) as caught:
        build_v1_example(_chart(0.0), meta=_meta())
    assert caught.value.reason == BAD_AUDIO_DURATION


def test_positive_duration_with_no_stem_frames_is_rejected() -> None:
    # The redline is just after 0, so coverage is one minimum beat that snaps to one frame index.
    chart = ChartAudio(
        notes=(),
        redlines=(Redline(1e-8, 1e-4, 4, 0),),
        pcm=np.zeros(0, dtype=np.float32),
        t_audio_ms=5e-9,
    )
    assert chart.t_audio_ms > 0.0
    with pytest.raises(ChartRejected) as caught:
        build_v1_example(chart, meta=_meta())
    assert caught.value.reason == BAD_AUDIO_DURATION


def test_wrong_cond_version_rejects_the_chart() -> None:
    with pytest.raises(ChartMetaVersionError):
        build_v1_example(
            _chart(200.0),
            meta={"osu_path": "old.osu", "cond_vec_version": 1, "pattern_stream": 0.9},
        )


def test_frame_phase_matches_phase_at_across_a_residual() -> None:
    length = 500.0
    t_red = 1000.0 + 4.25e-5
    notes = (ManiaNote(0, 0, NoteType.TAP),)
    redlines = (Redline(0.0, length, 4, 0), Redline(t_red, 400.0, 4, 1))
    chart = ChartAudio(notes=notes, redlines=redlines, pcm=_pcm(2000.0), t_audio_ms=2000.0)
    batch = build_v1_example(chart, meta=_meta(), cond_mask="none")
    encoded = encode_chart(list(notes), list(redlines), 2000.0)
    residual = next(beat for beat in encoded.grid.beats if beat.is_residual)
    inside = residual.start_ms + 2.0e-5
    phi, beat = encoded.grid.phase_at(inside)
    assert beat == residual.global_b
    assert phi == pytest.approx(2.0e-5 / length)
    g0 = int(batch.frame_lo[0])
    for index in range(int(batch.n_frames[0])):
        expect_phi, expect_b = encoded.grid.phase_at((g0 + index) * DT_MS)
        assert batch.phi[0, index].item() == pytest.approx(expect_phi)
        assert batch.beat_b[0, index].item() == pytest.approx(float(expect_b))


def test_collate_pads_without_marking_silence_as_pad() -> None:
    meta = _meta()
    short = build_v1_example(_chart(200.0), meta=meta, cond_mask="none")
    long = build_v1_example(_chart(400.0), meta=meta, cond_mask="none")
    assert bool(short.audio_mask.all())
    packed = collate_v1([short, long])
    short_frames = int(short.n_frames[0])
    long_frames = int(long.n_frames[0])
    assert long_frames > short_frames
    assert bool(packed.audio_mask[0, :short_frames].all())
    assert not bool(packed.audio_mask[0, short_frames:].any())
    assert bool(packed.audio_mask[1].all())
    short_steps = int(short.token_ids.shape[1]) - 1
    assert bool(packed.token_mask[0, :short_steps].all())
    assert not bool(packed.token_mask[0, short_steps:].any())
    pad_id = int(packed.token_ids[0, short.token_ids.shape[1]])
    assert pad_id == build_vocab()[TOKEN_PAD]
    model = V1ChartModel()
    model.eval()
    logits = model(packed)
    loss = plain_ce(logits, packed.token_ids[:, 1:])
    assert logits.shape[0] == 2
    assert torch.isfinite(loss)


def _first_4k_osu() -> Path | None:
    root = raw_dir()
    if not root.is_dir():
        return None
    for path in root.rglob("*.osu"):
        try:
            if is_mania_4k(path):
                return path
        except OSError:
            continue
    return None


def test_local_osu_forward_uses_the_eval_mask() -> None:
    path = _first_4k_osu()
    if path is None:
        pytest.skip("no 4K .osu under the raw data directory")
    meta = ChartMeta(osu_path=str(path), official_sr=1.0)
    batch = build_v1_example(path, meta=meta, cond_mask="none")
    real = torch.from_numpy(build_cond(meta).m_real)
    assert torch.equal(batch.cond_m[0], real)
    assert batch.precomputed is False
    model = V1ChartModel()
    model.eval()
    logits = model(batch)
    loss = plain_ce(logits, batch.token_ids[:, 1:])
    assert torch.isfinite(loss)
