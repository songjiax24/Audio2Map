"""One full chart as a batch of size 1. Windowing stays undecided.

The batch carries normalized mel. ``V1ChartModel.stem`` is the trainable
convolution. This function never constructs an ``AudioStem``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from audio2map.features.audio.v1.constants import DT_MS, N_MELS, SAMPLE_RATE
from audio2map.features.audio.v1.decode import decode_audio
from audio2map.features.audio.v1.mel import mel_span
from audio2map.features.cond.chart import ChartMeta, compute_chart_meta
from audio2map.features.cond.v1.mask import sample_cond_mask
from audio2map.features.cond.v1.vec import build_cond
from audio2map.model.audio_stem import AudioStem, stem_span
from audio2map.model.v1.batch import V1Batch
from audio2map.osu.parser import chart_audio_path, is_mania_4k, parse_beatmap
from audio2map.osu.schema import ManiaNote
from audio2map.osu.v1.timing import Redline, parse_redlines_file
from audio2map.tokens.v1.codec import encode_chart
from audio2map.tokens.v1.cursor_time import cursor_times_ms
from audio2map.tokens.v1.vocab import build_vocab
from audio2map.utils.errors import BAD_AUDIO_DURATION, ChartRejected

CondMask = Literal["train", "none"]


@dataclass(frozen=True, slots=True)
class ChartAudio:
    """Parsed 4K chart and PCM already at the model rate."""

    notes: tuple[ManiaNote, ...]
    redlines: tuple[Redline, ...]
    pcm: np.ndarray
    t_audio_ms: float | None = None


def build_v1_example(
    source: Path | str | ChartAudio,
    *,
    meta: ChartMeta | dict | None = None,
    cond_mask: CondMask | torch.Tensor | None = "none",
    rng: np.random.Generator | None = None,
    precomputed_stem: AudioStem | None = None,
) -> V1Batch:
    """Teacher-forcing batch for the whole chart. ``meta.error`` does not reject it.

    ``cond_mask="none"`` (the default) sets ``cond_m = m_real``. ``"train"``
    draws the Composer mask. A tensor is used as given, then AND-ed with
    ``m_real``. A dict ``meta`` must carry the current v1 ``cond_vec_version``.

    ``precomputed_stem`` fills detached stem rows and sets ``precomputed``.
    Omit it so the model can train its own stem on ``mel``.
    """
    resolved = _resolve_meta(source, meta)
    notes, redlines, pcm, t_audio = _load_chart(source)
    if not np.isfinite(t_audio) or t_audio <= 0.0:
        raise ChartRejected(BAD_AUDIO_DURATION, f"T_audio={t_audio}")
    encoded = encode_chart(notes, redlines, t_audio)
    grid = encoded.grid
    span = mel_span(pcm, grid.coverage_start_ms, grid.coverage_end_ms)
    n_frames = span.frame_hi - span.frame_lo
    if n_frames <= 0 or span.mel.shape[1] == 0:
        raise ChartRejected(BAD_AUDIO_DURATION, "audio span has no stem frames")
    expected = 2 * n_frames + 3
    if span.mel.shape != (N_MELS, expected):
        raise ValueError(f"mel shape {span.mel.shape} != {(N_MELS, expected)}")
    phi, beat_b = _phase_rows(grid, span.frame_lo, n_frames)
    vocab = build_vocab()
    token_ids = [vocab[token] for token in encoded.tokens]
    times = cursor_times_ms(encoded.tokens, grid)
    cond = build_cond(resolved)
    chosen = _choose_mask(cond.m_real, cond_mask, rng)
    audio = None
    mel = torch.from_numpy(np.ascontiguousarray(span.mel)).unsqueeze(0)
    precomputed = False
    if precomputed_stem is not None:
        rows = stem_span(
            pcm,
            grid.coverage_start_ms,
            grid.coverage_end_ms,
            precomputed_stem,
        )
        audio = torch.from_numpy(np.ascontiguousarray(rows.features)).unsqueeze(0)
        mel = None
        precomputed = True
    length = len(token_ids)
    return V1Batch(
        frame_lo=torch.tensor([span.frame_lo], dtype=torch.long),
        cond_x=torch.from_numpy(cond.values).unsqueeze(0),
        cond_m=torch.from_numpy(np.ascontiguousarray(chosen)).unsqueeze(0),
        token_ids=torch.tensor([token_ids], dtype=torch.long),
        cursor_time_ms=torch.tensor([times], dtype=torch.float32),
        phi=torch.from_numpy(phi.astype(np.float32)).unsqueeze(0),
        beat_b=torch.from_numpy(beat_b.astype(np.float32)).unsqueeze(0),
        audio_mask=torch.ones(1, n_frames, dtype=torch.bool),
        token_mask=torch.ones(1, length - 1, dtype=torch.bool),
        mel=mel,
        n_frames=torch.tensor([n_frames], dtype=torch.long),
        audio=audio,
        precomputed=precomputed,
    )


def _choose_mask(
    m_real: np.ndarray,
    cond_mask: CondMask | torch.Tensor | None,
    rng: np.random.Generator | None,
) -> np.ndarray:
    if cond_mask is None or cond_mask == "none":
        return np.array(m_real, dtype=np.float32, copy=True)
    if cond_mask == "train":
        generator = rng if rng is not None else np.random.default_rng()
        return sample_cond_mask(m_real, generator)
    if isinstance(cond_mask, torch.Tensor):
        supplied = cond_mask.detach().to(dtype=torch.float32).reshape(-1).cpu().numpy()
        if supplied.shape != m_real.shape:
            raise ValueError("cond_mask length must match the condition")
        return np.asarray(supplied * m_real, dtype=np.float32)
    raise ValueError("cond_mask must be 'train', 'none', or a tensor")


def _resolve_meta(source: Path | str | ChartAudio, meta: ChartMeta | dict | None) -> ChartMeta:
    if isinstance(source, (str, Path)) and meta is None:
        return compute_chart_meta(source)
    osu_path = str(source) if isinstance(source, (str, Path)) else ""
    if meta is None:
        return ChartMeta(osu_path=osu_path)
    if isinstance(meta, dict):
        return ChartMeta.from_dict(meta)
    return meta


def _load_chart(
    source: Path | str | ChartAudio,
) -> tuple[list[ManiaNote], tuple[Redline, ...], np.ndarray, float]:
    if isinstance(source, ChartAudio):
        pcm = np.asarray(source.pcm, dtype=np.float32)
        if source.t_audio_ms is None:
            t_audio = float(pcm.shape[0]) / float(SAMPLE_RATE) * 1000.0
        else:
            t_audio = float(source.t_audio_ms)
        return list(source.notes), tuple(source.redlines), pcm, t_audio
    path = Path(source)
    if not is_mania_4k(path):
        raise ValueError(f"not a 4K mania chart: {path}")
    beatmap = parse_beatmap(path)
    redlines = parse_redlines_file(path).redlines
    decoded = decode_audio(chart_audio_path(path))
    return list(beatmap.notes), redlines, decoded.y, decoded.t_audio_ms


def _phase_rows(grid, frame_lo: int, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    phi = np.empty(n_frames, dtype=np.float64)
    beat_b = np.empty(n_frames, dtype=np.float64)
    for index in range(n_frames):
        t_ms = (frame_lo + index) * DT_MS
        phase, beat = grid.phase_at(t_ms)
        phi[index] = phase
        beat_b[index] = beat
    return phi, beat_b
