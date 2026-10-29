"""v1 chart generation. Not the v0 16-bar stitch and not ``teacher_decode``.

The inputs are PCM, red lines, and a caller-supplied condition. The result is
the chart. Writing a file is a later step.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from audio2map.features.audio.v1.constants import SAMPLE_RATE
from audio2map.features.cond.chart import ChartMeta
from audio2map.features.cond.v1.vec import build_cond
from audio2map.generate.v1.notes import V1Note, notes_from_events
from audio2map.generate.v1.sample import sample_chart
from audio2map.generate.v1.stream import KEEP_FRAMES, encode_context
from audio2map.grid.v1.beatgrid import build_beatgrid
from audio2map.model.v1.model import V1ChartModel
from audio2map.model.v1.weights import load_v1_state_dict
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1.codec import decode_tokens


@dataclass(frozen=True, slots=True)
class V1Chart:
    notes: tuple[V1Note, ...]
    warnings: tuple[str, ...]
    tokens: tuple[str, ...]
    frames: int
    segments: int


def load_v1_model(path: Path | None, device: torch.device) -> V1ChartModel:
    """Load a train ``step_N.pt``, or the initialized module when ``path`` is omitted."""
    model = V1ChartModel()
    if path is not None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        load_v1_state_dict(model, state)
        if isinstance(ckpt, dict) and "ca_train" in ckpt:
            model.set_ca_train(str(ckpt["ca_train"]))
    model.to(device)
    model.eval()
    return model


def condition_from_json(path: Path | str) -> ChartMeta:
    """Read a caller-supplied condition. ``osu_path`` may be absent."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("condition file must be a ChartMeta object")
    data.setdefault("osu_path", "")
    return ChartMeta.from_dict(data)


def generate_v1_chart(
    model: V1ChartModel,
    pcm: np.ndarray,
    redlines: Sequence[Redline],
    meta: ChartMeta,
    *,
    device: torch.device,
    seed: int = 0,
    temperature: float = 1.0,
    keep_frames: int = KEEP_FRAMES,
) -> V1Chart:
    """Sample one chart at rate 1. Condition dropout stays off.

    ``pcm`` is mono audio already at 22050 Hz. Duration is its sample count.
    ``redlines`` are the timing input. ``meta`` is the condition; missing
    fields stay masked.
    """
    model.eval()
    audio, t_audio = _pcm(pcm)
    grid = build_beatgrid(redlines, t_audio)
    context, g, segments = encode_context(
        model,
        audio,
        grid,
        redlines,
        device=device,
        keep_frames=keep_frames,
    )
    cond_vec = build_cond(meta)
    values = torch.from_numpy(cond_vec.values).unsqueeze(0).to(device)
    mask = torch.from_numpy(cond_vec.m_real).unsqueeze(0).to(device)
    with torch.inference_mode():
        cond = model.cond(values, mask)
    tokens = sample_chart(
        model,
        context,
        g,
        grid,
        cond=cond,
        seed=seed,
        temperature=temperature,
    )
    events = decode_tokens(tokens, grid)
    notes, warnings = notes_from_events(events)
    return V1Chart(notes, warnings, tuple(tokens), int(g.shape[1]), segments)


def _pcm(pcm: np.ndarray) -> tuple[np.ndarray, float]:
    audio = np.asarray(pcm, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("PCM audio must be a non-empty 1-d buffer at 22050 Hz")
    duration = float(audio.shape[0]) / float(SAMPLE_RATE) * 1000.0
    return np.ascontiguousarray(audio), duration
