#!/usr/bin/env python3
"""Compute MEL_NORM_V1: one global mean and std of ``ln(mel + 1e-5)``.

Sampling, fixed so the constants can be copied into
``audio2map/features/audio/v1/constants.py``:

- scheme ``beatmapset_v1``, seed 0, ratios 0.8 / 0.1 / 0.1
- group id is the chart's parent directory under ``raw/``
- keep the train split only
- one audio file per group (the file named by the first chart in that group)
- ``numpy`` Generator seed 0, shuffle those paths, take the first 50
- each file is decoded with the v1 frontend (ffmpeg native rate, equal-weight
  mono, soxr HQ to 22050 Hz)
- mel frames come from the v1 buffer of ``[0, T_audio)``, ``center=False``
- statistics are the population mean and std (``ddof=0``) over every frame and
  every mel bin together, so μ and σ are scalars

This does not read ``exports/mel_energy_stats/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from audio2map.dataset.split import _assign_split, group_id_for_chart
from audio2map.features.audio.tick_features import AUDIO_SAMPLE_RATE
from audio2map.features.audio.v1 import (
    MEL_FLOOR,
    buffer_sample_range,
    decode_audio,
    mel_power,
    slice_pcm,
    frame_index,
)
from audio2map.osu.parser import chart_audio_path, is_mania_4k
from audio2map.utils.paths import raw_dir


def train_audio_paths(raw_root: Path) -> list[Path]:
    chosen: dict[str, Path] = {}
    for path in sorted(raw_root.rglob("*.osu")):
        if not is_mania_4k(path):
            continue
        group = group_id_for_chart(path, raw_root)
        if _assign_split(group, seed=0, ratios={"train": 0.8, "val": 0.1, "test": 0.1}) != "train":
            continue
        chosen.setdefault(group, chart_audio_path(path))
    paths = sorted({path.resolve() for path in chosen.values() if path.is_file()})
    rng = np.random.default_rng(0)
    order = rng.permutation(len(paths))
    return [paths[int(index)] for index in order]


def song_moments(path: Path) -> tuple[float, float, int, str]:
    decoded = decode_audio(path)
    if decoded.timeline_broken:
        raise RuntimeError(f"timeline broken: {path}")
    t_audio = decoded.t_audio_ms
    frame_lo = frame_index(0.0)
    frame_hi = frame_index(t_audio)
    start, end = buffer_sample_range(frame_lo, frame_hi)
    buffer = slice_pcm(decoded.y, start, end)
    mel = mel_power(buffer, AUDIO_SAMPLE_RATE)
    logged = np.log(np.asarray(mel, dtype=np.float64) + MEL_FLOOR)
    total = float(logged.size)
    return float(logged.sum()), float(np.square(logged).sum()), int(total), decoded.ffmpeg_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--songs", type=int, default=50)
    args = parser.parse_args()
    paths = train_audio_paths(raw_dir())[: args.songs]
    if len(paths) < args.songs:
        raise SystemExit(f"only {len(paths)} train audio files")
    total = 0
    sum_x = 0.0
    sum_x2 = 0.0
    for index, path in enumerate(paths, start=1):
        sx, sx2, count, version = song_moments(path)
        if index == 1:
            print(f"ffmpeg {version}", flush=True)
        sum_x += sx
        sum_x2 += sx2
        total += count
        print(f"{index}/{len(paths)} values {count} {path.name}", flush=True)
    mu = sum_x / total
    var = sum_x2 / total - mu * mu
    sigma = float(np.sqrt(max(var, 0.0)))
    print(f"MEL_NORM_MU = {mu:.10f}")
    print(f"MEL_NORM_SIGMA = {sigma:.10f}")
    print(f"n_values {total} n_songs {len(paths)}")


if __name__ == "__main__":
    main()
