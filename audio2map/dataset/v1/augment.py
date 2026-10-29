"""Training-only mirror and clock-rate. Inference leaves both off.

Mirror swaps 4K columns and does not touch the audio. Rate resamples the
PCM with :func:`stretch_pcm` and divides redline times by the same rate.
Tokens are labeled on the original integer milliseconds, then the beat grid
is scaled. Note times are not rounded onto the V80 window again. Difficulty
scalars are not computed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from audio2map.features.audio.v1.stretch import stretch_pcm
from audio2map.grid.v1.beatgrid import scale_beatgrid
from audio2map.osu.schema import ManiaNote
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1.codec import EncodeResult, encode_chart

MIRROR_P = 0.5
RATE_P = 0.2
RATE_MIN = 1.0
RATE_MAX = 1.5
N_COLUMNS = 4


@dataclass(frozen=True, slots=True)
class AugmentChoice:
    """``rate=1`` and ``mirror=False`` is the inference chart."""

    mirror: bool
    rate: float


def sample_augment(rng: np.random.Generator, *, enabled: bool) -> AugmentChoice:
    """Draw mirror and rate. Disabled training, validation, and inference stay at rate 1."""
    if not enabled:
        return AugmentChoice(mirror=False, rate=1.0)
    mirror = float(rng.random()) < MIRROR_P
    if float(rng.random()) >= RATE_P:
        return AugmentChoice(mirror=mirror, rate=1.0)
    rate = float(rng.uniform(RATE_MIN, RATE_MAX))
    return AugmentChoice(mirror=mirror, rate=rate)


def mirror_column(col: int) -> int:
    if col < 0 or col >= N_COLUMNS:
        raise ValueError(f"4K column must be 0-3, got {col}")
    return N_COLUMNS - 1 - col


@dataclass(frozen=True, slots=True)
class ClockChart:
    """Tokens from the osu integers, and a timeline at ``choice.rate``."""

    notes: tuple[ManiaNote, ...]
    redlines: tuple[Redline, ...]
    pcm: np.ndarray
    encoded: EncodeResult


def apply_clock(
    notes: list[ManiaNote],
    redlines: tuple[Redline, ...],
    pcm: np.ndarray,
    choice: AugmentChoice,
) -> tuple[list[ManiaNote], tuple[Redline, ...], np.ndarray]:
    """Mirror columns, stretch PCM, and scale redlines. Note times stay put."""
    audio = np.asarray(pcm, dtype=np.float32)
    scaled_notes = [_mirror_note(note) for note in notes] if choice.mirror else notes
    if choice.rate == 1.0:
        return scaled_notes, redlines, audio
    _check_rate(choice.rate)
    audio = stretch_pcm(audio, choice.rate)
    scaled_lines = tuple(_scale_redline(line, choice.rate) for line in redlines)
    return scaled_notes, scaled_lines, audio


def clock_chart(
    notes: list[ManiaNote],
    redlines: tuple[Redline, ...],
    pcm: np.ndarray,
    t_audio_ms: float,
    choice: AugmentChoice,
) -> ClockChart:
    """Label at rate 1, then scale the grid and the audio clock together.

    Mirror is applied before labeling, so a mirrored chart keeps that token
    string at every rate. The returned notes still carry the original
    millisecond times.
    """
    mirrored = [_mirror_note(note) for note in notes] if choice.mirror else list(notes)
    encoded = encode_chart(mirrored, redlines, t_audio_ms)
    if choice.rate == 1.0:
        audio = np.asarray(pcm, dtype=np.float32)
        return ClockChart(tuple(mirrored), redlines, audio, encoded)
    _check_rate(choice.rate)
    audio = stretch_pcm(np.asarray(pcm, dtype=np.float32), choice.rate)
    scaled_lines = tuple(_scale_redline(line, choice.rate) for line in redlines)
    grid = scale_beatgrid(encoded.grid, choice.rate)
    encoded = EncodeResult(
        tokens=encoded.tokens,
        grid=grid,
        labels=encoded.labels,
        endpoints=encoded.endpoints,
        near_duplicate_pairs=grid.near_duplicate_pairs,
    )
    return ClockChart(tuple(mirrored), scaled_lines, audio, encoded)


def _check_rate(rate: float) -> None:
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"rate must be positive and finite, got {rate}")


def _scale_redline(line: Redline, rate: float) -> Redline:
    return Redline(
        offset_ms=line.offset_ms / rate,
        beat_length_ms=line.beat_length_ms / rate,
        meter=line.meter,
        source_index=line.source_index,
    )


def _mirror_note(note: ManiaNote) -> ManiaNote:
    return ManiaNote(
        note.time_ms,
        mirror_column(note.col),
        note.note_type,
        note.end_time_ms,
    )
