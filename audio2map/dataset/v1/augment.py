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
from audio2map.grid.v1.beatgrid import build_beatgrid, scale_beatgrid
from audio2map.osu.schema import ManiaNote
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1.codec import EncodeResult, encode_chart
from audio2map.tokens.v1.labels import LabelBatch
from audio2map.tokens.v1.store import cached_rate1_tokens

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


def mirror_tokens(tokens: list[str]) -> list[str]:
    """Swap lanes inside each ``<ROW_>``. ``<POS>``, ``<BEAT>``, and ``<EOS>`` stay put.

    Column ``0`` trades with ``3`` and ``1`` with ``2``, which reverses the four
    lane digits. That is the token string of the mirrored chart at every rate.
    """
    mirrored: list[str] = []
    for token in tokens:
        if (
            token.startswith("<ROW_")
            and token.endswith(">")
            and len(token) == 10
            and token[5:9].isdigit()
        ):
            mirrored.append("<ROW_" + token[5:9][::-1] + ">")
        else:
            mirrored.append(token)
    return mirrored


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


def _blank_labels() -> LabelBatch:
    empty_i16 = np.zeros(0, dtype=np.int16)
    empty_i32 = np.zeros(0, dtype=np.int32)
    return LabelBatch(
        numer=empty_i16,
        denom=empty_i16,
        time_ms=np.zeros(0, dtype=np.float64),
        section=empty_i32,
        beat_k=empty_i32,
        beat_id=empty_i32,
        global_b=empty_i32,
        from_window=np.zeros(0, dtype=bool),
    )


def _encode_rate1(
    notes: list[ManiaNote],
    redlines: tuple[Redline, ...],
    t_audio_ms: float,
    mirror: bool,
    relpath: str | None,
) -> EncodeResult:
    """Use the stored rate-1 string when the audio duration matches."""
    if relpath is not None:
        cached = cached_rate1_tokens(relpath, t_audio_ms)
        if cached is not None:
            tokens = mirror_tokens(cached) if mirror else list(cached)
            grid = build_beatgrid(redlines, t_audio_ms)
            return EncodeResult(
                tokens=tokens,
                grid=grid,
                labels=_blank_labels(),
                endpoints=[],
                near_duplicate_pairs=grid.near_duplicate_pairs,
            )
    return encode_chart(notes, redlines, t_audio_ms)


def clock_chart(
    notes: list[ManiaNote],
    redlines: tuple[Redline, ...],
    pcm: np.ndarray,
    t_audio_ms: float,
    choice: AugmentChoice,
    relpath: str | None = None,
) -> ClockChart:
    """Label at rate 1, then scale the grid and the audio clock together.

    Mirror is applied before labeling, so a mirrored chart keeps that token
    string at every rate. The returned notes still carry the original
    millisecond times. A cached rate-1 string is mirrored by reversing
    ``<ROW_>`` digits instead of labeling again.
    """
    mirrored = [_mirror_note(note) for note in notes] if choice.mirror else list(notes)
    encoded = _encode_rate1(mirrored, redlines, t_audio_ms, choice.mirror, relpath)
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
