"""Normalized log-mel on the v1 lattice. No stem convolution."""

from __future__ import annotations

import math

import numpy as np

from audio2map.features.audio.v1.constants import (
    DT_MS,
    FMIN_HZ,
    HOP,
    MEL_FLOOR,
    MEL_NORM_MU,
    MEL_NORM_SIGMA,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    FRAME_INDEX_SNAP,
)


def frame_index(time_ms: float) -> int:
    """``ceil(time_ms / Δt)``, snapping quotients within ``FRAME_INDEX_SNAP`` of an integer.

    The result is a mel-frame index. Beat coverage uses ``coverage_start_ms``.
    """
    quot = time_ms / DT_MS
    nearest = round(quot)
    if math.isfinite(quot) and abs(quot - nearest) <= FRAME_INDEX_SNAP:
        return int(nearest)
    return int(math.ceil(quot))


def buffer_sample_range(frame_lo: int, frame_hi: int) -> tuple[int, int]:
    """``[(2 frame_lo - 2) * hop - n_fft/2, 2 frame_hi * hop + n_fft/2)``."""
    start = (2 * frame_lo - 2) * HOP - N_FFT // 2
    end = 2 * frame_hi * HOP + N_FFT // 2
    return start, end


def slice_pcm(y: np.ndarray, start: int, end: int) -> np.ndarray:
    """Copy ``y[start:end]``. Indexes outside ``y`` stay 0."""
    out = np.zeros(end - start, dtype=np.float32)
    if end <= start:
        return out
    src_lo = max(start, 0)
    src_hi = min(end, int(y.shape[0]))
    if src_hi > src_lo:
        dst = src_lo - start
        out[dst : dst + (src_hi - src_lo)] = y[src_lo:src_hi]
    return out


def mel_power(y_buffer: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Linear mel power, shape ``(n_mels, frames)``, ``center=False``."""
    import librosa

    return librosa.feature.melspectrogram(
        y=np.asarray(y_buffer, dtype=np.float32),
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=FMIN_HZ,
        center=False,
        power=2.0,
    )


def normalize_mel(
    mel: np.ndarray,
    *,
    mu: float = MEL_NORM_MU,
    sigma: float = MEL_NORM_SIGMA,
) -> np.ndarray:
    """``(ln(mel + 1e-5) - μ) / σ``."""
    if sigma == 0.0:
        raise ValueError("mel norm sigma is 0")
    logged = np.log(np.asarray(mel, dtype=np.float64) + MEL_FLOOR)
    return ((logged - mu) / sigma).astype(np.float32)
