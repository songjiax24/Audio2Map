"""Mel-spectrogram features aligned to the chart frame grid."""

from __future__ import annotations

import numpy as np

from audio2map.osu.events import DEFAULT_HOP_MS


def hop_samples(sample_rate: int, hop_ms: int = DEFAULT_HOP_MS) -> int:
    return int(round(sample_rate * hop_ms / 1000))


def compute_mel(
    y: np.ndarray,
    sample_rate: int,
    *,
    hop_ms: int = DEFAULT_HOP_MS,
    n_fft: int = 2048,
    n_mels: int = 80,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Return log-mel of shape ``(T, n_mels)`` with one frame per ``hop_ms``."""
    import librosa

    hop = hop_samples(sample_rate, hop_ms)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.T.astype(np.float32)


def align_num_frames(
    mel_frames: int,
    chart_frames: int,
) -> int:
    """Length shared by mel and chart (pad the shorter side during export)."""
    return max(mel_frames, chart_frames, 1)
