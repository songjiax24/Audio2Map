"""Per-tick log-mel features aligned to the canonical BPM grid (v2).

**Active spec (v3):**

- log-mel only, ``n_mels=128``
- ``n_fft=1024``, ``hop_length=128`` @ 22.05 kHz (~5.80 ms/frame)
- STFT frames linearly interpolated onto tick starts (note onset time):
  ``tick_start_ms = offset_ms + tick * tick_ms``

Legacy grids on disk may use older specs (142-d mean-pool, etc.).
"""

from __future__ import annotations

import numpy as np

V2_SAMPLE_RATE = 22_050
V2_N_FFT = 1024
V2_HOP_LENGTH = 128
V2_N_MELS = 128
V2_FEATURE_DIM = V2_N_MELS

# ``power_to_db(..., ref=np.max)`` floor for out-of-grid padding (not 0 dB / song peak).
V2_LOG_MEL_FLOOR_DB = -80.0

AUDIO_FEATURE_SPEC_VERSION = 3


def frame_hop_ms(*, sample_rate: int = V2_SAMPLE_RATE, hop_length: int = V2_HOP_LENGTH) -> float:
    """Milliseconds between consecutive STFT frames."""
    return hop_length * 1000.0 / float(sample_rate)


def tick_start_ms(tick: int | np.ndarray, *, offset_ms: float, tick_ms: float) -> np.ndarray:
    """Return tick-start time(s) in milliseconds (note placement time on the grid)."""
    ticks = np.asarray(tick, dtype=np.float64)
    return offset_ms + ticks * tick_ms


def align_log_mel_to_ticks(
    log_mel: np.ndarray,
    tick_times_ms: np.ndarray,
    *,
    hop_ms: float,
) -> np.ndarray:
    """Map STFT log-mel frames onto tick-start times via linear interpolation."""
    n_frames = log_mel.shape[0]
    if n_frames == 0:
        return np.zeros((len(tick_times_ms), log_mel.shape[1]), dtype=np.float32)

    frame_times_ms = np.arange(n_frames, dtype=np.float64) * hop_ms
    out = np.empty((len(tick_times_ms), log_mel.shape[1]), dtype=np.float32)
    for band in range(log_mel.shape[1]):
        out[:, band] = np.interp(tick_times_ms, frame_times_ms, log_mel[:, band])
    return out


def compute_tick_grid_features(
    y: np.ndarray,
    sample_rate: int,
    *,
    offset_ms: float,
    tick_ms: float,
    tick_min: int,
    tick_max: int,
    n_fft: int = V2_N_FFT,
    hop_length: int = V2_HOP_LENGTH,
) -> np.ndarray:
    """Return ``(tick_max - tick_min, V2_N_MELS)`` float32 log-mel features."""
    import librosa

    if tick_max <= tick_min:
        return np.zeros((0, V2_FEATURE_DIM), dtype=np.float32)

    hop_ms = frame_hop_ms(sample_rate=sample_rate, hop_length=hop_length)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=V2_N_MELS,
        fmin=20.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float64)

    tick_times_ms = tick_start_ms(
        np.arange(tick_min, tick_max, dtype=np.float64),
        offset_ms=offset_ms,
        tick_ms=tick_ms,
    )
    return align_log_mel_to_ticks(log_mel, tick_times_ms, hop_ms=hop_ms)
