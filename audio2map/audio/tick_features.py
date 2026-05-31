"""Per-tick audio features aligned to the canonical BPM grid (v2)."""

from __future__ import annotations

import math

import numpy as np

V2_SAMPLE_RATE = 22_050
V2_N_MELS = 128
V2_N_CHROMA = 12
V2_FEATURE_DIM = V2_N_MELS + 1 + 1 + V2_N_CHROMA  # mel + onset + rms + chroma


def _frame_hop_ms(tick_ms: float) -> float:
    """Fine-enough hop for pooling into tick bins (~4 frames per tick)."""
    return max(tick_ms / 4.0, 1.0)


def compute_tick_grid_features(
    y: np.ndarray,
    sample_rate: int,
    *,
    offset_ms: float,
    tick_ms: float,
    tick_min: int,
    tick_max: int,
) -> np.ndarray:
    """Return ``(tick_max - tick_min, V2_FEATURE_DIM)`` float32 features."""
    import librosa

    if tick_max <= tick_min:
        return np.zeros((0, V2_FEATURE_DIM), dtype=np.float32)

    hop_ms = _frame_hop_ms(tick_ms)
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    n_fft = 2048

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=V2_N_MELS,
        fmin=20.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0].astype(np.float32)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop
    ).T.astype(np.float32)
    onset = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop).astype(np.float32)

    n_frames = log_mel.shape[0]
    num_ticks = tick_max - tick_min
    acc = np.zeros((num_ticks, V2_FEATURE_DIM), dtype=np.float64)
    counts = np.zeros(num_ticks, dtype=np.float64)

    for frame_idx in range(n_frames):
        frame_ms = frame_idx * hop_ms
        tick = math.floor((frame_ms - offset_ms) / tick_ms)
        if tick < tick_min or tick >= tick_max:
            continue
        row_idx = tick - tick_min
        acc[row_idx, :V2_N_MELS] += log_mel[frame_idx]
        acc[row_idx, V2_N_MELS] += float(onset[frame_idx]) if frame_idx < len(onset) else 0.0
        acc[row_idx, V2_N_MELS + 1] += float(rms[frame_idx]) if frame_idx < len(rms) else 0.0
        acc[row_idx, V2_N_MELS + 2 :] += chroma[frame_idx]
        counts[row_idx] += 1.0

    out = np.zeros((num_ticks, V2_FEATURE_DIM), dtype=np.float32)
    valid = counts > 0
    out[valid] = (acc[valid] / counts[valid, None]).astype(np.float32)
    return out
