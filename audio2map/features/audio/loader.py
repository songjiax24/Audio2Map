"""Load audio as mono PCM."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audio2map.features.audio.tick_features import AUDIO_SAMPLE_RATE


def load_mono_audio(path: Path, *, sample_rate: int = AUDIO_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 in ``[-1, 1]``."""
    import librosa

    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    return y.astype(np.float32), sr


def audio_duration_ms(y: np.ndarray, sample_rate: int) -> int:
    return int(round(len(y) / sample_rate * 1000))
