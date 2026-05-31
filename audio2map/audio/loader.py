"""Load audio from beatmap set folders."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_SAMPLE_RATE = 44_100


def find_audio_file(set_dir: Path) -> Path:
    mp3 = list(set_dir.glob("*.mp3"))
    if len(mp3) != 1:
        raise FileNotFoundError(f"expected exactly one .mp3 in {set_dir}, found {len(mp3)}")
    return mp3[0]


def load_mono_audio(path: Path, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 in ``[-1, 1]``."""
    import librosa

    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    return y.astype(np.float32), sr


def audio_duration_ms(y: np.ndarray, sample_rate: int) -> int:
    return int(round(len(y) / sample_rate * 1000))
