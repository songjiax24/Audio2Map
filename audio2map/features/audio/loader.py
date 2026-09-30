"""Load audio as mono PCM."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from audio2map.features.audio.tick_features import AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)


def load_mono_audio(path: Path, *, sample_rate: int = AUDIO_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 in ``[-1, 1]``.

    librosa via libsndfile is the decoder for files it can open. When libsndfile
    rejects a file, ffmpeg writes a temporary wav and librosa resamples that wav.
    Files that already decode do not go through ffmpeg.
    """
    import librosa

    try:
        y, sr = librosa.load(path, sr=sample_rate, mono=True)
    except Exception as exc:
        if type(exc).__name__ != "LibsndfileError":
            raise
        logger.warning("libsndfile failed on %s; decoding with ffmpeg", path)
        y, sr = _load_via_ffmpeg(path, sample_rate=sample_rate)
    return np.asarray(y, dtype=np.float32), int(sr)


def _load_via_ffmpeg(path: Path, *, sample_rate: int) -> tuple[np.ndarray, int]:
    import librosa

    with tempfile.TemporaryDirectory(prefix="audio2map-decode-") as tmp:
        wav_path = Path(tmp) / "decoded.wav"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-f", "wav", str(wav_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not wav_path.is_file():
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"ffmpeg could not decode {path}: {detail}") from None
        y, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
    return y, sr


def audio_duration_ms(y: np.ndarray, sample_rate: int) -> int:
    return int(round(len(y) / sample_rate * 1000))
