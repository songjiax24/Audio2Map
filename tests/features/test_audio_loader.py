"""Audio loader: libsndfile first, ffmpeg only when that decoder fails."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from audio2map.features.audio.loader import load_mono_audio
from audio2map.features.audio.tick_features import AUDIO_SAMPLE_RATE


def _write_tone(path: Path, *, seconds: float = 0.2, rate: int = 44100) -> None:
    n = int(rate * seconds)
    samples = (0.2 * np.sin(2 * np.pi * 440 * np.arange(n) / rate) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(samples.tobytes())


def test_primary_path_does_not_call_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "tone.wav"
    _write_tone(src)

    def fail_ffmpeg(*_args, **_kwargs):
        raise AssertionError("ffmpeg should not run when libsndfile can read the file")

    monkeypatch.setattr("audio2map.features.audio.loader.subprocess.run", fail_ffmpeg)
    y, sr = load_mono_audio(src, sample_rate=AUDIO_SAMPLE_RATE)
    assert sr == AUDIO_SAMPLE_RATE
    assert y.ndim == 1
    assert y.dtype == np.float32
    assert len(y) > 0


def test_libsndfile_failure_falls_back_to_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "tone.wav"
    _write_tone(src)
    import librosa

    real_load = librosa.load

    def load(path, sr=None, mono=True):
        if Path(path) == src:
            raise type("LibsndfileError", (RuntimeError,), {})("Unspecified internal error.")
        return real_load(path, sr=sr, mono=mono)

    monkeypatch.setattr(librosa, "load", load)
    y, sr = load_mono_audio(src, sample_rate=AUDIO_SAMPLE_RATE)
    assert sr == AUDIO_SAMPLE_RATE
    assert y.ndim == 1
    assert y.dtype == np.float32
    assert len(y) > 0
