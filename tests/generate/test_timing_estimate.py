"""Synthetic-grid tests for the demo timing estimator."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from audio2map.features.audio.tick_features import AUDIO_SAMPLE_RATE
from audio2map.generate.timing_estimate import estimate_bpm_offset, estimate_from_audio
from audio2map.grid import canonicalize_bpm


def _wrap_ms(delta: float, period_ms: float) -> float:
    return float((delta + period_ms / 2.0) % period_ms - period_ms / 2.0)


def _click_track(bpm: float, offset_s: float, duration: float = 8.0) -> np.ndarray:
    import librosa

    sr = AUDIO_SAMPLE_RATE
    period = 60.0 / bpm
    times = offset_s + np.arange(0, int((duration - offset_s) / period) + 1) * period
    times = times[times < duration]
    return librosa.clicks(times=times, sr=sr, click_freq=1200, length=int(sr * duration))


def _write_wav(path: Path, y: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> None:
    pcm = np.clip(y * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(path), "w") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())


@pytest.mark.parametrize(
    ("bpm", "offset_s"),
    [
        (160.0, 0.40),
        (175.0, 1.234),
        (180.0, 0.50),
        (210.0, 0.355),
        (104.0, 0.980),
    ],
)
def test_estimate_recovers_click_grid(bpm: float, offset_s: float) -> None:
    y = _click_track(bpm, offset_s)
    est_bpm, est_off, debug = estimate_from_audio(y, AUDIO_SAMPLE_RATE)
    gt_canon, _ = canonicalize_bpm(bpm)
    est_canon, _ = canonicalize_bpm(est_bpm)
    assert est_canon == pytest.approx(gt_canon, rel=0.02)
    period_ms = 60_000.0 / gt_canon
    phase = _wrap_ms(est_off - offset_s * 1000.0, period_ms)
    assert abs(phase) < 20.0
    assert debug["method"] == "onset_comb_canonical"


def test_estimate_bpm_offset_loads_wav(tmp_path: Path) -> None:
    y = _click_track(180.0, 0.5)
    path = tmp_path / "click.wav"
    _write_wav(path, y)
    bpm, offset_ms, _ = estimate_bpm_offset(path)
    canon, _ = canonicalize_bpm(bpm)
    assert canon == pytest.approx(180.0, rel=0.02)
    phase = _wrap_ms(offset_ms - 500.0, 60_000.0 / 180.0)
    assert abs(phase) < 20.0
