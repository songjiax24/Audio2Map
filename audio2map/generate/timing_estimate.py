"""Constant-BPM beat grid from audio (demo / generate when there is no ``.osu``).

Librosa ``beat_track`` defaults pull tempo toward 120 BPM (octave errors) and
treat the first detected beat as offset (STFT/flux lag). This module scores
canonical-range tempo candidates with a comb filter and fits phase to onsets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audio2map.features.audio.loader import load_mono_audio
from audio2map.features.audio.tick_features import AUDIO_SAMPLE_RATE
from audio2map.features.cond import canonical_bpm_norm_from_bpm
from audio2map.grid import canonicalize_bpm

HOP_LENGTH = 256
N_FFT = 2048
BPM_MIN = 70.0
BPM_MAX = 340.0

__all__ = ["estimate_bpm_offset", "estimate_from_audio"]


def estimate_bpm_offset(audio_path: str | Path) -> tuple[float, float, dict]:
    """Return ``(bpm, offset_ms, debug)`` for a constant-tempo beat grid."""
    path = Path(audio_path)
    y, sr = load_mono_audio(path, sample_rate=AUDIO_SAMPLE_RATE)
    return estimate_from_audio(y, sr)


def estimate_from_audio(y: np.ndarray, sr: int) -> tuple[float, float, dict]:
    """Same as :func:`estimate_bpm_offset` from mono PCM."""
    import librosa

    if y.size < sr // 4:
        raise ValueError("Audio is too short to estimate timing.")

    onset = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        aggregate=np.median,
    )
    onset = np.maximum(np.asarray(onset, dtype=np.float64), 0.0)
    if not np.any(onset > 0):
        raise ValueError("No beats detected. Please provide BPM/offset manually.")

    times = librosa.times_like(onset, sr=sr, hop_length=HOP_LENGTH)
    # Spectral flux is ~2 hops behind the energy rise.
    times = times - 2.0 * HOP_LENGTH / sr
    duration = float(len(y) / sr)

    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        onset_envelope=onset,
        hop_length=HOP_LENGTH,
        backtrack=True,
        units="time",
    )
    onsets = np.asarray(onsets, dtype=np.float64)

    raw = _tempo_peaks(onset, sr)
    candidates = _canonical_candidates(raw)
    if not candidates:
        raise ValueError("No beats detected. Please provide BPM/offset manually.")

    scored: list[tuple[float, float, float]] = []
    for bpm in candidates:
        score, hist_phase = _comb_fit(onset, times, bpm)
        scored.append((score, bpm, hist_phase))
    scored.sort(reverse=True)
    _best_score, bpm, hist_phase = scored[0]
    phase_s = _fit_phase(onsets, bpm, hist_phase)
    offset_s = _first_strong_beat(onset, times, bpm, phase_s, duration, onsets)
    offset_ms = float(offset_s * 1000.0)

    canonical_bpm, canonical_bpm_norm, scale_exp = canonical_bpm_norm_from_bpm(bpm)
    return bpm, offset_ms, {
        "method": "onset_comb_canonical",
        "num_beats": int(max(0, np.floor((duration - offset_s) * bpm / 60.0) + 1)),
        "canonical_bpm": canonical_bpm,
        "canonical_bpm_norm": canonical_bpm_norm,
        "bpm_scale_exp": scale_exp,
        "tempo_candidates": [round(b, 3) for _, b, _ in scored[:8]],
        "comb_score": float(_best_score),
    }


def _tempo_peaks(onset: np.ndarray, sr: int) -> list[float]:
    import librosa

    win_length = librosa.time_to_frames(8.0, sr=sr, hop_length=HOP_LENGTH).item()
    win_length = int(max(32, win_length))
    tg = librosa.feature.tempogram(
        onset_envelope=onset.astype(np.float32),
        sr=sr,
        hop_length=HOP_LENGTH,
        win_length=win_length,
        norm=None,
    )
    salience = np.mean(np.maximum(tg, 0.0), axis=-1)
    bpms = librosa.tempo_frequencies(len(salience), hop_length=HOP_LENGTH, sr=sr)

    peaks: list[tuple[float, float]] = []
    for i in range(1, len(salience) - 1):
        bpm = float(bpms[i])
        if not np.isfinite(bpm) or bpm < BPM_MIN or bpm > BPM_MAX:
            continue
        if salience[i] >= salience[i - 1] and salience[i] >= salience[i + 1]:
            peaks.append((float(salience[i]), bpm))
    peaks.sort(reverse=True)
    out = [b for _, b in peaks[:10]]

    # Wide-prior global estimate so 120 BPM is not a magnet.
    tempo = librosa.feature.tempo(
        onset_envelope=onset.astype(np.float32),
        sr=sr,
        hop_length=HOP_LENGTH,
        start_bpm=160.0,
        std_bpm=1.4,
        max_tempo=BPM_MAX,
        aggregate=np.median,
    )
    out.append(float(np.asarray(tempo).reshape(-1)[0]))
    return out


def _canonical_candidates(raw: list[float]) -> list[float]:
    folded: list[float] = []
    for bpm in raw:
        if bpm <= 0:
            continue
        for scale in (1.0, 2.0, 0.5, 1.5, 2.0 / 3.0, 3.0, 1.0 / 3.0):
            try:
                canon, _ = canonicalize_bpm(bpm * scale)
            except ValueError:
                continue
            if not any(abs(canon - existing) < 0.35 for existing in folded):
                folded.append(float(canon))
    return folded


def _comb_fit(onset: np.ndarray, times: np.ndarray, bpm: float) -> tuple[float, float]:
    period = 60.0 / bpm
    step = float(times[1] - times[0]) if len(times) > 1 else period / 48.0
    n_bins = int(np.clip(round(period / step), 24, 120))
    idx = np.floor(np.mod(times, period) / period * n_bins).astype(np.int64) % n_bins
    hist = np.bincount(idx, weights=onset, minlength=n_bins).astype(np.float64)
    count = np.bincount(idx, minlength=n_bins).astype(np.float64)
    hist /= np.maximum(count, 1.0)
    peak = int(np.argmax(hist))
    half = (peak + n_bins // 2) % n_bins
    score = float(hist[peak] - hist[half])
    phase_s = (peak + 0.5) / n_bins * period
    return score, phase_s


def _fit_phase(onsets: np.ndarray, bpm: float, hist_phase: float) -> float:
    """Prefer circular mean of backtracked onsets when they concentrate on the grid."""
    period = 60.0 / bpm
    if onsets.size < 8:
        return hist_phase
    theta = 2.0 * np.pi * np.mod(onsets, period) / period
    cos_m = float(np.mean(np.cos(theta)))
    sin_m = float(np.mean(np.sin(theta)))
    if float(np.hypot(cos_m, sin_m)) < 0.3:
        return hist_phase
    ang = float(np.arctan2(sin_m, cos_m)) % (2.0 * np.pi)
    return ang / (2.0 * np.pi) * period


def _first_strong_beat(
    onset: np.ndarray,
    times: np.ndarray,
    bpm: float,
    phase_s: float,
    duration: float,
    onsets: np.ndarray,
) -> float:
    period = 60.0 / bpm
    phase_s = float(np.mod(phase_s, period))
    if phase_s >= duration:
        return phase_s
    beats = phase_s + np.arange(0, int((duration - phase_s) / period) + 1) * period
    beats = beats[(beats >= 0.0) & (beats < duration)]
    if beats.size == 0:
        return phase_s
    vals = np.interp(beats, times, onset)
    peak = float(np.max(vals))
    thresh = 0.35 * peak if peak > 0 else 0.0
    chosen = float(beats[0])
    for beat, value in zip(beats, vals, strict=True):
        if value >= thresh:
            chosen = float(beat)
            break
    if onsets.size:
        nearest = onsets[int(np.argmin(np.abs(onsets - chosen)))]
        if abs(nearest - chosen) < min(0.04, 0.2 * period):
            chosen = float(nearest)
    return chosen
