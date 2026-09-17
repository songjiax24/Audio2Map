"""Tests for tick log-mel and audio grid cache."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from audio2map.features.audio.tick_features import (
    AUDIO_FEATURE_DIM,
    AUDIO_HOP_LENGTH,
    AUDIO_LOG_MEL_FLOOR_DB,
    AUDIO_LOG_MEL_TOP_DB,
    AUDIO_N_FFT,
    AUDIO_N_MELS,
    compute_tick_grid_features,
    frame_hop_ms,
)
from audio2map.features.audio.grid import (
    AudioGridMeta,
    grid_cache_paths,
    is_valid_audio_grid_cache,
    iter_set_timing_jobs,
    load_audio_grid,
    save_audio_grid,
    slice_audio_window,
)


def _grid_meta(**overrides: Any) -> AudioGridMeta:
    fields: dict[str, Any] = {
        "tick_min": 0,
        "tick_max": 5,
        "offset_ms": 0,
        "canonical_bpm": 180.0,
        "audio_hash": "deadbeef",
    }
    fields.update(overrides)
    return AudioGridMeta(**fields)


def test_slice_audio_window_negative_tick_min() -> None:
    meta = _grid_meta(tick_min=-100, tick_max=200, offset_ms=980)
    grid = np.arange(meta.num_ticks * AUDIO_FEATURE_DIM, dtype=np.float32).reshape(
        meta.num_ticks, AUDIO_FEATURE_DIM
    )
    out, info = slice_audio_window(grid, meta, -1, 0)
    assert out.shape == (192, AUDIO_FEATURE_DIM)
    assert info["padded_ticks_start"] > 0
    pad = info["padded_ticks_start"]
    np.testing.assert_allclose(out[:pad], AUDIO_LOG_MEL_FLOOR_DB)
    assert not np.all(out[pad:] == AUDIO_LOG_MEL_FLOOR_DB)


def test_tick_features_shape() -> None:
    sr = 22050
    y = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    feats = compute_tick_grid_features(
        y,
        sr,
        offset_ms=100.0,
        tick_ms=10.0,
        tick_min=-5,
        tick_max=10,
    )
    assert feats.shape == (15, AUDIO_FEATURE_DIM)
    assert feats.dtype == np.float32


def test_tick_features_spec_constants() -> None:
    from audio2map.features.audio.tick_features import AUDIO_FEATURE_SPEC_VERSION

    assert AUDIO_N_FFT == 1024
    assert AUDIO_HOP_LENGTH == 128
    assert AUDIO_N_MELS == 128
    assert AUDIO_FEATURE_DIM == 128
    assert AUDIO_FEATURE_SPEC_VERSION == 3
    assert AUDIO_LOG_MEL_FLOOR_DB == -AUDIO_LOG_MEL_TOP_DB
    assert frame_hop_ms() == pytest.approx(128 * 1000.0 / 22050.0)


def test_tick_features_matches_manual_linear_interp() -> None:
    sr = 22050
    duration_s = 4.0
    y = np.sin(2 * np.pi * 440 * np.arange(int(sr * duration_s)) / sr).astype(np.float32)
    offset_ms = 50.0
    tick_ms = 12.5
    tick_min, tick_max = 0, 8

    feats = compute_tick_grid_features(
        y,
        sr,
        offset_ms=offset_ms,
        tick_ms=tick_ms,
        tick_min=tick_min,
        tick_max=tick_max,
    )

    import librosa

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=AUDIO_N_FFT,
        hop_length=AUDIO_HOP_LENGTH,
        n_mels=AUDIO_N_MELS,
        fmin=20.0,
    )
    log_mel = librosa.power_to_db(
        mel, ref=np.max, top_db=AUDIO_LOG_MEL_TOP_DB, axes=None
    ).T.astype(np.float64)
    hop_ms = frame_hop_ms(sample_rate=sr, hop_length=AUDIO_HOP_LENGTH)
    frame_times_ms = np.arange(log_mel.shape[0], dtype=np.float64) * hop_ms
    tick_times_ms = offset_ms + np.arange(tick_min, tick_max, dtype=np.float64) * tick_ms

    expected = np.empty_like(feats, dtype=np.float64)
    for band in range(AUDIO_N_MELS):
        expected[:, band] = np.interp(tick_times_ms, frame_times_ms, log_mel[:, band])
    np.testing.assert_allclose(feats, expected.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_is_valid_audio_grid_cache(tmp_path) -> None:
    meta = _grid_meta(tick_min=-5, tick_max=10, offset_ms=100, audio_hash="abc123")
    features = np.zeros((meta.num_ticks, AUDIO_FEATURE_DIM), dtype=np.float32)
    npy, json_path = save_audio_grid(features, meta, tmp_path)
    assert is_valid_audio_grid_cache(npy, json_path)

    npy.write_bytes(b"partial")
    assert not is_valid_audio_grid_cache(npy, json_path)

    json_path.unlink()
    assert not is_valid_audio_grid_cache(npy, json_path)


def test_audio_grid_meta_from_dict_requires_spec() -> None:
    payload = {
        "tick_min": 0,
        "tick_max": 5,
        "offset_ms": 0,
        "canonical_bpm": 180.0,
        "audio_hash": "deadbeef",
    }
    with pytest.raises(ValueError, match="missing feature_spec_version"):
        AudioGridMeta.from_dict(payload)
    payload["feature_spec_version"] = 3
    payload["n_fft"] = 999
    payload["audio_path"] = "/tmp/x.mp3"
    meta = AudioGridMeta.from_dict(payload)
    assert meta.feature_spec_version == 3
    assert meta.audio_hash == "deadbeef"


def test_audio_grid_meta_bar_range() -> None:
    meta = _grid_meta(tick_min=-100, tick_max=200)
    assert meta.audio_bar_range == (-1, 2)


def test_load_audio_grid_rejects_invalid(tmp_path) -> None:
    meta = _grid_meta()
    features = np.zeros((meta.num_ticks, AUDIO_FEATURE_DIM), dtype=np.float32)
    save_audio_grid(features, meta, tmp_path)
    stem = "deadbeef_bpm180.000_offset0.0"
    npy, _ = grid_cache_paths(tmp_path, stem)
    npy.write_bytes(b"truncated")
    with pytest.raises(FileNotFoundError):
        load_audio_grid(tmp_path, stem)


_ELIGIBLE_OSU = """osu file format v14

[General]
AudioFilename: song.mp3
Mode: 3

[Metadata]
Title: Grid
Artist: Test
Creator: Test
Version: 4K
BeatmapID: 1
BeatmapSetID: 1

[Difficulty]
CircleSize: 4

[TimingPoints]
0,333.333333333333,4,2,0,50,1,0

[HitObjects]
64,192,0,1,0,0:0:0:0:
192,192,333,1,0,0:0:0:0:
"""


def test_iter_set_timing_jobs_logs_missing_audio(tmp_path, caplog) -> None:
    from audio2map.dataset.filter import check_osu_path

    set_dir = tmp_path / "1 set"
    set_dir.mkdir()
    (set_dir / "chart.osu").write_text(_ELIGIBLE_OSU, encoding="utf-8")

    with caplog.at_level("WARNING"):
        scan = iter_set_timing_jobs(
            tmp_path, is_eligible=lambda p: check_osu_path(p).eligible
        )
    assert scan.jobs == []
    assert scan.enum_failed == 1
    assert "skip grid job" in caplog.text


def test_iter_set_timing_jobs_skips_ineligible_silently(tmp_path, caplog) -> None:
    set_dir = tmp_path / "1 set"
    set_dir.mkdir()
    (set_dir / "chart.osu").write_text(_ELIGIBLE_OSU, encoding="utf-8")

    with caplog.at_level("WARNING"):
        scan = iter_set_timing_jobs(tmp_path, is_eligible=lambda _p: False)
    assert scan.jobs == []
    assert scan.enum_failed == 0
    assert caplog.text == ""


def test_iter_set_timing_jobs_unique_stem(tmp_path) -> None:
    from audio2map.dataset.filter import check_osu_path

    set_dir = tmp_path / "1 set"
    set_dir.mkdir()
    (set_dir / "a.osu").write_text(_ELIGIBLE_OSU, encoding="utf-8")
    (set_dir / "b.osu").write_text(_ELIGIBLE_OSU.replace("Version: 4K", "Version: 4K Extra"), encoding="utf-8")
    (set_dir / "song.mp3").write_bytes(b"fake-audio")

    scan = iter_set_timing_jobs(
        tmp_path, is_eligible=lambda p: check_osu_path(p).eligible
    )
    assert scan.enum_failed == 0
    assert len(scan.jobs) == 1
    assert scan.jobs[0][0] == set_dir
    assert scan.jobs[0][3] == set_dir / "song.mp3"
