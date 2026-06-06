"""Tests for v2 cond_vec, audio_grid, and window sampling."""

from __future__ import annotations

import random

import pytest
import numpy as np

from audio2map.audio.tick_features import V2_FEATURE_DIM, V2_HOP_LENGTH, V2_LOG_MEL_FLOOR_DB, V2_N_FFT, V2_N_MELS, compute_tick_grid_features, frame_hop_ms
from audio2map.data.audio_grid import (
    AudioGridMeta,
    is_valid_audio_grid_cache,
    save_audio_grid,
    slice_audio_window,
)
from audio2map.data.cond_vec import COND_VEC_DIM, CondVecError, build_cond_vec
from audio2map.difficulty.chart_meta import ChartMeta
from audio2map.data.window_sampler import (
    build_loss_mask,
    sample_window_bars,
)
from audio2map.utils.paths import audio_grid_dir, raw_dir
from audio2map.osu.row_tokens import TOKEN_BOS, TOKEN_EOS, row_state_to_token


def test_build_cond_vec_dim() -> None:
    meta = ChartMeta(
        osu_path="x.osu",
        set_id=1,
        beatmap_id=2,
        version="Normal",
        constant_bpm=True,
        bpm=180.0,
        original_bpm=180.0,
        canonical_bpm=180.0,
        bpm_scale_exp=0,
        offset_ms=980,
        meter=4,
        note_count=100,
        hold_ratio=0.2,
        hold_coverage=0.15,
        official_sr=3.5,
        msd_overall=20.0,
        msd_stream=18.0,
        msd_jumpstream=12.0,
        msd_handstream=10.0,
        msd_stamina=8.0,
        msd_jack_speed=6.0,
        msd_chordjack=4.0,
        msd_technical=5.0,
        analyzer_ln_percent=0.3,
        analyzer_hb_row_ratio=0.1,
        analyzer_stream=0.5,
        analyzer_chordstream=0.2,
        analyzer_jacks=0.1,
        analyzer_coordination=0.05,
        analyzer_density=0.1,
        analyzer_wildcard=0.05,
        analyzer_available=1,
        msd_available=1,
    )
    v = build_cond_vec(meta)
    assert v.shape == (COND_VEC_DIM,)
    assert COND_VEC_DIM == 18
    assert v[0] == 0.35  # 3.5/10
    assert v[1] == 0.3  # analyzer_ln_percent
    assert v[9] == 0.5  # msd_overall 20/40
    assert v[17] == 0.5  # (180-120)/120


def test_build_cond_vec_raises_on_meta_error() -> None:
    meta = ChartMeta(
        osu_path="bad.osu",
        set_id=1,
        beatmap_id=2,
        version="Normal",
        constant_bpm=True,
        bpm=180.0,
        original_bpm=180.0,
        canonical_bpm=180.0,
        bpm_scale_exp=0,
        offset_ms=980,
        meter=4,
        note_count=100,
        hold_ratio=0.2,
        hold_coverage=0.15,
        official_sr=3.5,
        msd_overall=20.0,
        msd_stream=18.0,
        msd_jumpstream=12.0,
        msd_handstream=10.0,
        msd_stamina=8.0,
        msd_jack_speed=6.0,
        msd_chordjack=4.0,
        msd_technical=5.0,
        analyzer_ln_percent=0.3,
        analyzer_hb_row_ratio=0.1,
        analyzer_stream=0.5,
        analyzer_chordstream=0.2,
        analyzer_jacks=0.1,
        analyzer_coordination=0.05,
        analyzer_density=0.1,
        analyzer_wildcard=0.05,
        error="msd: runner failed",
    )
    with pytest.raises(CondVecError, match="msd: runner failed"):
        build_cond_vec(meta)


def test_build_cond_vec_raises_on_missing_field() -> None:
    meta = ChartMeta(
        osu_path="bad.osu",
        set_id=1,
        beatmap_id=2,
        version="Normal",
        constant_bpm=True,
        bpm=180.0,
        original_bpm=180.0,
        canonical_bpm=180.0,
        bpm_scale_exp=0,
        offset_ms=980,
        meter=4,
        note_count=100,
        hold_ratio=0.2,
        hold_coverage=0.15,
        official_sr=None,
        msd_overall=20.0,
        msd_stream=18.0,
        msd_jumpstream=12.0,
        msd_handstream=10.0,
        msd_stamina=8.0,
        msd_jack_speed=6.0,
        msd_chordjack=4.0,
        msd_technical=5.0,
        analyzer_ln_percent=0.3,
        analyzer_hb_row_ratio=0.1,
        analyzer_stream=0.5,
        analyzer_chordstream=0.2,
        analyzer_jacks=0.1,
        analyzer_coordination=0.05,
        analyzer_density=0.1,
        analyzer_wildcard=0.05,
    )
    with pytest.raises(CondVecError, match="missing official_sr"):
        build_cond_vec(meta)


def test_build_loss_mask() -> None:
    tokens = [
        TOKEN_BOS,
        row_state_to_token((0, 3, 0, 0)),
        "<BAR>",
        "<POS_0>",
        row_state_to_token((1, 0, 0, 0)),
        TOKEN_EOS,
    ]
    assert build_loss_mask(tokens) == [0, 0, 1, 1, 1, 1]


def test_slice_audio_window_negative_tick_min() -> None:
    meta = AudioGridMeta(
        tick_min=-100,
        tick_max=200,
        offset_ms=980.0,
        canonical_bpm=180.0,
        ticks_per_beat=48,
        feature_dim=V2_FEATURE_DIM,
        sample_rate=22050,
        audio_hash="abc",
        audio_path="/tmp/x.mp3",
        duration_ms=120000,
    )
    grid = np.arange(meta.num_ticks * V2_FEATURE_DIM, dtype=np.float32).reshape(
        meta.num_ticks, V2_FEATURE_DIM
    )
    out, info = slice_audio_window(grid, meta, -1, 0)
    assert out.shape == (192, V2_FEATURE_DIM)
    assert info["padded_ticks_start"] > 0
    pad = info["padded_ticks_start"]
    np.testing.assert_allclose(out[:pad], V2_LOG_MEL_FLOOR_DB)
    assert not np.all(out[pad:] == V2_LOG_MEL_FLOOR_DB)


def test_sample_window_bars_allows_negative_start() -> None:
    from audio2map.data.window_sampler import assert_window_start_bar_aligned, train_bar_range

    train_start, train_end = train_bar_range(
        audio_start_bar=-5,
        audio_end_bar=50,
    )
    assert train_start == -5
    assert train_end == 50
    rng = random.Random(0)
    result = sample_window_bars(
        audio_start_bar=-5,
        audio_end_bar=50,
        rng=rng,
    )
    assert result is not None
    start, end = result
    assert end - start == 16
    assert train_start <= start < train_end
    assert_window_start_bar_aligned(start)
    assert_window_start_bar_aligned(-74)


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
    assert feats.shape == (15, V2_FEATURE_DIM)
    assert feats.dtype == np.float32


def test_tick_features_spec_constants() -> None:
    from audio2map.audio.tick_features import (
        AUDIO_FEATURE_SPEC_VERSION,
        V2_HOP_LENGTH,
        V2_N_FFT,
        V2_N_MELS,
    )

    assert V2_N_FFT == 1024
    assert V2_HOP_LENGTH == 128
    assert V2_N_MELS == 128
    assert V2_FEATURE_DIM == 128
    assert AUDIO_FEATURE_SPEC_VERSION == 3
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
        n_fft=V2_N_FFT,
        hop_length=V2_HOP_LENGTH,
        n_mels=V2_N_MELS,
        fmin=20.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float64)
    hop_ms = frame_hop_ms(sample_rate=sr, hop_length=V2_HOP_LENGTH)
    frame_times_ms = np.arange(log_mel.shape[0], dtype=np.float64) * hop_ms
    tick_times_ms = offset_ms + np.arange(tick_min, tick_max, dtype=np.float64) * tick_ms

    expected = np.empty_like(feats, dtype=np.float64)
    for band in range(V2_N_MELS):
        expected[:, band] = np.interp(tick_times_ms, frame_times_ms, log_mel[:, band])
    np.testing.assert_allclose(feats, expected.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_is_valid_audio_grid_cache(tmp_path) -> None:
    meta = AudioGridMeta(
        tick_min=-5,
        tick_max=10,
        offset_ms=100.0,
        canonical_bpm=180.0,
        ticks_per_beat=48,
        feature_dim=V2_FEATURE_DIM,
        sample_rate=22050,
        audio_hash="abc123",
        audio_path="/tmp/x.mp3",
        duration_ms=5000,
    )
    features = np.zeros((meta.num_ticks, V2_FEATURE_DIM), dtype=np.float32)
    npy, json_path = save_audio_grid(features, meta, tmp_path)
    assert is_valid_audio_grid_cache(npy, json_path)

    npy.write_bytes(b"partial")
    assert not is_valid_audio_grid_cache(npy, json_path)

    json_path.unlink()
    assert not is_valid_audio_grid_cache(npy, json_path)


def test_load_audio_grid_rejects_invalid(tmp_path) -> None:
    from audio2map.data.audio_grid import grid_cache_paths, load_audio_grid

    meta = AudioGridMeta(
        tick_min=0,
        tick_max=5,
        offset_ms=0.0,
        canonical_bpm=180.0,
        ticks_per_beat=48,
        feature_dim=V2_FEATURE_DIM,
        sample_rate=22050,
        audio_hash="deadbeef",
        audio_path="/tmp/x.mp3",
        duration_ms=1000,
    )
    features = np.zeros((meta.num_ticks, V2_FEATURE_DIM), dtype=np.float32)
    save_audio_grid(features, meta, tmp_path)
    stem = "deadbeef_bpm180.000_offset0.0"
    npy, _ = grid_cache_paths(tmp_path, stem)
    npy.write_bytes(b"truncated")
    try:
        load_audio_grid(tmp_path, stem)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


@pytest.mark.skipif(not raw_dir().is_dir(), reason="no dataset")
def test_preload_chart_bundles_share_audio_grid() -> None:
    from audio2map.data.chart_bundle import (
        SharedGridCache,
        has_audio_grid,
        preload_chart_bundle,
    )
    from audio2map.utils.paths import raw_dir

    root = raw_dir()
    pair: list[Path] = []
    for set_dir in root.iterdir():
        if not set_dir.is_dir():
            continue
        charts = [p for p in set_dir.glob("*.osu") if has_audio_grid(p)]
        if len(charts) >= 2:
            pair = charts[:2]
            break
    if len(pair) < 2:
        pytest.skip("no set with two grid-backed charts")

    cache = SharedGridCache(audio_grid_dir())
    b0 = preload_chart_bundle(pair[0], grid_cache=cache, require_grid=True)
    b1 = preload_chart_bundle(pair[1], grid_cache=cache, require_grid=True)
    assert b0 is not None and b1 is not None
    assert np.shares_memory(b0.grid, b1.grid)
    assert len(cache) == 1
