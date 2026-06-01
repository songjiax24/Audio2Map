"""Tests for v2 cond_vec, audio_grid, and window sampling."""

from __future__ import annotations

import random

import numpy as np

from audio2map.audio.tick_features import V2_FEATURE_DIM, compute_tick_grid_features
from audio2map.data.audio_grid import (
    AudioGridMeta,
    is_valid_audio_grid_cache,
    save_audio_grid,
    slice_audio_window,
)
from audio2map.data.cond_vec import COND_VEC_DIM, build_cond_vec
from audio2map.data.window_sampler import (
    WindowSamplingConfig,
    build_loss_mask,
    inference_bar_windows,
    sample_window_bars,
)
from audio2map.difficulty.chart_meta import ChartMeta
from audio2map.osu.row_tokens import TOKEN_BOS, TOKEN_EOS, row_state_to_token


def test_build_cond_vec_dim_and_flags() -> None:
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
    assert v[0] == 0.35  # 3.5/10
    assert v[11] == 0.5  # msd_overall 20/40
    assert v[21] == 1.0  # analyzer_available
    assert v[22] == 1.0  # msd_available
    assert v[19] == 0.5  # (180-120)/120
    assert v[20] == 0.0  # bpm_scale_exp


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


def test_sample_window_bars_allows_negative_start() -> None:
    cfg = WindowSamplingConfig(
        window_bars=4,
        window_bars_choices=(4,),
        pre_event_margin_bars=2,
        post_event_margin_bars=2,
    )
    train_start, train_end = __import__(
        "audio2map.data.window_sampler", fromlist=["train_bar_range"]
    ).train_bar_range(
        chart_start_bar=-1,
        chart_end_bar=10,
        audio_start_bar=-5,
        audio_end_bar=50,
        cfg=cfg,
    )
    assert train_start <= -1
    rng = random.Random(0)
    result = sample_window_bars(
        chart_start_bar=-1,
        chart_end_bar=10,
        audio_start_bar=-5,
        audio_end_bar=50,
        cfg=cfg,
        rng=rng,
    )
    assert result is not None
    start, end = result
    assert end - start == 4
    assert train_start <= start < train_end


def test_inference_bar_windows() -> None:
    windows = inference_bar_windows(-2, 6, window_bars=4)
    assert windows[0][0] == -2
    assert windows[-1][1] == 6


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
