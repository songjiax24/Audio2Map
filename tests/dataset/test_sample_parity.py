"""Parity: ``build_sample(Path)`` ≡ ``build_sample(ChartBundle)`` on one window."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audio2map.features.audio.tick_features import AUDIO_FEATURE_DIM
from audio2map.features.cond import ChartMeta
from audio2map.grid import TICKS_PER_BAR

MINIMAL_OSU = """osu file format v14

[General]
AudioFilename: song.mp3
Mode: 3

[Metadata]
Title: Parity
Artist: Test
Creator: Test
Version: 4K Normal
BeatmapID: 2
BeatmapSetID: 1

[Difficulty]
HPDrainRate: 7
CircleSize: 4
OverallDifficulty: 8

[TimingPoints]
980,333.333333333333,4,2,0,50,1,0

[HitObjects]
64,192,980,1,0,0:0:0:0:
192,192,1313,1,0,0:0:0:0:
320,192,1646,1,0,0:0:0:0:
448,192,1980,128,0,2313:0:0:0:0:
"""


def _chart_meta(osu_path: Path) -> ChartMeta:
    return ChartMeta(
        osu_path=str(osu_path),
        beatmap_id=2,
        canonical_bpm=180.0,
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
    )


def test_build_sample_path_matches_bundle(tmp_path: Path) -> None:
    from audio2map.dataset.bundle import ChartBundle
    from audio2map.dataset.sample import build_sample
    from audio2map.features.audio.grid import (
        AudioGridMeta,
        audio_content_hash,
        save_audio_grid,
    )
    from audio2map.features.cond import build_cond_vec
    from audio2map.osu.parser import parse_beatmap
    from audio2map.grid import CanonicalTiming

    set_dir = tmp_path / "1 Parity Set"
    set_dir.mkdir()
    osu_path = set_dir / "chart.osu"
    osu_path.write_text(MINIMAL_OSU, encoding="utf-8")
    audio_path = set_dir / "song.mp3"
    audio_path.write_bytes(b"parity-fake-audio")

    beatmap = parse_beatmap(osu_path)
    timing = CanonicalTiming.from_beatmap(beatmap)

    n_ticks = 16 * TICKS_PER_BAR
    grid = np.arange(n_ticks * AUDIO_FEATURE_DIM, dtype=np.float32).reshape(
        n_ticks, AUDIO_FEATURE_DIM
    )
    grid_meta = AudioGridMeta(
        tick_min=0,
        tick_max=n_ticks,
        offset_ms=timing.offset_ms,
        canonical_bpm=timing.canonical_bpm,
        audio_hash=audio_content_hash(audio_path),
    )
    grid_dir = tmp_path / "grids"
    save_audio_grid(grid, grid_meta, grid_dir)

    meta = _chart_meta(osu_path)
    bundle = ChartBundle(
        path=osu_path,
        beatmap=beatmap,
        timing=timing,
        cond_vec=build_cond_vec(meta),
        grid=grid,
        grid_meta=grid_meta,
    )

    from_bundle = build_sample(bundle, start_bar=0)
    from_path = build_sample(
        osu_path,
        grid_dir=grid_dir,
        chart_meta=meta,
        build_grid_if_missing=False,
        start_bar=0,
    )

    assert from_bundle is not None
    assert from_path is not None
    assert from_bundle.tokens == from_path.tokens
    assert from_bundle.token_ids == from_path.token_ids
    assert from_bundle.loss_mask == from_path.loss_mask
    assert from_bundle.window == from_path.window
    assert from_bundle.beatmap_id == from_path.beatmap_id == 2
    assert from_bundle.window.start_bar == 0
    np.testing.assert_array_equal(from_bundle.cond_vec, from_path.cond_vec)
    np.testing.assert_array_equal(from_bundle.audio_features, from_path.audio_features)


def test_preload_skips_audio_shorter_than_training_window(tmp_path: Path) -> None:
    from audio2map.dataset.bundle import preload_chart_bundle
    from audio2map.features.audio.grid import AudioGridMeta, audio_content_hash, save_audio_grid
    from audio2map.osu.parser import parse_beatmap
    from audio2map.grid import CanonicalTiming

    set_dir = tmp_path / "1 Short Set"
    set_dir.mkdir()
    osu_path = set_dir / "chart.osu"
    osu_path.write_text(MINIMAL_OSU, encoding="utf-8")
    audio_path = set_dir / "song.mp3"
    audio_path.write_bytes(b"short-fake-audio")

    beatmap = parse_beatmap(osu_path)
    timing = CanonicalTiming.from_beatmap(beatmap)
    n_ticks = 8 * TICKS_PER_BAR
    grid = np.zeros((n_ticks, AUDIO_FEATURE_DIM), dtype=np.float32)
    grid_meta = AudioGridMeta(
        tick_min=0,
        tick_max=n_ticks,
        offset_ms=timing.offset_ms,
        canonical_bpm=timing.canonical_bpm,
        audio_hash=audio_content_hash(audio_path),
    )
    grid_dir = tmp_path / "grids"
    save_audio_grid(grid, grid_meta, grid_dir)

    assert (
        preload_chart_bundle(
            osu_path,
            grid_dir=grid_dir,
            chart_meta=_chart_meta(osu_path),
            build_grid_if_missing=False,
        )
        is None
    )
