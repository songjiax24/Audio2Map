"""Cached per-chart data for fast training sample construction."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from audio2map.data.audio_grid import (
    AudioGridMeta,
    grid_cache_paths,
    load_audio_grid,
    resolve_grid_stem,
    slice_audio_window,
)
from audio2map.data.v2_dataset import V2TrainingSample
from audio2map.data.chart_filter import check_beatmap_eligibility
from audio2map.data.cond_vec import build_cond_vec
from audio2map.data.window_sampler import (
    WindowSamplingConfig,
    audio_bar_range_from_duration,
    build_loss_mask,
    chart_bar_range,
    sample_window_bars,
)
from audio2map.difficulty.chart_meta import ChartMeta, compute_chart_meta
from audio2map.osu.grid_config import TICKS_PER_BAR
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.row_tokens import CanonicalTiming, beatmap_to_window_tokens
from audio2map.osu.schema import Beatmap
from audio2map.utils.paths import audio_grid_dir

logger = logging.getLogger(__name__)

_VOCAB: dict[str, int] | None = None


def get_vocab() -> dict[str, int]:
    global _VOCAB
    if _VOCAB is None:
        from audio2map.osu.row_tokens import build_vocab

        _VOCAB = build_vocab()
    return _VOCAB


@dataclass(slots=True)
class ChartBundle:
    path: Path
    beatmap: Beatmap
    timing: CanonicalTiming
    cond_vec: np.ndarray
    grid: np.ndarray
    grid_meta: AudioGridMeta


def has_audio_grid(path: Path, *, grid_dir: Path | None = None) -> bool:
    grid_dir = grid_dir or audio_grid_dir()
    from audio2map.audio.loader import find_audio_file

    try:
        timing = CanonicalTiming.from_beatmap(parse_beatmap(path))
        audio_path = find_audio_file(path.parent)
        stem = resolve_grid_stem(audio_path, timing)
        npy, json_path = grid_cache_paths(grid_dir, stem)
        return npy.is_file() and json_path.is_file()
    except Exception:
        return False


def filter_paths_with_grid(
    paths: list[Path],
    *,
    grid_dir: Path | None = None,
) -> list[Path]:
    grid_dir = grid_dir or audio_grid_dir()
    return [p for p in paths if has_audio_grid(p, grid_dir=grid_dir)]


def preload_chart_bundle(
    path: Path,
    *,
    grid_dir: Path | None = None,
    chart_meta: ChartMeta | None = None,
    require_grid: bool = True,
) -> ChartBundle | None:
    path = Path(path)
    grid_dir = grid_dir or audio_grid_dir()
    try:
        beatmap = parse_beatmap(path)
        if not check_beatmap_eligibility(beatmap).eligible:
            return None
        timing = CanonicalTiming.from_beatmap(beatmap)
        from audio2map.audio.loader import find_audio_file

        audio_path = find_audio_file(path.parent)
        stem = resolve_grid_stem(audio_path, timing)
        if require_grid:
            npy, json_path = grid_cache_paths(grid_dir, stem)
            if not npy.is_file():
                return None
        grid, grid_meta = load_audio_grid(grid_dir, stem)
        meta = chart_meta or compute_chart_meta(path, skip_msd=True)
        return ChartBundle(
            path=path,
            beatmap=beatmap,
            timing=timing,
            cond_vec=build_cond_vec(meta),
            grid=grid,
            grid_meta=grid_meta,
        )
    except Exception as exc:
        logger.debug("skip bundle %s: %s", path, exc)
        return None


def preload_chart_bundles(
    paths: list[Path],
    *,
    grid_dir: Path | None = None,
    require_grid: bool = True,
) -> list[ChartBundle]:
    bundles: list[ChartBundle] = []
    for path in tqdm(paths, desc="preload charts"):
        bundle = preload_chart_bundle(path, grid_dir=grid_dir, require_grid=require_grid)
        if bundle is not None:
            bundles.append(bundle)
    return bundles


def build_sample_from_bundle(
    bundle: ChartBundle,
    *,
    cfg: WindowSamplingConfig | None = None,
    rng: random.Random | None = None,
    start_bar: int | None = None,
) -> V2TrainingSample | None:
    cfg = cfg or WindowSamplingConfig()
    rng = rng or random.Random()

    chart_start, chart_end = chart_bar_range(bundle.beatmap, bundle.timing)
    audio_start, audio_end = audio_bar_range_from_duration(
        bundle.grid_meta.duration_ms, bundle.timing
    )

    if start_bar is None:
        sampled = sample_window_bars(
            chart_start_bar=chart_start,
            chart_end_bar=chart_end,
            audio_start_bar=audio_start,
            audio_end_bar=audio_end,
            cfg=cfg,
            rng=rng,
        )
        if sampled is None:
            return None
        start_bar, end_bar = sampled
    else:
        end_bar = start_bar + cfg.pick_window_bars(rng)

    tokens = beatmap_to_window_tokens(
        bundle.beatmap,
        start_bar=start_bar,
        window_bars=end_bar - start_bar,
        timing=bundle.timing,
    )
    vocab = get_vocab()
    token_ids = [vocab[t] for t in tokens]
    audio_features, slice_info = slice_audio_window(
        bundle.grid, bundle.grid_meta, start_bar, end_bar
    )

    return V2TrainingSample(
        set_id=bundle.beatmap.metadata.beatmap_set_id,
        beatmap_id=bundle.beatmap.metadata.beatmap_id,
        osu_path=str(bundle.path),
        window={"start_bar": start_bar, "window_bars": end_bar - start_bar},
        tokens=tokens,
        token_ids=token_ids,
        loss_mask=build_loss_mask(tokens),
        cond_vec=bundle.cond_vec,
        audio_features=audio_features,
        audio_slice={
            "start_tick": slice_info["window_start_tick"],
            "length_ticks": (end_bar - start_bar) * TICKS_PER_BAR,
            "feature_dim": bundle.grid_meta.feature_dim,
        },
        meta={
            "chart_start_bar": chart_start,
            "chart_end_bar": chart_end,
            "audio_start_bar": audio_start,
            "audio_end_bar": audio_end,
        },
    )
