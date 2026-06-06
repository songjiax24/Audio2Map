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
    is_valid_audio_grid_cache,
    load_audio_grid,
    resolve_grid_stem,
    slice_audio_window,
)
from audio2map.data.v2_dataset import V2TrainingSample
from audio2map.data.chart_filter import check_beatmap_eligibility
from audio2map.data.cond_vec import CondVecError, build_cond_vec
from audio2map.data.window_sampler import (
    WindowSamplingConfig,
    assert_window_start_bar_aligned,
    audio_bar_range_from_duration,
    build_loss_mask,
    chart_bar_range,
    sample_window_bars,
)
from audio2map.difficulty.chart_meta import ChartMeta, compute_chart_meta
from audio2map.difficulty.chart_meta_manifest import load_chart_meta_manifest
from audio2map.osu.grid_config import TICKS_PER_BAR, WINDOW_BARS
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.row_tokens import CanonicalTiming, beatmap_to_window_tokens
from audio2map.osu.schema import Beatmap
from audio2map.training.config import MAX_DECODER_LEN
from audio2map.utils.paths import audio_grid_dir

logger = logging.getLogger(__name__)

_window_token_overflow_count = 0


def window_token_overflow_count() -> int:
    return _window_token_overflow_count

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


@dataclass
class SharedGridCache:
    """Load each audio grid stem once; share ``grid`` arrays across chart bundles."""

    grid_dir: Path
    _entries: dict[str, tuple[np.ndarray, AudioGridMeta]] | None = None

    def __post_init__(self) -> None:
        if self._entries is None:
            self._entries = {}

    def load(self, stem: str) -> tuple[np.ndarray, AudioGridMeta]:
        assert self._entries is not None
        if stem not in self._entries:
            self._entries[stem] = load_audio_grid(self.grid_dir, stem)
        return self._entries[stem]

    def __len__(self) -> int:
        assert self._entries is not None
        return len(self._entries)


def has_audio_grid(path: Path, *, grid_dir: Path | None = None) -> bool:
    grid_dir = grid_dir or audio_grid_dir()
    from audio2map.audio.loader import find_audio_file

    try:
        timing = CanonicalTiming.from_beatmap(parse_beatmap(path))
        audio_path = find_audio_file(path.parent)
        stem = resolve_grid_stem(audio_path, timing)
        npy, json_path = grid_cache_paths(grid_dir, stem)
        return is_valid_audio_grid_cache(npy, json_path)
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
    meta_manifest: dict[str, ChartMeta] | None = None,
    require_grid: bool = True,
    grid_cache: SharedGridCache | None = None,
) -> ChartBundle | None:
    path = Path(path)
    grid_dir = grid_dir or audio_grid_dir()
    try:
        beatmap = parse_beatmap(path)
        if not check_beatmap_eligibility(beatmap).eligible:
            logger.warning("skip bundle %s: not eligible", path)
            return None
        timing = CanonicalTiming.from_beatmap(beatmap)
        from audio2map.audio.loader import find_audio_file

        audio_path = find_audio_file(path.parent)
        stem = resolve_grid_stem(audio_path, timing)
        if require_grid:
            npy, json_path = grid_cache_paths(grid_dir, stem)
            if not is_valid_audio_grid_cache(npy, json_path):
                logger.warning("skip bundle %s: missing or invalid audio grid %s", path, stem)
                return None
        if grid_cache is not None:
            grid, grid_meta = grid_cache.load(stem)
        else:
            grid, grid_meta = load_audio_grid(grid_dir, stem)
        if chart_meta is not None:
            meta = chart_meta
        elif meta_manifest is not None:
            meta = meta_manifest.get(str(path.resolve()))
            if meta is None:
                logger.warning("skip bundle %s: missing from chart meta manifest", path)
                return None
        else:
            meta = compute_chart_meta(path)
        try:
            cond_vec = build_cond_vec(meta)
        except CondVecError as exc:
            logger.warning("%s", exc)
            return None
        return ChartBundle(
            path=path,
            beatmap=beatmap,
            timing=timing,
            cond_vec=cond_vec,
            grid=grid,
            grid_meta=grid_meta,
        )
    except Exception as exc:
        logger.warning("skip bundle %s: %s", path, exc)
        return None


def preload_chart_bundles(
    paths: list[Path],
    *,
    grid_dir: Path | None = None,
    require_grid: bool = True,
    meta_manifest_path: Path | None = None,
) -> list[ChartBundle]:
    grid_dir = grid_dir or audio_grid_dir()
    grid_cache = SharedGridCache(grid_dir)
    meta_manifest = (
        load_chart_meta_manifest(meta_manifest_path)
        if meta_manifest_path is not None
        else None
    )
    bundles: list[ChartBundle] = []
    for path in tqdm(paths, desc="preload charts"):
        bundle = preload_chart_bundle(
            path,
            grid_dir=grid_dir,
            require_grid=require_grid,
            grid_cache=grid_cache,
            meta_manifest=meta_manifest,
        )
        if bundle is not None:
            bundles.append(bundle)
    logger.info(
        "preloaded %d charts sharing %d unique audio grids",
        len(bundles),
        len(grid_cache),
    )
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
            audio_start_bar=audio_start,
            audio_end_bar=audio_end,
            cfg=cfg,
            rng=rng,
        )
        if sampled is None:
            return None
        start_bar, end_bar = sampled
    else:
        end_bar = start_bar + WINDOW_BARS

    assert_window_start_bar_aligned(start_bar)

    tokens = beatmap_to_window_tokens(
        bundle.beatmap,
        start_bar=start_bar,
        window_bars=end_bar - start_bar,
        timing=bundle.timing,
    )
    vocab = get_vocab()
    token_ids = [vocab[t] for t in tokens]
    if len(token_ids) > MAX_DECODER_LEN:
        global _window_token_overflow_count
        _window_token_overflow_count += 1
        logger.warning(
            "window token overflow len=%d > MAX_DECODER_LEN=%d for %s start_bar=%d (total_overflow=%d)",
            len(token_ids),
            MAX_DECODER_LEN,
            bundle.path,
            start_bar,
            _window_token_overflow_count,
        )
        return None
    audio_features, slice_info = slice_audio_window(
        bundle.grid, bundle.grid_meta, start_bar, end_bar
    )
    assert slice_info["window_start_tick"] == start_bar * TICKS_PER_BAR

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
