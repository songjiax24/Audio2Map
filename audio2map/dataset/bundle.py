"""Cached per-chart data for training sample construction.

Training path: ``preload_chart_bundles`` → ``dataset.sample.build_sample``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from audio2map.dataset.filter import check_beatmap_eligibility
from audio2map.dataset.windows import audio_covers_training_window
from audio2map.features.audio.grid import (
    AudioGridMeta,
    grid_cache_paths,
    is_valid_audio_grid_cache,
    load_audio_grid,
    precompute_chart_audio_grid,
    resolve_grid_stem,
)
from audio2map.features.audio.tick_features import AUDIO_FEATURE_SPEC_VERSION
from audio2map.features.cond import (
    ChartMeta,
    CondVecError,
    build_cond_vec,
    compute_chart_meta,
    load_chart_meta_manifest,
)
from audio2map.grid import CanonicalTiming
from audio2map.model.config import WINDOW_BARS
from audio2map.osu.parser import chart_audio_path, parse_beatmap
from audio2map.osu.schema import Beatmap
from audio2map.utils.paths import audio_grid_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
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
    _entries: dict[str, tuple[np.ndarray, AudioGridMeta]] = field(default_factory=dict)

    def load(self, stem: str) -> tuple[np.ndarray, AudioGridMeta]:
        if stem not in self._entries:
            self._entries[stem] = load_audio_grid(self.grid_dir, stem)
        return self._entries[stem]

    def __len__(self) -> int:
        return len(self._entries)


def _grid_stem(path: Path, timing: CanonicalTiming) -> str:
    return resolve_grid_stem(chart_audio_path(path), timing)


def has_audio_grid(
    path: Path,
    *,
    grid_dir: Path | None = None,
    timing: CanonicalTiming | None = None,
) -> bool:
    grid_dir = grid_dir or audio_grid_dir()
    try:
        if timing is None:
            timing = CanonicalTiming.from_beatmap(parse_beatmap(path))
        stem = _grid_stem(path, timing)
        npy, json_path = grid_cache_paths(grid_dir, stem)
        return is_valid_audio_grid_cache(npy, json_path)
    except (OSError, ValueError):
        return False


def grid_covers_training_window(
    path: Path,
    *,
    grid_dir: Path | None = None,
    timing: CanonicalTiming | None = None,
    window_bars: int = WINDOW_BARS,
) -> bool:
    """True if a valid grid cache exists and spans at least ``window_bars``."""
    grid_dir = grid_dir or audio_grid_dir()
    try:
        if timing is None:
            timing = CanonicalTiming.from_beatmap(parse_beatmap(path))
        stem = _grid_stem(path, timing)
        npy, json_path = grid_cache_paths(grid_dir, stem)
        if not npy.is_file() or not json_path.is_file():
            return False
        meta = AudioGridMeta.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
        if meta.feature_spec_version != AUDIO_FEATURE_SPEC_VERSION:
            return False
        start, end = meta.audio_bar_range
        return audio_covers_training_window(start, end, window_bars=window_bars)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def filter_paths_with_grid(
    paths: list[Path],
    *,
    grid_dir: Path | None = None,
    min_window_bars: int | None = None,
) -> list[Path]:
    grid_dir = grid_dir or audio_grid_dir()
    if min_window_bars is None:
        return [p for p in paths if has_audio_grid(p, grid_dir=grid_dir)]
    return [
        p
        for p in paths
        if grid_covers_training_window(p, grid_dir=grid_dir, window_bars=min_window_bars)
    ]


def preload_chart_bundle(
    path: Path,
    *,
    grid_dir: Path | None = None,
    chart_meta: ChartMeta | None = None,
    meta_manifest: dict[str, ChartMeta] | None = None,
    build_grid_if_missing: bool = False,
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
        stem = _grid_stem(path, timing)
        npy, json_path = grid_cache_paths(grid_dir, stem)
        if not is_valid_audio_grid_cache(npy, json_path):
            if not build_grid_if_missing:
                logger.warning("skip bundle %s: missing or invalid audio grid %s", path, stem)
                return None
            precompute_chart_audio_grid(chart_audio_path(path), timing, grid_dir, skip_existing=True)
        if grid_cache is not None:
            grid, grid_meta = grid_cache.load(stem)
        else:
            grid, grid_meta = load_audio_grid(grid_dir, stem)
        audio_start, audio_end = grid_meta.audio_bar_range
        if not audio_covers_training_window(audio_start, audio_end):
            logger.warning(
                "skip bundle %s: audio spans %d bars, need >= %d",
                path,
                audio_end - audio_start,
                WINDOW_BARS,
            )
            return None
        if chart_meta is not None:
            meta = chart_meta
        elif meta_manifest is not None:
            meta = meta_manifest.get(str(path.resolve()))
            if meta is None:
                logger.warning("skip bundle %s: missing from chart meta manifest", path)
                return None
        else:
            meta = compute_chart_meta(path)
        if meta.error is not None:
            logger.warning("skip bundle %s: %s", path, meta.error)
            return None
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
    build_grid_if_missing: bool = False,
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
    n_in = len(paths)
    for path in tqdm(paths, desc="preload charts"):
        bundle = preload_chart_bundle(
            path,
            grid_dir=grid_dir,
            build_grid_if_missing=build_grid_if_missing,
            grid_cache=grid_cache,
            meta_manifest=meta_manifest,
        )
        if bundle is not None:
            bundles.append(bundle)
    skipped = n_in - len(bundles)
    logger.info(
        "preloaded %d / %d charts (skipped %d) sharing %d unique audio grids",
        len(bundles),
        n_in,
        skipped,
        len(grid_cache),
    )
    return bundles
