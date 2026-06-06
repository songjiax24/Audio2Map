"""Build v2 training samples (tokens + audio slice + cond_vec)."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from audio2map.data.audio_grid import (
    AudioGridMeta,
    grid_cache_paths,
    is_valid_audio_grid_cache,
    load_audio_grid,
    precompute_chart_audio_grid,
    resolve_grid_stem,
    slice_audio_window,
)
from audio2map.data.chart_filter import check_beatmap_eligibility
from audio2map.data.cond_vec import CondVecError, build_cond_vec
from audio2map.data.window_sampler import (
    WindowSamplingConfig,
    audio_bar_range_from_duration,
    build_loss_mask,
    chart_bar_range,
    sample_window_bars,
)
from audio2map.difficulty.chart_meta import ChartMeta, compute_chart_meta
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.row_tokens import CanonicalTiming, beatmap_to_window_tokens, build_vocab
from audio2map.osu.grid_config import TICKS_PER_BAR, WINDOW_BARS
from audio2map.utils.paths import audio_grid_dir


@dataclass(slots=True)
class V2TrainingSample:
    set_id: int | None
    beatmap_id: int | None
    osu_path: str
    window: dict[str, int]
    tokens: list[str]
    token_ids: list[int]
    loss_mask: list[int]
    cond_vec: np.ndarray
    audio_features: np.ndarray
    audio_slice: dict[str, int | float]
    meta: dict

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cond_vec"] = self.cond_vec.tolist()
        d["audio_features"] = self.audio_features.tolist()
        return d


def _load_or_build_grid(
    set_dir: Path,
    timing: CanonicalTiming,
    *,
    grid_dir: Path,
    build_if_missing: bool,
) -> tuple[np.ndarray, AudioGridMeta]:
    from audio2map.audio.loader import find_audio_file

    audio_path = find_audio_file(set_dir)
    stem = resolve_grid_stem(audio_path, timing)
    npy, json_path = grid_cache_paths(grid_dir, stem)
    if is_valid_audio_grid_cache(npy, json_path):
        return load_audio_grid(grid_dir, stem)
    if not build_if_missing:
        raise FileNotFoundError(f"audio grid cache missing or invalid: {stem}")
    precompute_chart_audio_grid(set_dir, timing, grid_dir, skip_existing=False)
    return load_audio_grid(grid_dir, stem)


def build_training_sample(
    osu_path: Path,
    *,
    cfg: WindowSamplingConfig | None = None,
    rng: random.Random | None = None,
    grid_dir: Path | None = None,
    chart_meta: ChartMeta | None = None,
    build_grid_if_missing: bool = True,
    start_bar: int | None = None,
) -> V2TrainingSample | None:
    """Build one window sample; ``None`` if ineligible or sampling fails."""
    cfg = cfg or WindowSamplingConfig()
    rng = rng or random.Random()
    grid_dir = grid_dir or audio_grid_dir()

    beatmap = parse_beatmap(osu_path)
    eligibility = check_beatmap_eligibility(beatmap)
    if not eligibility.eligible:
        return None

    timing = CanonicalTiming.from_beatmap(beatmap)
    set_dir = osu_path.parent

    grid, grid_meta = _load_or_build_grid(
        set_dir,
        timing,
        grid_dir=grid_dir,
        build_if_missing=build_grid_if_missing,
    )

    chart_start, chart_end = chart_bar_range(beatmap, timing)
    audio_start, audio_end = audio_bar_range_from_duration(grid_meta.duration_ms, timing)

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

    tokens = beatmap_to_window_tokens(
        beatmap,
        start_bar=start_bar,
        window_bars=end_bar - start_bar,
        timing=timing,
    )
    vocab = build_vocab()
    token_ids = [vocab[t] for t in tokens]
    loss_mask = build_loss_mask(tokens)
    try:
        cond = build_cond_vec(chart_meta or compute_chart_meta(osu_path))
    except CondVecError as exc:
        import logging

        logging.getLogger(__name__).warning("%s", exc)
        return None

    audio_features, slice_info = slice_audio_window(grid, grid_meta, start_bar, end_bar)

    return V2TrainingSample(
        set_id=beatmap.metadata.beatmap_set_id,
        beatmap_id=beatmap.metadata.beatmap_id,
        osu_path=str(osu_path),
        window={"start_bar": start_bar, "window_bars": end_bar - start_bar},
        tokens=tokens,
        token_ids=token_ids,
        loss_mask=loss_mask,
        cond_vec=cond,
        audio_features=audio_features,
        audio_slice={
            "start_tick": slice_info["window_start_tick"],
            "length_ticks": (end_bar - start_bar) * TICKS_PER_BAR,
            "feature_dim": grid_meta.feature_dim,
        },
        meta={
            "chart_start_bar": chart_start,
            "chart_end_bar": chart_end,
            "audio_start_bar": audio_start,
            "audio_end_bar": audio_end,
            "tick_min": grid_meta.tick_min,
            "tick_max": grid_meta.tick_max,
            "offset_ms": grid_meta.offset_ms,
            "canonical_bpm": grid_meta.canonical_bpm,
        },
    )


def iter_training_samples(
    osu_paths: list[Path],
    *,
    cfg: WindowSamplingConfig | None = None,
    seed: int = 0,
    grid_dir: Path | None = None,
    build_grid_if_missing: bool = True,
) -> list[V2TrainingSample]:
    rng = random.Random(seed)
    out: list[V2TrainingSample] = []
    for path in osu_paths:
        sample = build_training_sample(
            path,
            cfg=cfg,
            rng=rng,
            grid_dir=grid_dir,
            build_grid_if_missing=build_grid_if_missing,
        )
        if sample is not None:
            out.append(sample)
    return out
