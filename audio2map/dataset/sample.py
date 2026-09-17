"""Unified training-sample construction (tokens + audio slice + cond_vec).

Single entry ``build_sample(Path | ChartBundle)`` for train (preloaded bundle)
and eval / one-off (``.osu`` path). Path loads go through
``preload_chart_bundle`` so both shapes share eligibility, grid, and cond_vec.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from audio2map.dataset.bundle import ChartBundle, preload_chart_bundle
from audio2map.dataset.windows import (
    SampleWindow,
    build_loss_mask,
    frame_window_tokens,
    sample_training_window,
)
from audio2map.features.audio.grid import slice_audio_window
from audio2map.features.cond import ChartMeta
from audio2map.model.config import MAX_DECODER_LEN, WINDOW_BARS
from audio2map.tokens import build_vocab, encode_notes, initial_row_at_bar
from audio2map.utils.paths import audio_grid_dir

logger = logging.getLogger(__name__)


@cache
def _vocab() -> dict[str, int]:
    return build_vocab()


@dataclass(frozen=True, slots=True)
class TrainingSample:
    beatmap_id: int | None
    osu_path: str
    window: SampleWindow
    tokens: list[str]
    token_ids: list[int]
    loss_mask: list[int]
    cond_vec: np.ndarray
    audio_features: np.ndarray


def build_sample(
    source: Path | ChartBundle,
    *,
    rng: random.Random | None = None,
    grid_dir: Path | None = None,
    chart_meta: ChartMeta | None = None,
    build_grid_if_missing: bool = True,
    start_bar: int | None = None,
    window_bars: int = WINDOW_BARS,
) -> TrainingSample | None:
    """Build one window sample; ``None`` if ineligible, grid/meta fails, or sampling fails.

    ``source`` is a preloaded ``ChartBundle`` (training) or a ``.osu`` ``Path``
    (eval / one-off; uses ``grid_dir`` / ``chart_meta`` / ``build_grid_if_missing``).
    """
    rng = rng or random.Random()

    if isinstance(source, ChartBundle):
        bundle = source
    else:
        bundle = preload_chart_bundle(
            Path(source),
            grid_dir=grid_dir or audio_grid_dir(),
            chart_meta=chart_meta,
            build_grid_if_missing=build_grid_if_missing,
        )
        if bundle is None:
            return None

    audio_start, audio_end = bundle.grid_meta.audio_bar_range

    if start_bar is None:
        window = sample_training_window(
            audio_start_bar=audio_start,
            audio_end_bar=audio_end,
            rng=rng,
            window_bars=window_bars,
        )
        if window is None:
            return None
    else:
        window = SampleWindow(start_bar=start_bar, window_bars=window_bars)

    chart_tokens = encode_notes(
        bundle.beatmap.notes,
        bundle.timing,
        start_bar=window.start_bar,
        end_bar=window.end_bar,
    )
    tokens = frame_window_tokens(
        chart_tokens,
        initial_row_at_bar(bundle.beatmap.notes, bundle.timing, start_bar=window.start_bar),
    )
    vocab = _vocab()
    token_ids = [vocab[t] for t in tokens]
    if len(token_ids) > MAX_DECODER_LEN:
        logger.warning(
            "window token overflow len=%d > MAX_DECODER_LEN=%d for %s start_bar=%d",
            len(token_ids),
            MAX_DECODER_LEN,
            bundle.path,
            window.start_bar,
        )
        return None
    audio_features, _ = slice_audio_window(
        bundle.grid, bundle.grid_meta, window.start_bar, window.end_bar
    )

    return TrainingSample(
        beatmap_id=bundle.beatmap.metadata.beatmap_id,
        osu_path=str(bundle.path),
        window=window,
        tokens=tokens,
        token_ids=token_ids,
        loss_mask=build_loss_mask(tokens),
        cond_vec=bundle.cond_vec,
        audio_features=audio_features,
    )
