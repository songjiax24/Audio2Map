"""Fill the rate-1 token cache before training. Reading it stays in ``tokens``."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from audio2map.dataset.split import ChartSnapshot
from audio2map.tokens.v1.store import (
    AUDIO_ABS_TOL_MS,
    TokenCache,
    TokenRecord,
    encode_missing_rate1,
    load_token_cache,
    publish_token_cache,
    write_token_cache,
)
from audio2map.train.v1.lengths import ChartLength
from audio2map.utils.paths import v1_token_dir

logger = logging.getLogger(__name__)


def ensure_rate1_tokens(
    charts: list[ChartSnapshot],
    root: Path,
    lengths: list[ChartLength],
    *,
    workers: int,
    directory: Path | None = None,
) -> TokenCache:
    """Encode any train chart whose rate-1 tokens are missing. Returns the cache."""
    directory = directory or v1_token_dir()
    cached = load_token_cache(directory)
    pending: list[tuple[str, float]] = []
    for chart, row in zip(charts, lengths, strict=True):
        if row.error is not None:
            continue
        have = cached.records.get(chart.relpath)
        if (
            have is not None
            and have.error is None
            and have.n == row.n_tokens
            and math.isclose(
                have.t_audio_ms,
                row.t_audio_ms,
                rel_tol=0.0,
                abs_tol=AUDIO_ABS_TOL_MS,
            )
        ):
            continue
        pending.append((chart.relpath, row.t_audio_ms))
    if not pending:
        logger.info("token cache hit %s (%s)", len(charts), directory)
        publish_token_cache(cached)
        return cached
    logger.info("token cache missing %s/%s; encoding rate-1 tokens", len(pending), len(charts))
    fresh = encode_missing_rate1(root, pending, workers=workers)
    merged = dict(cached.records)
    arrays: dict[str, np.ndarray] = {}
    for relpath, record in cached.records.items():
        if record.error is None and record.n > 0:
            arrays[relpath] = cached.ids[record.offset : record.offset + record.n].copy()
    for relpath, t_audio_ms, payload in fresh:
        if isinstance(payload, str):
            merged[relpath] = TokenRecord(relpath, t_audio_ms, 0, 0, error=payload)
            arrays.pop(relpath, None)
        else:
            arrays[relpath] = payload
            merged[relpath] = TokenRecord(relpath, t_audio_ms, 0, int(payload.shape[0]))
    written = write_token_cache(directory, merged, arrays)
    publish_token_cache(written)
    logger.info("wrote %s", directory)
    return written
