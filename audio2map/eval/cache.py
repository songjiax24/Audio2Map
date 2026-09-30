"""Generation-result cache keys.

The key includes the scenario even when two scenarios happen to pass the same
condition bytes, so matched_full and random_full cannot share a result. It
also includes the decode settings and the overlap and range settings that the
v0 backend actually uses. Diversity reuses a result only when every field
matches the run that produced it. The cache lives for one ``eval_generation``
call.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def generation_cache_key(
    *,
    backend: str,
    relpath: str,
    scenario: str,
    mode: str,
    seed: int,
    cond_vec: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
    range_mode: str,
    pre_margin_bars: int,
    post_margin_bars: int,
    window_bars: int,
    context_bars: int,
    keep_bars: int,
    future_bars: int,
    max_seq_len: int,
) -> tuple:
    return (
        backend,
        relpath,
        scenario,
        mode,
        int(seed),
        cond_vec.tobytes(),
        float(temperature),
        float(top_p),
        int(top_k),
        range_mode,
        int(pre_margin_bars),
        int(post_margin_bars),
        int(window_bars),
        int(context_bars),
        int(keep_bars),
        int(future_bars),
        int(max_seq_len),
    )


def cache_get_or_put(cache: dict, key: tuple, produce: Callable[[], object]) -> object:
    if key not in cache:
        cache[key] = produce()
    return cache[key]
