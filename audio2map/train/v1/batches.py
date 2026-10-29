"""Pack whole charts by rate-1 frame count.

Windows are a forward partition of the length-sorted charts: from the next
chart of ``f`` frames, take every following chart with ``frames <= f * (1 + frac)``.
Each window is shuffled, the windows stay in short-to-long order, then charts
are packed while ``n * max(frames) <= budget``. Batch order is shuffled last.
Both shuffles use ``default_rng(seed + epoch)``, so a resume rebuilds the order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# 7/8 of the measured 2×262144 frame product. That product fit, and its
# reserved memory sat on the cuDNN failure line.
FRAME_BUDGET = 458_752
WINDOW_FRAC = 0.10


def cut_windows(frames: Sequence[int], frac: float = WINDOW_FRAC) -> list[list[int]]:
    """Non-overlapping forward windows. Non-positive lengths are left out."""
    if frac <= 0.0:
        raise ValueError(f"window frac must be positive, got {frac}")
    order = sorted(
        (index for index, count in enumerate(frames) if count > 0),
        key=lambda index: (frames[index], index),
    )
    windows: list[list[int]] = []
    start = 0
    while start < len(order):
        limit = frames[order[start]] * (1.0 + frac)
        stop = start + 1
        while stop < len(order) and frames[order[stop]] <= limit:
            stop += 1
        windows.append(order[start:stop])
        start = stop
    return windows


def pack_in_order(
    indices: Sequence[int],
    frames: Sequence[int],
    budget: int = FRAME_BUDGET,
) -> list[list[int]]:
    """Greedy pack. ``budget`` is ``n * max(frames)`` on the given order."""
    if budget < 1:
        raise ValueError(f"frame budget must be positive, got {budget}")
    batches: list[list[int]] = []
    pending: list[int] = []
    for index in indices:
        if not pending:
            pending = [index]
            continue
        width = max(frames[index], max(frames[item] for item in pending))
        if (len(pending) + 1) * width <= budget:
            pending.append(index)
            continue
        batches.append(pending)
        pending = [index]
    if pending:
        batches.append(pending)
    return batches


def epoch_batches(
    frames: Sequence[int],
    *,
    seed: int,
    epoch: int,
    budget: int = FRAME_BUDGET,
    frac: float = WINDOW_FRAC,
) -> list[list[int]]:
    """One epoch. Every index appears once. Non-positive lengths are alone."""
    rng = np.random.default_rng(seed + epoch)
    ordered: list[int] = []
    for window in cut_windows(frames, frac):
        block = list(window)
        rng.shuffle(block)
        ordered.extend(block)
    batches = pack_in_order(ordered, frames, budget)
    for index, count in enumerate(frames):
        if count <= 0:
            batches.append([index])
    rng.shuffle(batches)
    return batches
