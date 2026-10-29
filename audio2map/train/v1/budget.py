"""R2 pack caps from 2026-10-26. The training loop does not import this module.

Packing is ``batches.py``. These numbers record the measurement below.

How many whole charts share a step on a 32 GiB 5090.

Numbers come from the Flex (C) grid in ``tools/r2_ca_memory_benchmark.md``.
Peaks there are ``max_memory_allocated`` after warmup, bf16, without Adam
(about 0.55 GiB). The card has 31.36 GiB.

- 28672 frames × 8192 tokens × bs 1 fits (22.97 GiB).
- 24064 frames × bs 2 is about 30 GiB. Adam fills the card.
- 24064 × 6144 × bs 2 and 28672 × 8192 × bs 2 OOM in the encoder.
- 11264 × 6144 × bs 2 fits (14.42 GiB). 8192 × 6144 × bs 4 fits (21.87 GiB).

Shared batches therefore stop at 16384 padded frames and 6144 padded tokens.
The slot caps reject a pair whose padded cost would still land on that cliff.
A single chart past the measured bs-1 point is skipped, not run alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Padded maxima for a batch with more than one chart.
PACK_FRAMES = 16_384
PACK_TOKENS = 6_144
# n * padded length. 2 * 16384 stays under the 24064×2 cliff.
FRAME_SLOTS = 32_768
TOKEN_SLOTS = 12_288
# A lone chart above these is not forwarded.
HARD_FRAMES = 28_672
HARD_TOKENS = 16_384


@dataclass(frozen=True, slots=True)
class BatchBudget:
    """Limits applied to post-augmentation frame and token counts."""

    batch_size: int
    alone_frames: int
    alone_tokens: int
    hard_frames: int
    hard_tokens: int
    frame_slots: int
    token_slots: int
    frame_p90: int
    token_p90: int


def resolve_budget(
    *,
    batch_size: int,
    frame_p90: float,
    token_p90: float,
    pack_frames: int = PACK_FRAMES,
    pack_tokens: int = PACK_TOKENS,
    frame_slots: int = FRAME_SLOTS,
    token_slots: int = TOKEN_SLOTS,
    hard_frames: int = HARD_FRAMES,
    hard_tokens: int = HARD_TOKENS,
) -> BatchBudget:
    """Shared batches stop at the tighter of the pack cap and the rate-1 p90.

    ``frame_p90`` / ``token_p90`` describe the training split at rate 1.
    The step itself compares these lines to the chart after augmentation.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if min(hard_frames, hard_tokens, frame_slots, token_slots) < 1:
        raise ValueError("length limits must be positive")
    alone_frames = min(pack_frames, hard_frames, max(1, int(math.floor(frame_p90))))
    alone_tokens = min(pack_tokens, hard_tokens, max(1, int(math.floor(token_p90))))
    return BatchBudget(
        batch_size=batch_size,
        alone_frames=alone_frames,
        alone_tokens=alone_tokens,
        hard_frames=hard_frames,
        hard_tokens=hard_tokens,
        frame_slots=frame_slots,
        token_slots=token_slots,
        frame_p90=max(1, int(math.floor(frame_p90))),
        token_p90=max(1, int(math.floor(token_p90))),
    )


def over_hard(frames: int, tokens: int, budget: BatchBudget) -> bool:
    """True when even bs 1 is past the measured point."""
    return frames > budget.hard_frames or tokens > budget.hard_tokens


def can_share(
    frames: list[int],
    tokens: list[int],
    extra_frames: int,
    extra_tokens: int,
    budget: BatchBudget,
) -> bool:
    """True when ``extra`` can join this group without padding past the slots."""
    count = len(frames) + 1
    if count > budget.batch_size:
        return False
    if extra_frames >= budget.alone_frames or extra_tokens >= budget.alone_tokens:
        return False
    padded_frames = max(max(frames), extra_frames)
    padded_tokens = max(max(tokens), extra_tokens)
    if padded_frames >= budget.alone_frames or padded_tokens >= budget.alone_tokens:
        return False
    return (
        count * padded_frames <= budget.frame_slots
        and count * padded_tokens <= budget.token_slots
    )
