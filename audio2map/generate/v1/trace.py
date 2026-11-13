"""Optional observations of native decoder decisions, before any sampling transform."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SamplingStep:
    """State before consuming chosen_token; position indexes the input token.

    raw_logits is an independent float32 CPU copy, before legality masking or
    temperature. Observers must not consume sampling RNG or alter model state.
    Forced prefix replay is identified separately from sampled continuation.
    """

    position: int
    input_token: str
    chosen_token: str
    raw_logits: torch.Tensor
    allowed_tokens: frozenset[str]
    beat_id: int
    last_fraction: tuple[int, int] | None
    pending_fraction: tuple[int, int] | None
    cursor_time_ms: float
    active_lanes: tuple[bool, ...]
    expect_row: bool
    forced: bool


StepObserver = Callable[[SamplingStep], None]
