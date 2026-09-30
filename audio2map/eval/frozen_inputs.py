"""Pinned paths and seeds for the frozen eval inputs.

Selection seeds are independent of generation seeds.
"""

from __future__ import annotations

from pathlib import Path

from audio2map.utils.paths import get_data_root

RANDOM_FULL_SEED = 2028
DIVERSITY_SUBSET_SEED = 2026
DIVERSITY_SUBSET_COUNT = 100
UNCONSTRAINED_SUBSET_SEED = 2027
UNCONSTRAINED_SUBSET_COUNT = 300
DIVERSITY_GENERATION_SEEDS = (0, 1, 2, 3, 4)
MOTIF_SCOPES = ("lane", "hand", "row")
MOTIF_LIST_COUNT = 50000


def default_motif_lists_path() -> Path:
    return get_data_root() / "freeze" / "motif_topk_fit_50000.lists.json"


def assert_motif_list_count(data: dict, count: int = MOTIF_LIST_COUNT) -> None:
    """Official counting uses the full frozen list for every scope."""
    if set(data) != set(MOTIF_SCOPES):
        raise ValueError(
            f"motif lists must contain exactly {MOTIF_SCOPES}, got {sorted(data)}"
        )
    for scope in MOTIF_SCOPES:
        items = data[scope]
        if not isinstance(items, list) or len(items) != count:
            length = len(items) if isinstance(items, list) else None
            raise ValueError(f"{scope} motif list length {length} != {count}")
