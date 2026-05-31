"""Default thresholds for the vendored YAVSRG/Prelude pattern port."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_DEFAULTS_PATH = Path(__file__).with_name("config_defaults.json")


def _load_defaults() -> dict[str, Any]:
    with _DEFAULTS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@dataclass
class PatternConfig:
    """Pattern analyser configuration (defaults from nonebot-plugin-osumania-toolkit)."""

    coordination_specific_order: list[str]
    core_rating_multiplier: dict[str, float]
    density_specific_order: list[str]
    inverse_gap_tolerance_ms: float
    inverse_min_filled_lanes: int
    jacky_context_window: int
    jacky_fallback_max_mspb: float
    jacks_min_bpm: float
    release_full_match_rows: int
    release_min_tail_rows: int
    release_roll_points: int
    release_scan_rows: int
    shield_max_beat_ratio: float
    subtype_rating_multiplier_by_mode: dict[str, dict[str, float]]
    rc_core_ln_scale: float
    rc_ln_core_scale: float
    wildcard_specific_order: list[str]
    ln_mode_low_threshold: float
    ln_mode_high_threshold: float
    hb_row_ratio_threshold: float
    bpm_cluster_threshold: float
    pattern_stability_threshold: float
    important_cluster_ratio: float
    category_js_hs_secondary_ratio: float
    sv_amount_threshold: float
    sv_speed_eps: float
    sv_extreme_bpm_min: float
    sv_extreme_bpm_max: float
    sv_extreme_bpm_ratio: float
    cluster_specific_name_min_ratio: float
    enable_multi_label_same_window: bool
    release_with_dw_multiplier: float

    @classmethod
    def load(cls) -> PatternConfig:
        return cls(**_load_defaults())


def _as_namespace(cfg: PatternConfig) -> SimpleNamespace:
    return SimpleNamespace(**cfg.__dict__)


config = _as_namespace(PatternConfig.load())
