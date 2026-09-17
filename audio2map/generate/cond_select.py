"""Pick a donor chart from the chart-meta manifest by original-unit ranges."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio2map.features.cond import (
    CANONICAL_BPM_NORM_INDEX,
    COND_VEC_NAMES,
    ChartMeta,
    CondVecError,
    USER_COND_SOURCE_FIELDS,
    build_cond_vec,
    load_chart_meta_manifest,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CondCandidate",
    "CondSelectionError",
    "USER_COND_SOURCE_FIELDS",
    "build_final_cond_vec",
    "load_cond_candidates",
    "select_candidate_from_ranges",
    "select_condition",
]


@dataclass(frozen=True, slots=True)
class CondCandidate:
    chart_id: str
    osu_path: str
    cond_vec: np.ndarray
    meta: ChartMeta


class CondSelectionError(RuntimeError):
    """No manifest chart satisfies the requested cond constraints."""


def _chart_id(meta: ChartMeta) -> str:
    if meta.beatmap_id is not None:
        return str(meta.beatmap_id)
    return Path(meta.osu_path).stem


def load_cond_candidates(manifest_path: Path | None = None) -> list[CondCandidate]:
    manifest = load_chart_meta_manifest(manifest_path)
    candidates: list[CondCandidate] = []
    n_error = n_cond = 0
    for meta in manifest.values():
        if meta.error is not None:
            n_error += 1
            continue
        try:
            cond = build_cond_vec(meta)
        except CondVecError:
            n_cond += 1
            continue
        candidates.append(
            CondCandidate(
                chart_id=_chart_id(meta),
                osu_path=meta.osu_path,
                cond_vec=cond,
                meta=meta,
            )
        )
    if n_error or n_cond:
        logger.info(
            "cond candidates: kept %d, skipped error=%d cond=%d",
            len(candidates),
            n_error,
            n_cond,
        )
    return candidates


def _in_range(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def _matches_in_ranges(
    candidates: list[CondCandidate],
    ranges: dict[str, tuple[float, float]],
) -> list[CondCandidate]:
    if "canonical_bpm_norm" in ranges or "canonical_bpm" in ranges:
        raise ValueError("canonical_bpm_norm must not be manually ranged")
    names = list(USER_COND_SOURCE_FIELDS)
    missing = [n for n in names if n not in ranges]
    if missing:
        raise ValueError(f"ranges missing cond dims: {missing}")
    return [
        cand
        for cand in candidates
        if all(_in_range(float(getattr(cand.meta, name)), *ranges[name]) for name in names)
    ]


def _pick_uniform(matched: list[CondCandidate], rng: random.Random | None) -> CondCandidate:
    if rng is None:
        rng = random.Random()
    return rng.choice(matched)


def select_candidate_from_ranges(
    candidates: list[CondCandidate],
    ranges: dict[str, tuple[float, float]],
    *,
    rng: random.Random | None = None,
) -> CondCandidate | None:
    """Filter by ChartMeta source fields; sample uniformly among matches."""
    matched = _matches_in_ranges(candidates, ranges)
    return _pick_uniform(matched, rng) if matched else None


def build_final_cond_vec(candidate: CondCandidate, canonical_bpm_norm: float) -> np.ndarray:
    out = candidate.cond_vec.copy()
    out[CANONICAL_BPM_NORM_INDEX] = float(canonical_bpm_norm)
    return out


def select_condition(
    candidates: list[CondCandidate],
    ranges: dict[str, tuple[float, float]],
    canonical_bpm_norm: float,
    *,
    rng: random.Random | None = None,
) -> dict:
    """Return selection payload; raises if no candidate matches."""
    matched = _matches_in_ranges(candidates, ranges)
    if not matched:
        raise CondSelectionError("no manifest charts match the requested cond ranges")
    selected = _pick_uniform(matched, rng)
    final = build_final_cond_vec(selected, canonical_bpm_norm)
    return {
        "selected_chart_id": selected.chart_id,
        "selected_osu_path": selected.osu_path,
        "matched_candidate_count": len(matched),
        "cond_vec_names": list(COND_VEC_NAMES),
        "selected_original_cond_vec": selected.cond_vec.tolist(),
        "selected_source_values": {
            name: float(getattr(selected.meta, name)) for name in USER_COND_SOURCE_FIELDS
        },
        "final_cond_vec": final.tolist(),
        "canonical_bpm_norm_source": "user_entered_bpm",
        "selection_rule": "uniform_among_matches",
    }
