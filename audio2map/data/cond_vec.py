"""18-dim condition vector for v2 training."""

from __future__ import annotations

import numpy as np

from audio2map.difficulty.chart_meta import ChartMeta

COND_VEC_DIM = 18
COND_VEC_VERSION = 1

# Fixed index map — do not reorder without spec update.
COND_VEC_NAMES: tuple[str, ...] = (
    "osu_sr_norm",
    "analyzer_ln_percent",
    "analyzer_hb_row_ratio",
    "analyzer_stream_ratio",
    "analyzer_chordstream_ratio",
    "analyzer_jacks_ratio",
    "analyzer_coordination_ratio",
    "analyzer_density_ratio",
    "analyzer_wildcard_ratio",
    "etterna_overall_norm",
    "etterna_stream_norm",
    "etterna_jumpstream_norm",
    "etterna_handstream_norm",
    "etterna_stamina_norm",
    "etterna_jack_norm",
    "etterna_chordjack_norm",
    "etterna_technical_norm",
    "canonical_bpm_norm",
)


class CondVecError(ValueError):
    """``ChartMeta`` is missing required difficulty fields for ``cond_vec``."""


def _require(meta: ChartMeta, field: str, value: object | None) -> None:
    if value is None:
        raise CondVecError(f"cond_vec unavailable for {meta.osu_path}: missing {field}")


def build_cond_vec(meta: ChartMeta, *, require_msd: bool = True) -> np.ndarray:
    """Map ``ChartMeta`` to normalized ``(18,)`` float32 vector.

    Raises :class:`CondVecError` when any required source field is missing or
    ``meta.error`` is set. Does not silently substitute zeros for failures.
    """
    if meta.error:
        raise CondVecError(f"cond_vec unavailable for {meta.osu_path}: {meta.error}")

    _require(meta, "official_sr", meta.official_sr)
    _require(meta, "analyzer_ln_percent", meta.analyzer_ln_percent)
    _require(meta, "analyzer_hb_row_ratio", meta.analyzer_hb_row_ratio)
    _require(meta, "analyzer_stream", meta.analyzer_stream)
    _require(meta, "analyzer_chordstream", meta.analyzer_chordstream)
    _require(meta, "analyzer_jacks", meta.analyzer_jacks)
    _require(meta, "analyzer_coordination", meta.analyzer_coordination)
    _require(meta, "analyzer_density", meta.analyzer_density)
    _require(meta, "analyzer_wildcard", meta.analyzer_wildcard)
    _require(meta, "canonical_bpm", meta.canonical_bpm)

    msd_fields = (
        "msd_overall",
        "msd_stream",
        "msd_jumpstream",
        "msd_handstream",
        "msd_stamina",
        "msd_jack_speed",
        "msd_chordjack",
        "msd_technical",
    )
    if require_msd:
        for name in msd_fields:
            _require(meta, name, getattr(meta, name))

    v = np.zeros(COND_VEC_DIM, dtype=np.float32)
    v[0] = float(np.clip(meta.official_sr / 10.0, 0.0, 1.5))  # type: ignore[operator]
    v[1] = float(meta.analyzer_ln_percent)  # type: ignore[arg-type]
    v[2] = float(meta.analyzer_hb_row_ratio)  # type: ignore[arg-type]
    v[3] = float(meta.analyzer_stream)  # type: ignore[arg-type]
    v[4] = float(meta.analyzer_chordstream)  # type: ignore[arg-type]
    v[5] = float(meta.analyzer_jacks)  # type: ignore[arg-type]
    v[6] = float(meta.analyzer_coordination)  # type: ignore[arg-type]
    v[7] = float(meta.analyzer_density)  # type: ignore[arg-type]
    v[8] = float(meta.analyzer_wildcard)  # type: ignore[arg-type]

    if require_msd:
        msd_vals = (
            meta.msd_overall,
            meta.msd_stream,
            meta.msd_jumpstream,
            meta.msd_handstream,
            meta.msd_stamina,
            meta.msd_jack_speed,
            meta.msd_chordjack,
            meta.msd_technical,
        )
        for i, val in enumerate(msd_vals):
            v[9 + i] = float(np.clip(val / 40.0, 0.0, 1.5))  # type: ignore[operator]

    v[17] = float((meta.canonical_bpm - 120.0) / 120.0)  # type: ignore[operator]
    return v
