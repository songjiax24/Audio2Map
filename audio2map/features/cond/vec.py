"""18-dim condition vector for the encoder/decoder."""

from __future__ import annotations

import numpy as np

from audio2map.features.cond.chart import ChartMeta
from audio2map.grid import canonicalize_bpm

COND_VEC_DIM = 18
COND_VEC_VERSION = 1

# Fixed index map — do not reorder without bumping COND_VEC_VERSION.
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

CANONICAL_BPM_NORM_INDEX = COND_VEC_NAMES.index("canonical_bpm_norm")
# Internal vector names for dims 0–16 (not ChartMeta field names).
USER_COND_VEC_NAMES: tuple[str, ...] = COND_VEC_NAMES[:CANONICAL_BPM_NORM_INDEX]


def canonical_bpm_to_norm(canonical_bpm: float) -> float:
    """Map canonical BPM to cond dim 17: ``(bpm - 120) / 120``."""
    return float((canonical_bpm - 120.0) / 120.0)


def canonical_bpm_norm_from_bpm(bpm: float) -> tuple[float, float, int]:
    """Return ``(canonical_bpm, canonical_bpm_norm, bpm_scale_exp)``."""
    canonical_bpm, scale_exp = canonicalize_bpm(bpm)
    return float(canonical_bpm), canonical_bpm_to_norm(canonical_bpm), scale_exp

# ChartMeta fields for cond indices 1–8 and 9–16. Vector names in COND_VEC_NAMES
# differ (``*_ratio`` / ``etterna_*_norm``).
_ANALYZER_FIELDS = (
    "analyzer_ln_percent",
    "analyzer_hb_row_ratio",
    "analyzer_stream",
    "analyzer_chordstream",
    "analyzer_jacks",
    "analyzer_coordination",
    "analyzer_density",
    "analyzer_wildcard",
)
_MSD_FIELDS = (
    "msd_overall",
    "msd_stream",
    "msd_jumpstream",
    "msd_handstream",
    "msd_stamina",
    "msd_jack_speed",
    "msd_chordjack",
    "msd_technical",
)

# ChartMeta fields for dims 0–16 (same order as USER_COND_VEC_NAMES). Range
# filters use these original units; ``build_cond_vec`` is the only place that
# normalizes.
USER_COND_SOURCE_FIELDS: tuple[str, ...] = ("official_sr",) + _ANALYZER_FIELDS + _MSD_FIELDS
if len(USER_COND_SOURCE_FIELDS) != len(USER_COND_VEC_NAMES):
    raise RuntimeError("USER_COND_SOURCE_FIELDS must align with USER_COND_VEC_NAMES")


class CondVecError(ValueError):
    """``ChartMeta`` is missing required difficulty fields for ``cond_vec``."""


def _float(meta: ChartMeta, field: str) -> float:
    value = getattr(meta, field)
    if value is None:
        raise CondVecError(f"cond_vec unavailable for {meta.osu_path}: missing {field}")
    return float(value)


def build_cond_vec(meta: ChartMeta) -> np.ndarray:
    """Map ``ChartMeta`` to normalized ``(18,)`` float32.

    Raises :class:`CondVecError` when ``meta.error`` is set or a required field
    is missing. Does not fill missing values with zeros.
    """
    if meta.error:
        raise CondVecError(f"cond_vec unavailable for {meta.osu_path}: {meta.error}")

    v = np.zeros(COND_VEC_DIM, dtype=np.float32)
    v[0] = float(np.clip(_float(meta, "official_sr") / 10.0, 0.0, 1.5))
    for i, name in enumerate(_ANALYZER_FIELDS):
        v[1 + i] = _float(meta, name)
    for i, name in enumerate(_MSD_FIELDS):
        v[9 + i] = float(np.clip(_float(meta, name) / 40.0, 0.0, 1.5))
    v[17] = canonical_bpm_to_norm(_float(meta, "canonical_bpm"))
    return v
