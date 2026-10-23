"""Fixed 18-d v1 condition vector. Missing fields stay in ``m_real``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio2map.features.cond.chart import ChartMeta

COND_VEC_VERSION = 3
COND_VEC_DIM = 18
COND_MLP_IN = 36
COND_MLP_HIDDEN = 256
SR_DIVISOR = 10.0
MSD_DIVISOR = 40.0
CLIP_MIN = 0.0
CLIP_MAX = 1.5

COND_VEC_NAMES: tuple[str, ...] = (
    "official_sr_norm",
    "rework_sr_norm",
    "msd_overall_norm",
    "msd_stream_norm",
    "msd_jumpstream_norm",
    "msd_handstream_norm",
    "msd_stamina_norm",
    "msd_jack_speed_norm",
    "msd_chordjack_norm",
    "msd_technical_norm",
    "ln_percent",
    "hb_row_ratio",
    "pattern_stream",
    "pattern_chordstream",
    "pattern_jacks",
    "pattern_coordination",
    "pattern_density",
    "pattern_wildcard",
)

# ChartMeta attributes, same order as COND_VEC_NAMES. "sr" and "msd" are clipped.
_SOURCE: tuple[tuple[str, str], ...] = (
    ("official_sr", "sr"),
    ("rework_sr", "sr"),
    ("msd_overall", "msd"),
    ("msd_stream", "msd"),
    ("msd_jumpstream", "msd"),
    ("msd_handstream", "msd"),
    ("msd_stamina", "msd"),
    ("msd_jack_speed", "msd"),
    ("msd_chordjack", "msd"),
    ("msd_technical", "msd"),
    ("analyzer_ln_percent", "ratio"),
    ("analyzer_hb_row_ratio", "ratio"),
    ("pattern_stream", "ratio"),
    ("pattern_chordstream", "ratio"),
    ("pattern_jacks", "ratio"),
    ("pattern_coordination", "ratio"),
    ("pattern_density", "ratio"),
    ("pattern_wildcard", "ratio"),
)


@dataclass(frozen=True, slots=True)
class CondVector:
    """``values`` is x. ``m_real`` is 1 where that field was present."""

    values: np.ndarray
    m_real: np.ndarray


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _scale(value: float, kind: str) -> float:
    if kind == "sr":
        scaled = value / SR_DIVISOR
        return float(np.clip(scaled, CLIP_MIN, CLIP_MAX))
    if kind == "msd":
        scaled = value / MSD_DIVISOR
        return float(np.clip(scaled, CLIP_MIN, CLIP_MAX))
    return value


def build_cond(meta: ChartMeta) -> CondVector:
    """Build x and m_real. A missing field does not reject the chart."""
    values = np.zeros(COND_VEC_DIM, dtype=np.float32)
    mask = np.zeros(COND_VEC_DIM, dtype=np.float32)
    for index, (name, kind) in enumerate(_SOURCE):
        number = _finite(getattr(meta, name))
        if number is None:
            continue
        values[index] = np.float32(_scale(number, kind))
        mask[index] = np.float32(1.0)
    return CondVector(values, mask)
