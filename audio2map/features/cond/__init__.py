"""Chart conditions: the per-chart table and the 18-d model vector."""

from audio2map.features.cond.chart import ChartMeta, compute_chart_meta, load_chart_meta_manifest
from audio2map.features.cond.patterns import PatternFeatures, analyze_chart_patterns
from audio2map.features.cond.vec import (
    CANONICAL_BPM_NORM_INDEX,
    COND_VEC_DIM,
    COND_VEC_NAMES,
    COND_VEC_VERSION,
    USER_COND_SOURCE_FIELDS,
    USER_COND_VEC_NAMES,
    CondVecError,
    build_cond_vec,
    canonical_bpm_norm_from_bpm,
    canonical_bpm_to_norm,
)

__all__ = [
    "CANONICAL_BPM_NORM_INDEX",
    "COND_VEC_DIM",
    "COND_VEC_NAMES",
    "COND_VEC_VERSION",
    "ChartMeta",
    "CondVecError",
    "PatternFeatures",
    "USER_COND_SOURCE_FIELDS",
    "USER_COND_VEC_NAMES",
    "analyze_chart_patterns",
    "build_cond_vec",
    "canonical_bpm_norm_from_bpm",
    "canonical_bpm_to_norm",
    "compute_chart_meta",
    "load_chart_meta_manifest",
]
