"""v1 condition vector. v0 ``build_cond_vec`` stays on ``features.cond.vec``."""

from audio2map.features.cond.v1.mask import DROP_ALL_P, KEEP_ALL_P, P_KEEP, sample_cond_mask
from audio2map.features.cond.v1.patterns import (
    PatternPass,
    analyze_patterns_once,
    PATTERN_ORDER,
    amounts_from_found,
    union_ms,
)
from audio2map.features.cond.v1.rework import ReworkSRError, rework_star_rating
from audio2map.features.cond.v1.vec import (
    CLIP_MAX,
    CLIP_MIN,
    COND_MLP_HIDDEN,
    COND_MLP_IN,
    COND_VEC_DIM,
    COND_VEC_NAMES,
    COND_VEC_VERSION,
    MSD_DIVISOR,
    SR_DIVISOR,
    CondVector,
    build_cond,
)

__all__ = [
    "CLIP_MAX",
    "CLIP_MIN",
    "COND_MLP_HIDDEN",
    "COND_MLP_IN",
    "COND_VEC_DIM",
    "COND_VEC_NAMES",
    "COND_VEC_VERSION",
    "DROP_ALL_P",
    "KEEP_ALL_P",
    "MSD_DIVISOR",
    "P_KEEP",
    "SR_DIVISOR",
    "CondVector",
    "ReworkSRError",
    "PatternPass",
    "analyze_patterns_once",
    "build_cond",
    "rework_star_rating",
    "sample_cond_mask",
    "PATTERN_ORDER",
    "amounts_from_found",
    "union_ms",
]
