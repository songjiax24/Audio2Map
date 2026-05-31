"""23-dim condition vector for v2 training (see docs/v2_spec.md §7)."""

from __future__ import annotations

import numpy as np

from audio2map.difficulty.chart_meta import ChartMeta

COND_VEC_DIM = 23

# Index map for debugging / logging.
COND_VEC_NAMES: tuple[str, ...] = (
    "osu_sr_norm",
    "hold_ratio_norm",
    "hold_coverage_norm",
    "analyzer_ln_percent",
    "analyzer_hb_row_ratio",
    "analyzer_stream",
    "analyzer_chordstream",
    "analyzer_jacks",
    "analyzer_coordination",
    "analyzer_density",
    "analyzer_wildcard",
    "analyzer_available",
    "msd_overall_norm",
    "msd_stream_norm",
    "msd_jumpstream_norm",
    "msd_handstream_norm",
    "msd_stamina_norm",
    "msd_jack_speed_norm",
    "msd_chordjack_norm",
    "msd_technical_norm",
    "msd_available",
    "canonical_bpm_norm",
    "bpm_scale_exp_norm",
)


def build_cond_vec(meta: ChartMeta) -> np.ndarray:
    """Map ``ChartMeta`` to a normalized ``(23,)`` float32 vector."""
    v = np.zeros(COND_VEC_DIM, dtype=np.float32)

    if meta.official_sr is not None:
        v[0] = float(np.clip(meta.official_sr / 10.0, 0.0, 1.5))

    v[1] = float(np.clip(meta.hold_ratio, 0.0, 1.0))
    v[2] = float(np.clip(meta.hold_coverage, 0.0, 1.0))

    if meta.analyzer_available:
        v[3] = float(meta.analyzer_ln_percent or 0.0)
        v[4] = float(meta.analyzer_hb_row_ratio or 0.0)
        v[5] = float(meta.analyzer_stream or 0.0)
        v[6] = float(meta.analyzer_chordstream or 0.0)
        v[7] = float(meta.analyzer_jacks or 0.0)
        v[8] = float(meta.analyzer_coordination or 0.0)
        v[9] = float(meta.analyzer_density or 0.0)
        v[10] = float(meta.analyzer_wildcard or 0.0)
    v[11] = float(meta.analyzer_available)

    if meta.msd_available:
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
            if val is not None:
                v[12 + i] = float(np.clip(val / 40.0, 0.0, 1.5))
    v[20] = float(meta.msd_available)

    if meta.canonical_bpm is not None:
        v[21] = float((meta.canonical_bpm - 120.0) / 120.0)
    if meta.bpm_scale_exp is not None:
        v[22] = float(meta.bpm_scale_exp / 4.0)

    return v
