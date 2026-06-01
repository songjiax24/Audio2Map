"""23-dim condition vector for v2 training (V2_MASTER_SPEC.md §14.1)."""

from __future__ import annotations

import numpy as np

from audio2map.difficulty.chart_meta import ChartMeta

COND_VEC_DIM = 23

# Fixed index map — do not reorder without spec update.
COND_VEC_NAMES: tuple[str, ...] = (
    "osu_sr_norm",
    "hold_ratio",
    "hold_coverage",
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
    "bpm_scale_exp_norm",
    "analyzer_available",
    "msd_available",
)


def build_cond_vec(meta: ChartMeta) -> np.ndarray:
    """Map ``ChartMeta`` to normalized ``(23,)`` float32 vector."""
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
    v[21] = float(meta.analyzer_available)

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
                v[11 + i] = float(np.clip(val / 40.0, 0.0, 1.5))
    v[22] = float(meta.msd_available)

    if meta.canonical_bpm is not None:
        v[19] = float((meta.canonical_bpm - 120.0) / 120.0)
    if meta.bpm_scale_exp is not None:
        v[20] = float(meta.bpm_scale_exp / 4.0)

    return v
