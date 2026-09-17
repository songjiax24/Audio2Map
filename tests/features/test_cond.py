"""Tests for ``build_cond_vec``."""

from __future__ import annotations

from typing import Any

import pytest

from audio2map.features.cond import (
    COND_VEC_DIM,
    USER_COND_SOURCE_FIELDS,
    USER_COND_VEC_NAMES,
    ChartMeta,
    CondVecError,
    build_cond_vec,
)


def _meta(**overrides: Any) -> ChartMeta:
    fields: dict[str, Any] = {
        "osu_path": "x.osu",
        "beatmap_id": 2,
        "canonical_bpm": 180.0,
        "official_sr": 3.5,
        "msd_overall": 20.0,
        "msd_stream": 18.0,
        "msd_jumpstream": 12.0,
        "msd_handstream": 10.0,
        "msd_stamina": 8.0,
        "msd_jack_speed": 6.0,
        "msd_chordjack": 4.0,
        "msd_technical": 5.0,
        "analyzer_ln_percent": 0.3,
        "analyzer_hb_row_ratio": 0.1,
        "analyzer_stream": 0.5,
        "analyzer_chordstream": 0.2,
        "analyzer_jacks": 0.1,
        "analyzer_coordination": 0.05,
        "analyzer_density": 0.1,
        "analyzer_wildcard": 0.05,
    }
    fields.update(overrides)
    return ChartMeta(**fields)


def test_user_cond_source_fields_align_with_vec_names() -> None:
    assert USER_COND_SOURCE_FIELDS[0] == "official_sr"
    assert USER_COND_SOURCE_FIELDS[1] == "analyzer_ln_percent"
    assert USER_COND_SOURCE_FIELDS[9] == "msd_overall"
    assert len(USER_COND_SOURCE_FIELDS) == len(USER_COND_VEC_NAMES) == 17


def test_build_cond_vec_dim() -> None:
    v = build_cond_vec(_meta())
    assert v.shape == (COND_VEC_DIM,)
    assert COND_VEC_DIM == 18
    assert v[0] == 0.35
    assert v[1] == 0.3
    assert v[9] == 0.5
    assert v[17] == 0.5


def test_build_cond_vec_raises_on_meta_error() -> None:
    with pytest.raises(CondVecError, match="msd: runner failed"):
        build_cond_vec(_meta(error="msd: runner failed"))


def test_build_cond_vec_raises_on_missing_field() -> None:
    with pytest.raises(CondVecError, match="missing official_sr"):
        build_cond_vec(_meta(official_sr=None))
    with pytest.raises(CondVecError, match="missing msd_overall"):
        build_cond_vec(_meta(msd_overall=None))
