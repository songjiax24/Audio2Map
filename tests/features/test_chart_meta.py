"""Tests for ``ChartMeta`` (official SR, Etterna MSD, manifest)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio2map.features.cond import ChartMeta, compute_chart_meta
from audio2map.features.cond.minacalc import compute_msd
from audio2map.features.cond.official_sr import official_star_rating
from audio2map.osu.parser import parse_beatmap
from audio2map.utils.paths import raw_data_available, raw_dir


def _sample_osu() -> Path | None:
    if not raw_data_available():
        return None
    root = raw_dir()
    try:
        for path in root.rglob("*.osu"):
            return path
    except OSError:
        return None
    return None


@pytest.fixture
def sample_osu() -> Path:
    path = _sample_osu()
    if path is None:
        pytest.skip("no raw .osu files available")
    return path


@pytest.mark.requires_data
def test_official_sr_positive(sample_osu: Path) -> None:
    sr = official_star_rating(sample_osu)
    assert sr >= 0.0


@pytest.mark.requires_data
def test_msd_eight_skillsets(sample_osu: Path) -> None:
    bm = parse_beatmap(sample_osu)
    msd = compute_msd(bm)
    assert msd.overall >= 0.0
    assert msd.stream >= 0.0


@pytest.mark.requires_data
def test_compute_chart_meta_no_error(sample_osu: Path) -> None:
    meta = compute_chart_meta(sample_osu)
    assert meta.error is None
    assert meta.official_sr is not None
    assert meta.msd_overall is not None
    assert meta.canonical_bpm is not None
    assert "hold_ratio" not in meta.to_dict()
    assert "hold_coverage" not in meta.to_dict()
    assert "constant_bpm" not in meta.to_dict()
    assert "note_count" not in meta.to_dict()
    assert "set_id" not in meta.to_dict()
    assert "version" not in meta.to_dict()


def test_chart_meta_from_dict_ignores_unknown_keys() -> None:
    meta = ChartMeta.from_dict(
        {
            "osu_path": "x.osu",
            "set_id": 1,
            "beatmap_id": 2,
            "version": "Normal",
            "constant_bpm": True,
            "bpm": 180.0,
            "original_bpm": 180.0,
            "canonical_bpm": 180.0,
            "bpm_scale_exp": 0,
            "offset_ms": 980,
            "meter": 4,
            "note_count": 100,
            "hold_ratio": 0.9,
            "hold_coverage": 0.8,
            "analyzer_available": 1,
            "msd_available": 1,
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
    )
    assert "hold_ratio" not in meta.to_dict()
    assert "hold_coverage" not in meta.to_dict()
    assert "constant_bpm" not in meta.to_dict()
    assert "note_count" not in meta.to_dict()
    assert "set_id" not in meta.to_dict()
    assert "version" not in meta.to_dict()
    assert meta.canonical_bpm == 180.0
    assert meta.official_sr == 3.5
