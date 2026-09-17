"""Tests for generate/service.py — pure helpers with synthetic fixtures."""

from __future__ import annotations

import random
import zipfile
from pathlib import Path

import numpy as np
import pytest

from audio2map.features.cond import CANONICAL_BPM_NORM_INDEX, COND_VEC_DIM, ChartMeta, build_cond_vec
from audio2map.generate.service import (
    CondCandidate,
    CondSelectionError,
    build_final_cond_vec,
    canonical_bpm_norm_from_bpm,
    pack_osz,
    select_candidate_from_ranges,
    select_condition,
    timing_from_bpm_offset,
    timing_from_osu,
)

_TIMING_OSU = """\
osu file format v14

[General]
AudioFilename:song.mp3
Mode:0

[Metadata]
Title:Song
Artist:Artist
Creator:Mapper
Version:Insane

[Difficulty]
CircleSize:4

[TimingPoints]
980,500,4,2,0,50,1,0

[HitObjects]
256,192,980,1,0,0:0:0:0:
"""


def test_canonical_bpm_norm_and_timing_from_bpm_offset() -> None:
    canonical_bpm, norm, scale_exp = canonical_bpm_norm_from_bpm(120.0)
    assert canonical_bpm == pytest.approx(120.0)
    assert norm == pytest.approx(0.0)
    assert scale_exp == 0

    timing = timing_from_bpm_offset(bpm=180.0, offset_ms=980.4)
    assert timing.offset_ms == 980
    assert timing.original_bpm == pytest.approx(180.0)
    assert timing.canonical_bpm == pytest.approx(180.0)


def test_timing_from_osu(tmp_path: Path) -> None:
    src = tmp_path / "timing.osu"
    src.write_text(_TIMING_OSU, encoding="utf-8")
    timing = timing_from_osu(src)
    assert timing.offset_ms == 980
    assert timing.original_bpm == pytest.approx(120.0)

    bad = tmp_path / "empty.osu"
    bad.write_text("osu file format v14\n\n[General]\nMode:0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no timing points"):
        timing_from_osu(bad)


def _meta(osu_path: str, *, sr: float, ln: float, stream: float) -> ChartMeta:
    return ChartMeta(
        osu_path=osu_path,
        beatmap_id=2,
        canonical_bpm=180.0,
        official_sr=sr,
        msd_overall=20.0,
        msd_stream=18.0,
        msd_jumpstream=12.0,
        msd_handstream=10.0,
        msd_stamina=8.0,
        msd_jack_speed=6.0,
        msd_chordjack=4.0,
        msd_technical=5.0,
        analyzer_ln_percent=ln,
        analyzer_hb_row_ratio=0.1,
        analyzer_stream=stream,
        analyzer_chordstream=0.2,
        analyzer_jacks=0.1,
        analyzer_coordination=0.05,
        analyzer_density=0.1,
        analyzer_wildcard=0.05,
    )


def _candidate(name: str, *, sr: float, ln: float, stream: float) -> CondCandidate:
    meta = _meta(f"/data/{name}.osu", sr=sr, ln=ln, stream=stream)
    return CondCandidate(
        chart_id=name,
        osu_path=meta.osu_path,
        cond_vec=build_cond_vec(meta),
        meta=meta,
    )


def _wide_ranges(**overrides: tuple[float, float]) -> dict[str, tuple[float, float]]:
    from audio2map.generate.service import USER_COND_SOURCE_FIELDS

    ranges: dict[str, tuple[float, float]] = {n: (-10.0, 100.0) for n in USER_COND_SOURCE_FIELDS}
    ranges.update(overrides)
    return ranges


def test_select_condition_samples_match_and_sets_bpm_dim() -> None:
    inside = _candidate("inside", sr=5.0, ln=0.2, stream=0.8)
    also = _candidate("also", sr=5.05, ln=0.1, stream=0.9)
    outside = _candidate("outside", sr=5.0, ln=0.9, stream=0.1)
    cands = [inside, also, outside]
    ranges = _wide_ranges(
        analyzer_ln_percent=(0.0, 0.4),
        analyzer_stream=(0.6, 1.0),
    )
    rng = random.Random(0)
    picked: set[str] = set()
    for _ in range(32):
        selected = select_candidate_from_ranges(cands, ranges, rng=rng)
        assert selected is not None
        picked.add(selected.chart_id)
    assert picked == {"inside", "also"}

    selected = select_candidate_from_ranges(cands, ranges, rng=random.Random(1))
    assert selected is not None
    final = build_final_cond_vec(selected, 0.25)
    assert final.shape == (COND_VEC_DIM,)
    assert final[CANONICAL_BPM_NORM_INDEX] == pytest.approx(0.25)
    assert selected.cond_vec[CANONICAL_BPM_NORM_INDEX] == pytest.approx(0.5)

    payload = select_condition(cands, ranges, canonical_bpm_norm=0.25, rng=random.Random(1))
    assert payload["selected_chart_id"] in {"inside", "also"}
    assert payload["final_cond_vec"][CANONICAL_BPM_NORM_INDEX] == pytest.approx(0.25)
    assert payload["matched_candidate_count"] == 2
    assert payload["selection_rule"] == "uniform_among_matches"
    assert payload["selected_source_values"]["official_sr"] == pytest.approx(
        {"inside": 5.0, "also": 5.05}[payload["selected_chart_id"]]
    )
    assert payload["selected_source_values"]["analyzer_ln_percent"] <= 0.4


def test_select_condition_no_match_raises() -> None:
    cands = [_candidate("a", sr=9.0, ln=0.9, stream=0.1)]
    ranges = _wide_ranges(analyzer_ln_percent=(0.0, 0.1))
    assert select_candidate_from_ranges(cands, ranges) is None
    with pytest.raises(CondSelectionError, match="no manifest charts match"):
        select_condition(cands, ranges, canonical_bpm_norm=0.0)


def test_select_condition_rejects_bpm_range() -> None:
    cands = [_candidate("a", sr=5.0, ln=0.2, stream=0.5)]
    ranges = _wide_ranges()
    ranges["canonical_bpm_norm"] = (0.0, 1.0)
    with pytest.raises(ValueError, match="canonical_bpm_norm"):
        select_candidate_from_ranges(cands, ranges)


def test_select_condition_missing_range_key() -> None:
    cands = [_candidate("a", sr=5.0, ln=0.2, stream=0.5)]
    ranges = _wide_ranges()
    del ranges["official_sr"]
    with pytest.raises(ValueError, match="missing cond dims"):
        select_candidate_from_ranges(cands, ranges)


def test_select_condition_matches_original_meta_units() -> None:
    cand = _candidate("a", sr=5.0, ln=0.2, stream=0.5)
    assert select_candidate_from_ranges([cand], _wide_ranges(official_sr=(4.0, 6.0))) is not None
    assert select_candidate_from_ranges([cand], _wide_ranges(official_sr=(0.4, 0.6))) is None
    assert select_candidate_from_ranges([cand], _wide_ranges(msd_overall=(19.0, 21.0))) is not None
    assert select_candidate_from_ranges([cand], _wide_ranges(msd_overall=(0.4, 0.6))) is None


def test_pack_osz_roundtrip(tmp_path: Path) -> None:
    osu = tmp_path / "chart.osu"
    osu.write_text(_TIMING_OSU, encoding="utf-8")
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"\xff" * 2048)
    bg = tmp_path / "bg.jpg"
    bg.write_bytes(b"\x00" * 16)

    out = tmp_path / "pack" / "chart.osz"
    pack_osz(osu, audio, out, extra_files=[bg, tmp_path / "missing.png"])
    with zipfile.ZipFile(out, "r") as zf:
        names = set(zf.namelist())
        assert names == {"chart.osu", "song.mp3", "bg.jpg"}
        assert zf.read("song.mp3") == b"\xff" * 2048
