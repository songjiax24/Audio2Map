"""v1 condition vector, clip, mask, and ReworkSR."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audio2map.features.cond import ChartMeta
from audio2map.features.cond.v1.mask import sample_cond_mask
from audio2map.features.cond.pattern_analyser.find_patterns import FoundPattern
from audio2map.features.cond.pattern_analyser.patterns_def import CorePattern
from audio2map.features.cond.v1.patterns import (
    analyze_patterns_once,
    amounts_from_found,
    union_ms,
)
from audio2map.features.cond.v1.rework import rework_star_rating
from audio2map.features.cond.v1.vec import (
    COND_VEC_DIM,
    COND_VEC_NAMES,
    COND_VEC_VERSION,
    CondVector,
    build_cond,
)


class _Draws:
    """``random()`` then ``random(shape)`` from a fixed list."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def random(self, size=None):
        if size is None:
            return self._values.pop(0)
        count = int(np.prod(size))
        drawn = np.asarray(self._values[:count], dtype=np.float64)
        del self._values[:count]
        return drawn.reshape(size)


def _meta(**overrides: float | None) -> ChartMeta:
    fields: dict[str, float | None] = {
        "official_sr": 3.5,
        "rework_sr": 7.5,
        "analyzer_ln_percent": 0.3,
        "analyzer_hb_row_ratio": 0.1,
        "pattern_stream": 0.4,
        "pattern_chordstream": 0.2,
        "pattern_jacks": 0.1,
        "pattern_coordination": 0.1,
        "pattern_density": 0.1,
        "pattern_wildcard": 0.1,
        "msd_overall": 20.0,
        "msd_stream": 18.0,
        "msd_jumpstream": 12.0,
        "msd_handstream": 10.0,
        "msd_stamina": 8.0,
        "msd_jack_speed": 6.0,
        "msd_chordjack": 4.0,
        "msd_technical": 5.0,
    }
    fields.update(overrides)
    return ChartMeta(osu_path="x.osu", canonical_bpm=240.0, **fields)


def test_names_and_version() -> None:
    assert COND_VEC_VERSION == 3
    assert COND_VEC_NAMES == (
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
    assert len(COND_VEC_NAMES) == COND_VEC_DIM == 18
    assert "canonical_bpm_norm" not in COND_VEC_NAMES


def test_clip_and_rework_enters() -> None:
    vector = build_cond(_meta(official_sr=20.0, rework_sr=7.5, msd_overall=80.0, msd_stream=-4.0))
    assert isinstance(vector, CondVector)
    names = COND_VEC_NAMES
    assert vector.values[names.index("official_sr_norm")] == np.float32(1.5)
    assert vector.values[names.index("rework_sr_norm")] == np.float32(0.75)
    assert vector.m_real[names.index("rework_sr_norm")] == 1.0
    assert vector.values[names.index("msd_overall_norm")] == np.float32(1.5)
    assert vector.values[names.index("msd_stream_norm")] == np.float32(0.0)
    assert vector.values[names.index("ln_percent")] == np.float32(0.3)
    assert vector.values[names.index("hb_row_ratio")] == np.float32(0.1)
    assert vector.values.shape == (18,)


def test_missing_field_stays_off_without_raising() -> None:
    vector = build_cond(_meta(rework_sr=None, official_sr=None))
    assert vector.m_real[0] == 0.0
    assert vector.m_real[1] == 0.0
    assert vector.values[0] == 0.0
    assert vector.m_real[COND_VEC_NAMES.index("ln_percent")] == 1.0
    assert vector.m_real[COND_VEC_NAMES.index("msd_overall_norm")] == 1.0


def test_mask_and_m_real() -> None:
    real = np.ones(18, dtype=np.float32)
    real[1] = 0.0
    real[5] = 0.0
    kept = sample_cond_mask(real, _Draws([0.0]))
    assert kept[0] == 1.0
    assert kept[1] == 0.0
    assert kept[5] == 0.0
    empty = sample_cond_mask(real, _Draws([0.15]))
    assert np.all(empty == 0.0)
    draws = [0.5] + [0.1] * 18
    mixed = sample_cond_mask(real, _Draws(draws))
    assert mixed[1] == 0.0
    assert mixed[5] == 0.0
    assert mixed[0] == 1.0


def _tiny_osu(path: Path, *, overall: str | None) -> None:
    difficulty = ["CircleSize:4"]
    if overall is not None:
        difficulty.append(f"OverallDifficulty:{overall}")
    path.write_text(
        "\n".join(
            [
                "osu file format v14",
                "[General]",
                "Mode: 3",
                "[Metadata]",
                "Title: t",
                "Artist: a",
                "Creator: c",
                "Version: v",
                "[Difficulty]",
                *difficulty,
                "[TimingPoints]",
                "0,500,4,2,0,100,1,0",
                "[HitObjects]",
                "64,192,1000,1,0,0:0:0:0:",
                "192,192,1125,1,0,0:0:0:0:",
                "320,192,1250,1,0,0:0:0:0:",
                "448,192,1375,1,0,0:0:0:0:",
                "64,192,1500,1,0,0:0:0:0:",
                "192,192,1625,128,0,1800:0:0:0:0:",
            ]
        ),
        encoding="utf-8",
    )


def test_rework_ignores_chart_od(tmp_path: Path) -> None:
    low = tmp_path / "od0.osu"
    high = tmp_path / "od10.osu"
    _tiny_osu(low, overall="0")
    _tiny_osu(high, overall="10")
    assert np.isclose(rework_star_rating(low), rework_star_rating(high))


def test_rework_without_overall_difficulty(tmp_path: Path) -> None:
    path = tmp_path / "no_od.osu"
    _tiny_osu(path, overall=None)
    star = rework_star_rating(path)
    assert np.isfinite(star)
    assert star > 0.0


def test_rework_on_a_small_chart(tmp_path: Path) -> None:
    path = tmp_path / "tiny.osu"
    _tiny_osu(path, overall="8")
    star = rework_star_rating(path)
    assert np.isfinite(star)
    assert star > 0.0
    vector = build_cond(ChartMeta(osu_path=str(path), rework_sr=star))
    assert vector.m_real[1] == 1.0
    assert vector.values[1] == np.float32(np.clip(star / 10.0, 0.0, 1.5))


def _found(pattern: CorePattern, start: float, end: float) -> FoundPattern:
    return FoundPattern(pattern, None, False, start, end, 100.0)


def test_pattern_union_does_not_double_count() -> None:
    assert union_ms([(0.0, 10.0), (5.0, 20.0)]) == 20.0
    assert union_ms([(0.0, 20.0), (5.0, 10.0)]) == 20.0
    found = [
        _found(CorePattern.Stream, 0.0, 10.0),
        _found(CorePattern.Stream, 5.0, 20.0),
        _found(CorePattern.Jacks, 0.0, 50.0),
    ]
    amounts = amounts_from_found(found, 50.0)
    assert amounts is not None
    assert amounts[0] == np.float64(20.0 / 50.0)
    assert amounts[2] == np.float64(1.0)
    assert float(amounts.sum()) > 1.0
    assert amounts_from_found(found, 0.0) is None


def test_amount_clip_is_per_pattern() -> None:
    over = amounts_from_found(
        [
            _found(CorePattern.Stream, 0.0, 20.0),
            _found(CorePattern.Jacks, 0.0, 15.0),
        ],
        10.0,
    )
    assert over is not None
    assert over[0] == 1.0
    assert over[2] == 1.0
    assert float(over.sum()) == 2.0
    under = amounts_from_found([_found(CorePattern.Stream, 0.0, 4.0)], 10.0)
    assert under is not None
    assert under[0] == np.float64(0.4)


def test_pattern_amounts_skip_cluster_filters(monkeypatch) -> None:
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("scheme A must not cluster")

    monkeypatch.setattr(
        "audio2map.features.cond.pattern_analyser.clustering.calculate_clustered_patterns",
        _forbidden,
    )
    amounts = amounts_from_found([_found(CorePattern.Stream, 0.0, 10.0)], 40.0)
    assert amounts is not None
    assert amounts[0] == np.float64(0.25)


def test_analyze_once_keeps_importance_and_amount(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "once.osu"
    _tiny_osu(path, overall="8")
    import audio2map.features.cond.pattern_analyser.summary as summary
    import audio2map.features.cond.v1.patterns as patterns

    parse_calls = {"n": 0}
    find_calls = {"n": 0}
    real_parse = patterns.parse_osu_mania
    real_find = patterns.find
    real_summary_find = summary.find

    def _parse(text: str):
        parse_calls["n"] += 1
        return real_parse(text)

    def _find(chart):
        find_calls["n"] += 1
        return real_find(chart)

    def _second_find(*_args, **_kwargs):
        raise AssertionError("pattern search ran twice")

    monkeypatch.setattr(patterns, "parse_osu_mania", _parse)
    monkeypatch.setattr(patterns, "find", _find)
    monkeypatch.setattr(summary, "find", _second_find)
    passed = analyze_patterns_once(path)
    assert parse_calls["n"] == 1
    assert find_calls["n"] == 1
    assert "calculate_clustered_patterns" not in amounts_from_found.__code__.co_names
    assert set(passed.features.core_dist) == {
        "Stream",
        "Chordstream",
        "Jacks",
        "Coordination",
        "Density",
        "Wildcard",
    }
    assert passed.pattern_amounts is not None
    assert passed.pattern_amounts.shape == (6,)
    monkeypatch.setattr(summary, "find", real_summary_find)
    from audio2map.features.cond.patterns import analyze_chart_patterns

    again = analyze_chart_patterns(path)
    assert again.core_dist == passed.features.core_dist
