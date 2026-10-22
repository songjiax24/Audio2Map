"""Redline field defaults and one-reason rejects."""

from __future__ import annotations

import pytest

from audio2map.utils.errors import (
    INVALID_TIMING_FIELD,
    NO_REDLINE,
    NON_FINITE_OFFSET,
    NONPOSITIVE_BEAT_LENGTH,
    ChartRejected,
)
from audio2map.osu.v1.timing import DEFAULT_METER, parse_redlines


def test_short_line_is_a_redline_with_meter_4() -> None:
    parsed = parse_redlines(["0,500"])
    assert len(parsed.redlines) == 1
    assert parsed.redlines[0].beat_length_ms == 500.0
    assert parsed.redlines[0].meter == DEFAULT_METER


def test_empty_meter_defaults_to_4() -> None:
    parsed = parse_redlines(["0,500,,2,0,50,1,0"])
    assert parsed.redlines[0].meter == 4


def test_non_numeric_meter_defaults_to_4() -> None:
    parsed = parse_redlines(["0,500,abc,2,0,50,1,0"])
    assert parsed.redlines[0].meter == 4


def test_green_only_chart_has_no_redline() -> None:
    with pytest.raises(ChartRejected) as caught:
        parse_redlines(["0,-100,4,2,0,50,0,0"])
    assert caught.value.reason == NO_REDLINE


def test_negative_green_line_does_not_reject_a_later_red_line() -> None:
    parsed = parse_redlines(["0,-100,4,2,0,50,0,0", "1000,500,4,2,0,50,1,0"])
    assert len(parsed.redlines) == 1
    assert parsed.redlines[0].offset_ms == 1000.0


def test_nonpositive_beat_length_is_its_own_reason() -> None:
    with pytest.raises(ChartRejected) as caught:
        parse_redlines(["0,0,4,2,0,50,1,0"])
    assert caught.value.reason == NONPOSITIVE_BEAT_LENGTH


def test_non_numeric_offset_is_not_a_beat_length_error() -> None:
    with pytest.raises(ChartRejected) as caught:
        parse_redlines(["abc,500,4,2,0,50,1,0"])
    assert caught.value.reason == INVALID_TIMING_FIELD


def test_non_finite_offset_is_its_own_reason() -> None:
    with pytest.raises(ChartRejected) as caught:
        parse_redlines(["nan,500,4,2,0,50,1,0"])
    assert caught.value.reason == NON_FINITE_OFFSET
