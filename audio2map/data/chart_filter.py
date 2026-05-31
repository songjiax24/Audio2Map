"""Phase 1 chart eligibility filtering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from audio2map.osu.mania import is_mania_4k_sections
from audio2map.osu.parser import parse_beatmap, parse_sections, parse_timing_points
from audio2map.osu.schema import Beatmap
from audio2map.osu.timing import summarize_beatmap_timing, summarize_timing


class FilterReason(str, Enum):
    NOT_4K = "not_4k"
    NO_TIMING = "no_timing"
    VARIABLE_BPM = "variable_bpm"
    VARIABLE_METER = "variable_meter"
    METER_NOT_4 = "meter_not_4"


REQUIRED_METER = 4


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    eligible: bool
    reason: FilterReason | None = None


def check_beatmap_eligibility(beatmap: Beatmap) -> EligibilityResult:
    timing = summarize_beatmap_timing(beatmap)
    return _eligibility_from_timing(timing)


def check_osu_path(path: Path | str) -> EligibilityResult:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        sections = parse_sections(text)
    except OSError:
        return EligibilityResult(False, FilterReason.NOT_4K)

    if not is_mania_4k_sections(sections):
        return EligibilityResult(False, FilterReason.NOT_4K)

    timing = summarize_timing(parse_timing_points(sections.get("[TimingPoints]", [])))
    return _eligibility_from_timing(timing)


def check_parsed_beatmap(path: Path | str) -> EligibilityResult:
    """Full parse (stricter 4K validation via parser)."""
    try:
        beatmap = parse_beatmap(path)
    except (ValueError, OSError):
        return EligibilityResult(False, FilterReason.NOT_4K)
    return check_beatmap_eligibility(beatmap)


def _eligibility_from_timing(timing) -> EligibilityResult:
    if timing.n_uninherited == 0:
        return EligibilityResult(False, FilterReason.NO_TIMING)
    if not timing.constant_bpm:
        return EligibilityResult(False, FilterReason.VARIABLE_BPM)
    if not timing.constant_meter:
        return EligibilityResult(False, FilterReason.VARIABLE_METER)
    if timing.meter_primary != REQUIRED_METER:
        return EligibilityResult(False, FilterReason.METER_NOT_4)
    return EligibilityResult(True)
