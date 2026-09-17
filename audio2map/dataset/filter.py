"""Chart eligibility: constant 4/4 BPM, 4K notes that encode on the grid."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from audio2map.grid import CanonicalTiming
from audio2map.osu.parser import (
    InvalidHitObjectError,
    InvalidTimingPointError,
    parse_beatmap,
)
from audio2map.osu.schema import Beatmap, ManiaNote, TimingPoint
from audio2map.osu.timing import TimingSummary, summarize_beatmap_timing
from audio2map.tokens import encode_notes
from audio2map.utils.paths import raw_dir


class FilterReason(str, Enum):
    NOT_4K = "not_4k"
    NO_TIMING = "no_timing"
    VARIABLE_BPM = "variable_bpm"
    VARIABLE_METER = "variable_meter"
    METER_NOT_4 = "meter_not_4"
    NO_NOTES = "no_notes"
    INVALID_HIT_OBJECTS = "invalid_hit_objects"
    INVALID_TIMING_POINTS = "invalid_timing_points"
    INVALID_GRID = "invalid_grid"


REQUIRED_METER = 4


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    eligible: bool
    reason: FilterReason | None = None


def check_beatmap_eligibility(beatmap: Beatmap) -> EligibilityResult:
    timing = summarize_beatmap_timing(beatmap)
    result = _eligibility_from_timing(timing)
    if not result.eligible:
        return result
    if beatmap.note_count == 0:
        return EligibilityResult(False, FilterReason.NO_NOTES)
    return _eligibility_from_grid(beatmap.notes, beatmap.timing_points)


def check_osu_path(path: Path | str) -> EligibilityResult:
    """Parse ``path`` and run :func:`check_beatmap_eligibility`."""
    try:
        beatmap = parse_beatmap(path)
    except InvalidHitObjectError:
        return EligibilityResult(False, FilterReason.INVALID_HIT_OBJECTS)
    except InvalidTimingPointError:
        return EligibilityResult(False, FilterReason.INVALID_TIMING_POINTS)
    except (ValueError, OSError):
        return EligibilityResult(False, FilterReason.NOT_4K)
    return check_beatmap_eligibility(beatmap)


def list_eligible_osu_paths(root: Path | None = None) -> list[Path]:
    root = raw_dir() if root is None else Path(root)
    return [path for path in sorted(root.rglob("*.osu")) if check_osu_path(path).eligible]


def _eligibility_from_timing(timing: TimingSummary) -> EligibilityResult:
    if timing.n_uninherited == 0:
        return EligibilityResult(False, FilterReason.NO_TIMING)
    if not timing.constant_bpm:
        return EligibilityResult(False, FilterReason.VARIABLE_BPM)
    if not timing.constant_meter:
        return EligibilityResult(False, FilterReason.VARIABLE_METER)
    if timing.meter != REQUIRED_METER:
        return EligibilityResult(False, FilterReason.METER_NOT_4)
    return EligibilityResult(True)


def _eligibility_from_grid(
    notes: list[ManiaNote],
    timing_points: list[TimingPoint],
) -> EligibilityResult:
    try:
        timing = CanonicalTiming.from_timing_points(timing_points)
        encode_notes(notes, timing)
    except ValueError:
        return EligibilityResult(False, FilterReason.INVALID_GRID)
    return EligibilityResult(True)
