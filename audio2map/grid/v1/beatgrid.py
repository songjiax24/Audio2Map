"""v1 BeatGrid. Float64 beat starts, indexed by ``(section, k)``.

Coincidence uses the nearest beat index and ``EPS``. A non-coincident tail is a
residual beat and is kept at any length. Grid points on that beat at or after
``t_red - EPS`` are dropped. ``k* == 0`` emits no beats for the earlier section.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from audio2map.grid.v1.positions import v80_cells
from audio2map.utils.errors import (
    BAD_AUDIO_DURATION,
    BEAT_INDEX_NOT_STARTED,
    BEAT_LENGTH_BELOW_MINIMUM,
    GRID_SHORT_OF_AUDIO,
    NO_BEAT_BEFORE_ZERO,
    NO_REDLINE,
    NONPOSITIVE_BEAT_LENGTH,
    NON_FINITE_OFFSET,
    RESIDUAL_WITHOUT_REDLINE,
    TOO_MANY_BEATS,
    ChartRejected,
)
from audio2map.osu.v1.timing import Redline
from audio2map.utils.tolerance import EPS

# Below this, a 1 ms label window covers a huge run of beats.
_MIN_BEAT_MS = 1e-4
_MAX_BEATS = 2_000_000


@dataclass(frozen=True, slots=True)
class Beat:
    section: int
    beat_k: int
    global_b: int
    start_ms: float
    section_offset_ms: float
    actual_length_ms: float
    nominal_length_ms: float
    is_residual: bool
    t_red_ms: float | None


@dataclass(frozen=True, slots=True)
class GridPoint:
    beat_id: int
    section: int
    beat_k: int
    global_b: int
    numer: int
    denom: int
    cell: int
    time_ms: float


@dataclass(frozen=True, slots=True)
class _SectionPlan:
    section: int
    offset_ms: float
    beat_length_ms: float
    # Exclusive end of k, or None for the last section (runs to T_audio).
    k_to: int | None
    residual_k: int | None
    t_red_ms: float | None

    @property
    def empty(self) -> bool:
        return self.k_to == 0


@dataclass(frozen=True, slots=True)
class BeatGrid:
    """Beats whose start lies in ``[coverage_start_ms, coverage_end_ms)``.

    ``near_duplicate_pairs`` is derived from consecutive red lines. It is not
    a caller-supplied annotation.
    """

    beats: tuple[Beat, ...]
    points: tuple[tuple[GridPoint, ...], ...]
    coverage_start_ms: float
    coverage_end_ms: float
    t_audio_ms: float
    cells: tuple[tuple[int, int], ...]
    near_duplicate_pairs: tuple[tuple[float, float], ...]

    def beat_has_legal_point(self, beat_id: int) -> bool:
        return len(self.points[beat_id]) > 0


def _near_duplicate_pairs(redlines: tuple[Redline, ...]) -> tuple[tuple[float, float], ...]:
    """Consecutive red lines with ``0 < later - earlier < EPS``, in list order."""
    pairs: list[tuple[float, float]] = []
    for prev, nxt in zip(redlines, redlines[1:]):
        delta = nxt.offset_ms - prev.offset_ms
        if 0.0 < delta < EPS:
            pairs.append((prev.offset_ms, nxt.offset_ms))
    return tuple(pairs)


def _plan_sections(redlines: tuple[Redline, ...]) -> list[_SectionPlan]:
    plans: list[_SectionPlan] = []
    last = len(redlines) - 1
    for index, redline in enumerate(redlines):
        start = redline.offset_ms
        length = redline.beat_length_ms
        if index == last:
            plans.append(
                _SectionPlan(index, start, length, None, None, None)
            )
            continue
        t_red = redlines[index + 1].offset_ms
        delta = t_red - start
        k_star = int(round(delta / length))
        nearest = start + k_star * length
        if abs(t_red - nearest) <= EPS:
            plans.append(
                _SectionPlan(index, start, length, max(k_star, 0), None, t_red)
            )
            continue
        n_full = math.floor(delta / length)
        if n_full < 0:
            n_full = 0
        plans.append(_SectionPlan(index, start, length, n_full + 1, n_full, t_red))
    return plans


def build_beatgrid(
    redlines: tuple[Redline, ...] | list[Redline],
    t_audio_ms: float,
) -> BeatGrid:
    """Build the note-bearing beats and their legal V80 points."""
    if not math.isfinite(t_audio_ms) or t_audio_ms < 0.0:
        raise ChartRejected(BAD_AUDIO_DURATION, f"T_audio={t_audio_ms}")
    redline_list = tuple(redlines)
    if not redline_list:
        raise ChartRejected(NO_REDLINE, "empty redline list")
    for redline in redline_list:
        length = redline.beat_length_ms
        if not math.isfinite(redline.offset_ms):
            raise ChartRejected(NON_FINITE_OFFSET, f"offset={redline.offset_ms}")
        if not math.isfinite(length) or length <= 0.0:
            raise ChartRejected(
                NONPOSITIVE_BEAT_LENGTH,
                f"offset={redline.offset_ms} beatLength={length}",
            )
        if length < _MIN_BEAT_MS:
            raise ChartRejected(
                BEAT_LENGTH_BELOW_MINIMUM,
                f"offset={redline.offset_ms} beatLength={length}",
            )
    plans = _plan_sections(redline_list)
    first = next(plan.section for plan in plans if not plan.empty)
    return _build_from_plans(
        plans,
        first=first,
        t_audio_ms=t_audio_ms,
        cells=v80_cells(),
        near_duplicate_pairs=_near_duplicate_pairs(redline_list),
    )


def _build_from_plans(
    plans: list[_SectionPlan],
    *,
    first: int,
    t_audio_ms: float,
    cells: tuple[tuple[int, int], ...],
    near_duplicate_pairs: tuple[tuple[float, float], ...],
) -> BeatGrid:
    emitted: list[Beat] = []
    next_b: int | None = None
    g_end: float | None = None
    for plan in plans:
        if g_end is not None or plan.empty:
            continue
        k_values: list[int] = []
        if plan.section == first and plan.offset_ms > 0.0:
            k_lo = math.floor((0.0 - plan.offset_ms) / plan.beat_length_ms)
            k_values.extend(range(k_lo, 0))
        if plan.k_to is None:
            k_forward = 0
            while True:
                start = plan.offset_ms + k_forward * plan.beat_length_ms
                if start >= t_audio_ms:
                    g_end = start
                    break
                k_values.append(k_forward)
                k_forward += 1
                if len(emitted) + len(k_values) > _MAX_BEATS:
                    raise ChartRejected(TOO_MANY_BEATS, "too many beats")
        else:
            k_values.extend(range(0, plan.k_to))
        for k in k_values:
            start = plan.offset_ms + k * plan.beat_length_ms
            if start >= t_audio_ms:
                g_end = start
                break
            if plan.section == first:
                b = k
                next_b = k + 1
            else:
                if next_b is None:
                    raise ChartRejected(BEAT_INDEX_NOT_STARTED, "beat index was not started")
                b = next_b
                next_b += 1
            residual = plan.residual_k is not None and k == plan.residual_k
            if residual:
                if plan.t_red_ms is None:
                    raise ChartRejected(RESIDUAL_WITHOUT_REDLINE, "residual without redline")
                actual = plan.t_red_ms - start
                t_red = plan.t_red_ms
            else:
                actual = plan.beat_length_ms
                t_red = None
            emitted.append(
                Beat(
                    section=plan.section,
                    beat_k=k,
                    global_b=b,
                    start_ms=start,
                    section_offset_ms=plan.offset_ms,
                    actual_length_ms=actual,
                    nominal_length_ms=plan.beat_length_ms,
                    is_residual=residual,
                    t_red_ms=t_red,
                )
            )
        if g_end is not None:
            break
    if g_end is None:
        raise ChartRejected(GRID_SHORT_OF_AUDIO, "grid never reached T_audio")

    at_or_before_zero = [beat.start_ms for beat in emitted if beat.start_ms <= 0.0]
    if at_or_before_zero:
        coverage_start = max(at_or_before_zero)
    elif g_end == 0.0:
        coverage_start = 0.0
    else:
        raise ChartRejected(NO_BEAT_BEFORE_ZERO, "no beat at or before 0")

    kept = tuple(beat for beat in emitted if coverage_start <= beat.start_ms < g_end)
    return BeatGrid(
        beats=kept,
        points=_index_points(kept, cells, t_audio_ms),
        coverage_start_ms=coverage_start,
        coverage_end_ms=g_end,
        t_audio_ms=t_audio_ms,
        cells=cells,
        near_duplicate_pairs=near_duplicate_pairs,
    )


def _index_points(
    beats: tuple[Beat, ...],
    cells: tuple[tuple[int, int], ...],
    t_audio_ms: float,
) -> tuple[tuple[GridPoint, ...], ...]:
    per_beat: list[tuple[GridPoint, ...]] = []
    for beat_id, beat in enumerate(beats):
        legal: list[GridPoint] = []
        for cell, (numer, denom) in enumerate(cells):
            time_ms = beat.start_ms + (numer / denom) * beat.nominal_length_ms
            if time_ms < 0.0 or time_ms >= t_audio_ms:
                continue
            if beat.is_residual and beat.t_red_ms is not None:
                if time_ms >= beat.t_red_ms - EPS:
                    continue
            legal.append(
                GridPoint(
                    beat_id=beat_id,
                    section=beat.section,
                    beat_k=beat.beat_k,
                    global_b=beat.global_b,
                    numer=numer,
                    denom=denom,
                    cell=cell,
                    time_ms=time_ms,
                )
            )
        per_beat.append(tuple(legal))
    return tuple(per_beat)
