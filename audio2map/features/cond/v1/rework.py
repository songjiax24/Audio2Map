"""ReworkSR for one mania 4K chart, no mods.

LeoBlack's card calls this number ReworkSR. The formula is Sunny's
Star-Rating-Rebirth (NM): J/X/P/A/R strain, then the weighted percentile
star. This port reads notes through ``osu.parser`` and does not import the
pattern analyser. Empty lanes stay at their column index so a missing lane
does not shift the others.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from audio2map.osu.parser import parse_beatmap
from audio2map.osu.schema import NoteType

_KEYS = 4
_CROSS = (
    0.175,
    0.25,
    0.05,
    0.25,
    0.175,
)


class ReworkSRError(ValueError):
    """This chart has no ReworkSR. The caller records a missing dimension."""


# Same OverallDifficulty that osu/export.py writes. The chart file's OD is ignored.
# Do not import the exporter; this constant is the cond copy of that default.
REWORK_COND_OD = 8.0


def _hit_leniency(overall_difficulty: float) -> float:
    inside = (64.5 - math.ceil(overall_difficulty * 3.0)) / 500.0
    value = 0.3 * inside**0.5
    return min(value, 0.6 * (value - 0.09) + 0.09)


def _note_columns(
    path: Path,
) -> tuple[list[list[tuple[int, int, int]]], list[tuple[int, int, int]]]:
    beatmap = parse_beatmap(path)
    if not beatmap.notes:
        raise ReworkSRError(f"no notes: {path}")
    notes: list[tuple[int, int, int]] = []
    for note in beatmap.notes:
        tail = note.end_time_ms if note.note_type == NoteType.HOLD else -1
        notes.append((note.col, note.time_ms, -1 if tail is None else tail))
    notes.sort(key=lambda item: (item[1], item[0]))
    columns: list[list[tuple[int, int, int]]] = [[] for _ in range(_KEYS)]
    for note in notes:
        columns[note[0]].append(note)
    return columns, notes


def _cumulative_sum(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    total = np.zeros(len(x))
    for i in range(1, len(x)):
        total[i] = total[i - 1] + values[i - 1] * (x[i] - x[i - 1])
    return total


def _query_cumsum(q: float, x: np.ndarray, total: np.ndarray, values: np.ndarray) -> float:
    if q <= x[0]:
        return 0.0
    if q >= x[-1]:
        return float(total[-1])
    index = int(np.searchsorted(x, q) - 1)
    return float(total[index] + values[index] * (q - x[index]))


def _smooth(
    x: np.ndarray,
    values: np.ndarray,
    window: float,
    *,
    scale: float,
    mode: str,
) -> np.ndarray:
    total = _cumulative_sum(x, values)
    out = np.empty_like(values)
    for i, center in enumerate(x):
        left = max(float(center - window), float(x[0]))
        right = min(float(center + window), float(x[-1]))
        span = _query_cumsum(right, x, total, values)
        span -= _query_cumsum(left, x, total, values)
        if mode == "avg":
            width = right - left
            out[i] = span / width if width > 0.0 else 0.0
        else:
            out[i] = scale * span
    return out


def _corners(
    end_time: int,
    notes: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base: set[int] = set()
    for _, head, tail in notes:
        base.add(head)
        if tail >= 0:
            base.add(tail)
    for mark in list(base):
        base.add(mark + 501)
        base.add(mark - 499)
        base.add(mark + 1)
    base.add(0)
    base.add(end_time)
    base_list = sorted(mark for mark in base if 0 <= mark <= end_time)

    wide: set[int] = set()
    for _, head, tail in notes:
        wide.add(head)
        if tail >= 0:
            wide.add(tail)
    for mark in list(wide):
        wide.add(mark + 1000)
        wide.add(mark - 1000)
    wide.add(0)
    wide.add(end_time)
    wide_list = sorted(mark for mark in wide if 0 <= mark <= end_time)
    all_marks = sorted(set(base_list) | set(wide_list))
    return (
        np.asarray(all_marks, dtype=float),
        np.asarray(base_list, dtype=float),
        np.asarray(wide_list, dtype=float),
    )


def _key_usage(
    notes: list[tuple[int, int, int]],
    base: np.ndarray,
    end_time: int,
) -> list[np.ndarray]:
    usage = [np.zeros(len(base), dtype=bool) for _ in range(_KEYS)]
    for col, head, tail in notes:
        start = max(head - 150, 0)
        stop = (head + 150) if tail < 0 else min(tail + 150, end_time - 1)
        left = int(np.searchsorted(base, start, side="left"))
        right = int(np.searchsorted(base, stop, side="left"))
        usage[col][left:right] = True
    return usage


def _key_usage_400(
    notes: list[tuple[int, int, int]],
    base: np.ndarray,
    end_time: int,
) -> list[np.ndarray]:
    usage = [np.zeros(len(base), dtype=float) for _ in range(_KEYS)]
    for col, head, tail in notes:
        start = max(head, 0)
        stop = head if tail < 0 else min(tail, end_time - 1)
        left_400 = int(np.searchsorted(base, start - 400, side="left"))
        left = int(np.searchsorted(base, start, side="left"))
        right = int(np.searchsorted(base, stop, side="left"))
        right_400 = int(np.searchsorted(base, stop + 400, side="left"))
        hold = min(stop - start, 1500) / 150.0
        usage[col][left:right] += 3.75 + hold
        before = base[left_400:left]
        usage[col][left_400:left] += 3.75 - 3.75 / 400.0**2 * (before - start) ** 2
        after = base[right:right_400]
        usage[col][right:right_400] += 3.75 - 3.75 / 400.0**2 * np.abs(after - stop) ** 2
    return usage


def _anchor(usage_400: list[np.ndarray]) -> np.ndarray:
    length = len(usage_400[0])
    anchor = np.zeros(length)
    for index in range(length):
        counts = np.array([usage_400[col][index] for col in range(_KEYS)])
        counts[::-1].sort()
        nonzero = counts[counts != 0]
        if nonzero.size > 1:
            ratio = nonzero[1:] / nonzero[:-1]
            walk = np.sum(nonzero[:-1] * (1.0 - 4.0 * (0.5 - ratio) ** 2))
            anchor[index] = walk / np.sum(nonzero[:-1])
    return 1.0 + np.minimum(anchor - 0.18, 5.0 * (anchor - 0.22) ** 3)


def _ln_representation(ln_notes: list[tuple[int, int, int]], end_time: int):
    delta: dict[int, float] = {}
    for _, head, tail in ln_notes:
        first = min(head + 60, tail)
        second = min(head + 120, tail)
        delta[first] = delta.get(first, 0.0) + 1.3
        delta[second] = delta.get(second, 0.0) + (-1.3 + 1.0)
        delta[tail] = delta.get(tail, 0.0) - 1.0
    points = sorted({0, end_time, *delta.keys()})
    values: list[float] = []
    running = 0.0
    for index in range(len(points) - 1):
        mark = points[index]
        if mark in delta:
            running += delta[mark]
        values.append(min(running, 2.5 + 0.5 * running))
    return points, values


def _ln_sum(left: float, right: float, points: list[int], values: list[float]) -> float:
    import bisect

    i = bisect.bisect_right(points, left) - 1
    j = bisect.bisect_right(points, right) - 1
    if i == j:
        return (right - left) * values[i]
    total = (points[i + 1] - left) * values[i]
    for mid in range(i + 1, j):
        total += (points[mid + 1] - points[mid]) * values[mid]
    total += (right - points[j]) * values[j]
    return total


def _jack_nerf(delta: float) -> float:
    return 1.0 - 7e-5 * (0.15 + abs(delta - 0.08)) ** (-4)


def _jbar(columns: list[list[tuple[int, int, int]]], base: np.ndarray, x: float):
    raw = [np.zeros(len(base)) for _ in range(_KEYS)]
    delta_k = [np.full(len(base), 1e9) for _ in range(_KEYS)]
    for col in range(_KEYS):
        notes = columns[col]
        for index in range(len(notes) - 1):
            start = notes[index][1]
            end = notes[index + 1][1]
            if end <= start:
                continue
            left = int(np.searchsorted(base, start, side="left"))
            right = int(np.searchsorted(base, end, side="left"))
            if right <= left:
                continue
            delta = 0.001 * (end - start)
            val = (delta ** (-1)) * (delta + 0.11 * x ** 0.25) ** (-1)
            raw[col][left:right] = val * _jack_nerf(delta)
            delta_k[col][left:right] = delta
    smoothed = [_smooth(base, raw[col], 500.0, scale=0.001, mode="sum") for col in range(_KEYS)]
    out = np.empty(len(base))
    for index in range(len(base)):
        numerator = 0.0
        denominator = 0.0
        for col in range(_KEYS):
            weight = 1.0 / delta_k[col][index]
            numerator += max(smoothed[col][index], 0.0) ** 5 * weight
            denominator += weight
        out[index] = (numerator / max(1e-9, denominator)) ** 0.2
    return delta_k, out


def _active_at(active: list[list[int]], index: int) -> list[int]:
    if not active:
        return []
    if index < 0:
        return []
    if index >= len(active):
        return active[-1]
    return active[index]


def _xbar(
    columns: list[list[tuple[int, int, int]]],
    active: list[list[int]],
    base: np.ndarray,
    x: float,
) -> np.ndarray:
    import heapq

    raw = [np.zeros(len(base)) for _ in range(_KEYS + 1)]
    fast = [np.zeros(len(base)) for _ in range(_KEYS + 1)]
    for slot in range(_KEYS + 1):
        if slot == 0:
            pair = columns[0]
        elif slot == _KEYS:
            pair = columns[_KEYS - 1]
        else:
            pair = list(
                heapq.merge(columns[slot - 1], columns[slot], key=lambda item: item[1])
            )
        for index in range(1, len(pair)):
            start = pair[index - 1][1]
            end = pair[index][1]
            left = int(np.searchsorted(base, start, side="left"))
            right = int(np.searchsorted(base, end, side="left"))
            if right <= left:
                continue
            delta = 0.001 * (end - start)
            if delta <= 0.0:
                continue
            value = 0.16 * max(x, delta) ** (-2)
            left_cols = _active_at(active, left)
            right_cols = _active_at(active, right)
            quiet_left = (slot - 1) not in left_cols and (slot - 1) not in right_cols
            quiet_right = slot not in left_cols and slot not in right_cols
            if quiet_left or quiet_right:
                value *= 1.0 - _CROSS[slot]
            raw[slot][left:right] = value
            fast[slot][left:right] = max(0.0, 0.4 * max(delta, 0.06, 0.75 * x) ** (-2) - 80.0)
    base_values = np.zeros(len(base))
    for index in range(len(base)):
        total = sum(raw[slot][index] * _CROSS[slot] for slot in range(_KEYS + 1))
        for slot in range(_KEYS):
            prod = fast[slot][index] * _CROSS[slot] * fast[slot + 1][index] * _CROSS[slot + 1]
            total += math.sqrt(prod)
        base_values[index] = total
    return _smooth(base, base_values, 500.0, scale=0.001, mode="sum")


def _stream_boost(delta: float) -> float:
    density = 7.5 / delta
    if 160.0 < density < 360.0:
        return 1.0 + 1.7e-7 * (density - 160.0) * (density - 360.0) ** 2
    return 1.0


def _pbar(
    notes: list[tuple[int, int, int]],
    ln_points: list[int],
    ln_values: list[float],
    anchor: np.ndarray,
    base: np.ndarray,
    x: float,
) -> np.ndarray:
    step = np.zeros(len(base))
    for index in range(len(notes) - 1):
        left_t = notes[index][1]
        right_t = notes[index + 1][1]
        gap = right_t - left_t
        if gap < 1e-9:
            spike = 1000.0 * (0.02 * (4.0 / x - 24.0)) ** 0.25
            left = int(np.searchsorted(base, left_t, side="left"))
            right = int(np.searchsorted(base, left_t, side="right"))
            if right > left:
                step[left:right] += spike
            continue
        left = int(np.searchsorted(base, left_t, side="left"))
        right = int(np.searchsorted(base, right_t, side="left"))
        if right <= left:
            continue
        delta = 0.001 * gap
        hold = 1.0 + 6.0 * 0.001 * _ln_sum(left_t, right_t, ln_points, ln_values)
        boost = _stream_boost(delta)
        if delta < 2.0 * x / 3.0:
            brace = 1.0 - 24.0 * x ** (-1) * (delta - x / 2.0) ** 2
        else:
            brace = 1.0 - 24.0 * x ** (-1) * (x / 6.0) ** 2
        inc = delta ** (-1) * (0.08 * x ** (-1) * brace) ** 0.25 * max(boost, hold)
        step[left:right] += np.minimum(inc * anchor[left:right], np.maximum(inc, inc * 2.0 - 10.0))
    return _smooth(base, step, 500.0, scale=0.001, mode="sum")


def _abar(
    active: list[list[int]],
    delta_k: list[np.ndarray],
    wide: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    gaps = [np.zeros(len(base)) for _ in range(_KEYS - 1)]
    for index, cols in enumerate(active):
        for pair in range(len(cols) - 1):
            left = cols[pair]
            right = cols[pair + 1]
            if left >= _KEYS - 1:
                continue
            largest = max(delta_k[left][index], delta_k[right][index])
            gaps[left][index] = abs(delta_k[left][index] - delta_k[right][index]) + 0.4 * max(
                0.0, largest - 0.11
            )
    step = np.ones(len(wide))
    for index, mark in enumerate(wide):
        slot = int(np.searchsorted(base, mark))
        if slot >= len(base):
            slot = len(base) - 1
        cols = active[slot]
        for pair in range(len(cols) - 1):
            left = cols[pair]
            right = cols[pair + 1]
            if left >= _KEYS - 1:
                continue
            gap = gaps[left][slot]
            largest = max(delta_k[left][slot], delta_k[right][slot])
            if gap < 0.02:
                step[index] *= min(0.75 + 0.5 * largest, 1.0)
            elif gap < 0.07:
                step[index] *= min(0.65 + 5.0 * gap + 0.5 * largest, 1.0)
    return _smooth(wide, step, 250.0, scale=1.0, mode="avg")


def _next_head(column: list[tuple[int, int, int]], head: int) -> int:
    import bisect

    times = [note[1] for note in column]
    index = bisect.bisect_left(times, head)
    if index + 1 < len(column):
        return column[index + 1][1]
    return 10**9


def _rbar(
    columns: list[list[tuple[int, int, int]]],
    tails: list[tuple[int, int, int]],
    base: np.ndarray,
    x: float,
) -> np.ndarray:
    release: list[float] = []
    for col, head, tail in tails:
        nxt = _next_head(columns[col], head)
        head_term = 0.001 * abs(tail - head - 80) / x
        tail_term = 0.001 * abs(nxt - tail - 80) / x
        release.append(
            2.0
            / (
                2.0
                + math.exp(-5.0 * (head_term - 0.75))
                + math.exp(-5.0 * (tail_term - 0.75))
            )
        )
    step = np.zeros(len(base))
    for index in range(len(tails) - 1):
        start = tails[index][2]
        end = tails[index + 1][2]
        left = int(np.searchsorted(base, start, side="left"))
        right = int(np.searchsorted(base, end, side="left"))
        if right <= left:
            continue
        delta = 0.001 * (end - start)
        if delta <= 0.0:
            continue
        both = release[index] + release[index + 1]
        step[left:right] = 0.08 * delta ** (-0.5) * x ** (-1) * (1.0 + 0.8 * both)
    return _smooth(base, step, 500.0, scale=0.001, mode="sum")


def _counts(notes: list[tuple[int, int, int]], usage: list[np.ndarray], base: np.ndarray):
    import bisect

    times = sorted(note[1] for note in notes)
    density = np.zeros(len(base))
    for index, mark in enumerate(base):
        hi = bisect.bisect_left(times, mark + 500)
        lo = bisect.bisect_left(times, mark - 500)
        density[index] = hi - lo
    used = np.array(
        [max(sum(1 for col in range(_KEYS) if usage[col][index]), 1) for index in range(len(base))],
        dtype=float,
    )
    return density, used


def _step(new_x: np.ndarray, old_x: np.ndarray, old: np.ndarray) -> np.ndarray:
    index = np.searchsorted(old_x, new_x, side="right") - 1
    index = np.clip(index, 0, len(old) - 1)
    return old[index]


def _rescale_high(star: float) -> float:
    if star <= 9.0:
        return star
    return 9.0 + (star - 9.0) / 1.2


def rework_star_rating(path: Path | str) -> float:
    """Return the NM Rework star for a mania 4K ``.osu`` file."""
    path = Path(path)
    x = _hit_leniency(REWORK_COND_OD)
    if not math.isfinite(x) or x <= 0.0:
        raise ReworkSRError(f"hit leniency is not positive: {path}")
    columns, notes = _note_columns(path)
    tails = [note[2] for note in notes]
    end_time = max(max(note[1] for note in notes), max(tails)) + 1
    if end_time < 2:
        raise ReworkSRError(f"chart is too short for ReworkSR: {path}")
    all_corners, base, wide = _corners(end_time, notes)
    if len(all_corners) < 2 or len(base) < 2:
        raise ReworkSRError(f"no strain grid: {path}")
    usage = _key_usage(notes, base, end_time)
    active = [[col for col in range(_KEYS) if usage[col][index]] for index in range(len(base))]
    usage_400 = _key_usage_400(notes, base, end_time)
    anchor = _anchor(usage_400)
    delta_k, jack = _jbar(columns, base, x)
    cross = _xbar(columns, active, base, x)
    ln_notes = [note for note in notes if note[2] >= 0]
    ln_points, ln_values = _ln_representation(ln_notes, end_time)
    press = _pbar(notes, ln_points, ln_values, anchor, base, x)
    accuracy = _abar(active, delta_k, wide, base)
    ordered_tails = sorted(ln_notes, key=lambda note: note[2])
    release = _rbar(columns, ordered_tails, base, x)
    density_step, keys_step = _counts(notes, usage, base)

    jack = np.interp(all_corners, base, jack)
    cross = np.interp(all_corners, base, cross)
    press = np.interp(all_corners, base, press)
    accuracy = np.interp(all_corners, wide, accuracy)
    release = np.interp(all_corners, base, release)
    density = _step(all_corners, base, density_step)
    keys = _step(all_corners, base, keys_step)

    strain = (
        0.4 * (accuracy ** (3.0 / keys) * np.minimum(jack, 8.0 + 0.85 * jack)) ** 1.5
        + 0.6 * (accuracy ** (2.0 / 3.0) * (0.8 * press + release * 35.0 / (density + 8.0))) ** 1.5
    ) ** (2.0 / 3.0)
    tech = (accuracy ** (3.0 / keys) * cross) / (cross + strain + 1.0)
    difficulty = 2.7 * np.sqrt(strain) * tech**1.5 + strain * 0.27

    gaps = np.empty_like(all_corners)
    gaps[0] = (all_corners[1] - all_corners[0]) / 2.0
    gaps[-1] = (all_corners[-1] - all_corners[-2]) / 2.0
    gaps[1:-1] = (all_corners[2:] - all_corners[:-2]) / 2.0
    weights = density * gaps
    order = np.argsort(difficulty, kind="quicksort")
    ordered = difficulty[order]
    ordered_w = weights[order]
    cumulative = np.cumsum(ordered_w)
    total_w = float(cumulative[-1])
    if total_w <= 0.0 or not math.isfinite(total_w):
        raise ReworkSRError(f"strain weights are empty: {path}")
    targets = np.array([0.945, 0.935, 0.925, 0.915, 0.845, 0.835, 0.825, 0.815])
    picked = np.searchsorted(cumulative / total_w, targets, side="left")
    picked = np.clip(picked, 0, len(ordered) - 1)
    high = float(np.mean(ordered[picked[:4]]))
    mid = float(np.mean(ordered[picked[4:8]]))
    mean_5 = float((np.sum(ordered**5 * ordered_w) / np.sum(ordered_w)) ** 0.2)
    star = 0.88 * high * 0.25 + 0.94 * mid * 0.2 + mean_5 * 0.55
    extra = 0.5 * sum(min(tail - head, 1000) / 200.0 for _, head, tail in ln_notes)
    count = len(notes) + extra
    star *= count / (count + 60.0)
    star = _rescale_high(star) * 0.975
    if not math.isfinite(star):
        raise ReworkSRError(f"non-finite ReworkSR: {path}")
    return float(star)
