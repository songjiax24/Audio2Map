"""Chart statistics over atomic tick, note, row, and hold counts.

``all`` is a marginal of those atomic counts. It is not stored as its own
category, and it is not the multiset union used by matching.
Pooled frequency divides by the statistic's whole observation universe.
"""

from __future__ import annotations

import math
from collections import Counter

from audio2map.grid import TICKS_PER_BEAT, CanonicalTiming, TickNote
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.tokens import EVENT_LANE_STATES, LaneState, RowState, chart_event_rows

_EVENT_NAME = {
    LaneState.TAP: "tap",
    LaneState.HOLD_START: "hold_head",
    LaneState.HOLD_END: "hold_tail",
}
_EVENT_STATES = ("tap", "hold_head", "hold_tail")
_NOTE_TYPES = ("tap", "hold")
_SNAP_NAMES = ("1/1", "1/2", "1/3", "1/4", "1/6", "1/8", "1/12", "1/16", "1/24", "1/48")


def snap_name(tick_mod: int) -> str:
    denom = TICKS_PER_BEAT // math.gcd(TICKS_PER_BEAT, tick_mod)
    return f"1/{denom}"


def _lane_name(col: int) -> str:
    return f"lane_{col + 1}"


class ChartStatAccumulator:
    """Sum atomic counts across charts. ``charts`` is the avgCount denominator."""

    def __init__(self) -> None:
        self.charts = 0
        self.tick: Counter[tuple[int, int, str]] = Counter()
        self.note: Counter[tuple[int, str]] = Counter()
        self.hold: Counter[tuple[int, int]] = Counter()
        self.row: Counter[RowState] = Counter()

    def add_chart(self, notes: list[ManiaNote], timing: CanonicalTiming) -> None:
        self.charts += 1
        rows = chart_event_rows(notes, timing)
        for tick, row in rows.items():
            q = tick % TICKS_PER_BEAT
            self.row[row] += 1
            for col, state in enumerate(row):
                if state in EVENT_LANE_STATES:
                    self.tick[(col, q, _EVENT_NAME[state])] += 1
        for note in TickNote.from_notes(notes, timing):
            kind = "tap" if note.note_type == NoteType.TAP else "hold"
            self.note[(note.col, kind)] += 1
            if note.note_type == NoteType.HOLD:
                assert note.end_tick is not None
                self.hold[(note.col, note.end_tick - note.head_tick)] += 1

    def to_dict(self) -> dict:
        n = self.charts
        tick_den = sum(self.tick.values())
        note_den = sum(self.note.values())
        hold_den = sum(self.hold.values())
        row_den = sum(self.row.values())
        return {
            "charts": n,
            "tick": _tick_report(self.tick, n, tick_den),
            "snap": _snap_report(self.tick, n, tick_den),
            "note": _note_report(self.note, n, note_den),
            "row": _row_report(self.row, n, row_den),
            "hold": _hold_report(self.hold, n, hold_den),
        }


def _avg(count: int, charts: int) -> float:
    return count / charts if charts else 0.0


def _freq(count: int, denominator: int) -> float:
    return count / denominator if denominator else 0.0


def _tick_report(counts: Counter[tuple[int, int, str]], charts: int, denominator: int) -> dict:
    return {
        "avg_count": _lane_state_grid(counts, charts, _avg, denominator=None),
        "pooled_frequency": _lane_state_grid(counts, charts, _freq, denominator=denominator),
    }


def _lane_state_grid(counts, charts: int, reduce, *, denominator: int | None) -> dict:
    qs = sorted({q for _col, q, _state in counts})
    out: dict[str, dict[str, dict[str, float]]] = {}
    scopes = [(col, _lane_name(col)) for col in range(4)]
    for col, name in scopes:
        out[name] = _state_bins(counts, qs, charts, reduce, denominator, col=col)
    out["all_lanes"] = _state_bins(counts, qs, charts, reduce, denominator, col=None)
    return out


def _state_bins(counts, qs, charts, reduce, denominator, *, col: int | None) -> dict:
    bins: dict[str, dict[str, float]] = {}
    for state in _EVENT_STATES:
        bins[state] = {}
        for q in qs:
            count = _tick_count(counts, q, state, col)
            bins[state][str(q)] = _reduced(count, charts, reduce, denominator)
    bins["all_event"] = {}
    for q in qs:
        count = sum(_tick_count(counts, q, state, col) for state in _EVENT_STATES)
        bins["all_event"][str(q)] = _reduced(count, charts, reduce, denominator)
    return bins


def _tick_count(counts, q: int, state: str, col: int | None) -> int:
    if col is None:
        return sum(counts[(lane, q, state)] for lane in range(4))
    return counts[(col, q, state)]


def _reduced(count: int, charts: int, reduce, denominator: int | None) -> float:
    if reduce is _avg:
        return _avg(count, charts)
    assert denominator is not None
    return _freq(count, denominator)


def _snap_report(counts: Counter[tuple[int, int, str]], charts: int, denominator: int) -> dict:
    """Same events and denominator as tick. ``q`` is replaced by its snap class."""
    snapped: Counter[tuple[int, str, str]] = Counter()
    for (col, q, state), count in counts.items():
        snapped[(col, snap_name(q), state)] += count
    names = [name for name in _SNAP_NAMES if any(key[1] == name for key in snapped)]
    return {
        "avg_count": _snap_grid(snapped, names, charts, _avg, denominator=None),
        "pooled_frequency": _snap_grid(snapped, names, charts, _freq, denominator=denominator),
    }


def _snap_grid(counts, names, charts, reduce, *, denominator: int | None) -> dict:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for col in range(4):
        out[_lane_name(col)] = _snap_states(counts, names, charts, reduce, denominator, col=col)
    out["all_lanes"] = _snap_states(counts, names, charts, reduce, denominator, col=None)
    return out


def _snap_states(counts, names, charts, reduce, denominator, *, col: int | None) -> dict:
    bins: dict[str, dict[str, float]] = {}
    for state in (*_EVENT_STATES, "all_event"):
        bins[state] = {}
        for name in names:
            if state == "all_event":
                count = sum(_snap_count(counts, name, atomic, col) for atomic in _EVENT_STATES)
            else:
                count = _snap_count(counts, name, state, col)
            bins[state][name] = _reduced(count, charts, reduce, denominator)
    return bins


def _snap_count(counts, name: str, state: str, col: int | None) -> int:
    if col is None:
        return sum(counts[(lane, name, state)] for lane in range(4))
    return counts[(col, name, state)]


def _note_report(counts: Counter[tuple[int, str]], charts: int, denominator: int) -> dict:
    def block(reduce, den: int | None) -> dict:
        out: dict[str, dict[str, float]] = {}
        for col in range(4):
            out[_lane_name(col)] = _note_types(counts, charts, reduce, den, col=col)
        out["all_lanes"] = _note_types(counts, charts, reduce, den, col=None)
        return out

    return {
        "avg_count": block(_avg, None),
        "pooled_frequency": block(_freq, denominator),
    }


def _note_types(counts, charts, reduce, denominator, *, col: int | None) -> dict:
    values = {}
    for kind in _NOTE_TYPES:
        count = _note_count(counts, kind, col)
        values[kind] = _reduced(count, charts, reduce, denominator)
    total = sum(_note_count(counts, kind, col) for kind in _NOTE_TYPES)
    values["all_notes"] = _reduced(total, charts, reduce, denominator)
    return values


def _note_count(counts, kind: str, col: int | None) -> int:
    if col is None:
        return sum(counts[(lane, kind)] for lane in range(4))
    return counts[(col, kind)]


def _row_report(counts: Counter[RowState], charts: int, denominator: int) -> dict:
    avg = {}
    freq = {}
    for row, count in sorted(counts.items(), key=lambda item: tuple(int(s) for s in item[0])):
        key = "".join(str(int(state)) for state in row)
        avg[key] = _avg(count, charts)
        freq[key] = _freq(count, denominator)
    return {"avg_count": avg, "pooled_frequency": freq}


def _hold_report(counts: Counter[tuple[int, int]], charts: int, denominator: int) -> dict:
    durations = sorted({d for _col, d in counts})

    def block(reduce, den: int | None) -> dict:
        out: dict[str, dict[str, float]] = {}
        for col in range(4):
            out[_lane_name(col)] = {
                str(d): _reduced(counts[(col, d)], charts, reduce, den) for d in durations
            }
        out["all_lanes"] = {
            str(d): _reduced(sum(counts[(lane, d)] for lane in range(4)), charts, reduce, den)
            for d in durations
        }
        return out

    return {
        "avg_count": block(_avg, None),
        "pooled_frequency": block(_freq, denominator),
    }
