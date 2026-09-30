"""Note and event matching. Identity is the tick lattice, not milliseconds.

``all`` here is the union of atomic identities in one multiset. It is not the
marginal used by chart statistics, and the two do not share a counter.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from audio2map.grid import CanonicalTiming, TickNote
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.tokens import EVENT_LANE_STATES, LaneState, chart_event_rows


@dataclass(slots=True)
class NoteMatchStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tap_tp: int = 0
    tap_fp: int = 0
    tap_fn: int = 0
    hold_tp: int = 0
    hold_fp: int = 0
    hold_fn: int = 0

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return _f1(self.tp, self.fp, self.fn)

    @property
    def tap_f1(self) -> float:
        return _f1(self.tap_tp, self.tap_fp, self.tap_fn)

    @property
    def hold_f1(self) -> float:
        return _f1(self.hold_tp, self.hold_fp, self.hold_fn)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["precision"] = self.precision
        d["recall"] = self.recall
        d["f1"] = self.f1
        d["tap_f1"] = self.tap_f1
        d["hold_f1"] = self.hold_f1
        return d


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    p = _ratio(tp, tp + fp)
    r = _ratio(tp, tp + fn)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def compare_note_lists(
    predicted: list[ManiaNote],
    expected: list[ManiaNote],
    timing: CanonicalTiming,
) -> NoteMatchStats:
    pred_c = Counter(TickNote.from_note(n, timing) for n in predicted)
    exp_c = Counter(TickNote.from_note(n, timing) for n in expected)
    stats = NoteMatchStats()
    for note in set(pred_c) | set(exp_c):
        p, e = pred_c[note], exp_c[note]
        matched = min(p, e)
        extra = p - matched
        missing = e - matched
        stats.tp += matched
        stats.fp += extra
        stats.fn += missing
        if note.note_type == NoteType.TAP:
            stats.tap_tp += matched
            stats.tap_fp += extra
            stats.tap_fn += missing
        else:
            stats.hold_tp += matched
            stats.hold_fp += extra
            stats.hold_fn += missing
    return stats


_EVENT_NAME = {
    LaneState.TAP: "tap",
    LaneState.HOLD_START: "hold_head",
    LaneState.HOLD_END: "hold_tail",
}


def _rates(pred: Counter, exp: Counter) -> dict[str, float | int]:
    tp = fp = fn = 0
    for key in set(pred) | set(exp):
        p, e = pred[key], exp[key]
        matched = min(p, e)
        tp += matched
        fp += p - matched
        fn += e - matched
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(tp, fp, fn),
    }


def _events(notes: list[ManiaNote], timing: CanonicalTiming) -> list[tuple[int, int, str]]:
    rows = chart_event_rows(notes, timing)
    found: list[tuple[int, int, str]] = []
    for tick, row in rows.items():
        for col, state in enumerate(row):
            if state in EVENT_LANE_STATES:
                found.append((tick, col, _EVENT_NAME[state]))
    return found


def _subset(counter: Counter, predicate) -> Counter:
    return Counter({key: count for key, count in counter.items() if predicate(key)})


def reference_match(
    predicted: list[ManiaNote],
    expected: list[ManiaNote],
    timing: CanonicalTiming,
) -> dict:
    """Lane-aware and lane-agnostic event/note P/R/F1.

    Lane-aware keeps column. Lane-agnostic drops column and matches one
    multiset of every lane. ``all_event`` / ``all_notes`` match the union of
    the atomic identities. They are not extra type labels inside the multiset.
    """
    events = _events(predicted, timing)
    expected_events = _events(expected, timing)
    aware_event = Counter(events)
    agnostic_event = Counter((tick, state) for tick, _col, state in events)
    exp_aware_event = Counter(expected_events)
    exp_agnostic_event = Counter((tick, state) for tick, _col, state in expected_events)

    def event_block(pred: Counter, exp: Counter, key_state) -> dict:
        return {
            "tap": _rates(_subset(pred, lambda k: key_state(k) == "tap"), _subset(exp, lambda k: key_state(k) == "tap")),
            "hold_head": _rates(
                _subset(pred, lambda k: key_state(k) == "hold_head"),
                _subset(exp, lambda k: key_state(k) == "hold_head"),
            ),
            "hold_tail": _rates(
                _subset(pred, lambda k: key_state(k) == "hold_tail"),
                _subset(exp, lambda k: key_state(k) == "hold_tail"),
            ),
            "all_event": _rates(pred, exp),
        }

    pred_notes = [TickNote.from_note(n, timing) for n in predicted]
    exp_notes = [TickNote.from_note(n, timing) for n in expected]
    aware_note = Counter(pred_notes)
    exp_aware_note = Counter(exp_notes)
    agnostic_note = Counter((n.head_tick, n.note_type, n.end_tick) for n in pred_notes)
    exp_agnostic_note = Counter((n.head_tick, n.note_type, n.end_tick) for n in exp_notes)

    def note_block(pred: Counter, exp: Counter, key_type) -> dict:
        return {
            "tap": _rates(
                _subset(pred, lambda k: key_type(k) == NoteType.TAP),
                _subset(exp, lambda k: key_type(k) == NoteType.TAP),
            ),
            "hold": _rates(
                _subset(pred, lambda k: key_type(k) == NoteType.HOLD),
                _subset(exp, lambda k: key_type(k) == NoteType.HOLD),
            ),
            "all_notes": _rates(pred, exp),
        }

    return {
        "event": {
            "lane_aware": event_block(aware_event, exp_aware_event, lambda k: k[2]),
            "lane_agnostic": event_block(agnostic_event, exp_agnostic_event, lambda k: k[1]),
        },
        "note": {
            "lane_aware": note_block(aware_note, exp_aware_note, lambda k: k.note_type),
            "lane_agnostic": note_block(agnostic_note, exp_agnostic_note, lambda k: k[1]),
        },
    }


def pairwise_note_f1(
    charts: list[list[ManiaNote]],
    timing: CanonicalTiming,
) -> dict:
    """Pairwise lane-aware and lane-agnostic all-notes F1. No similarity threshold.

    Official diversity uses ``seed_pair_report``, which keeps the full reference
    match rather than these two F1 values.
    """
    pairs = []
    for i in range(len(charts)):
        for j in range(i + 1, len(charts)):
            matched = reference_match(charts[i], charts[j], timing)
            pairs.append(
                {
                    "i": i,
                    "j": j,
                    "lane_aware_note_f1": matched["note"]["lane_aware"]["all_notes"]["f1"],
                    "lane_agnostic_note_f1": matched["note"]["lane_agnostic"]["all_notes"]["f1"],
                }
            )
    return {"pairs": pairs}
