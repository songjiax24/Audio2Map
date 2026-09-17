"""Note-level chart evaluation. Identity is the tick lattice, not milliseconds."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from audio2map.grid import CanonicalTiming, TickNote
from audio2map.osu.schema import ManiaNote, NoteType


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
