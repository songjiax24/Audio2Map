"""Note-level chart evaluation on the canonical tick grid."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from audio2map.osu.row_tokens import CanonicalTiming, ms_to_tick
from audio2map.osu.schema import ManiaNote, NoteType


@dataclass(frozen=True, slots=True)
class NoteSig:
    head_tick: int
    col: int
    kind: str  # "tap" | "hold"
    end_tick: int | None = None

    @classmethod
    def from_note(cls, note: ManiaNote, timing: CanonicalTiming) -> NoteSig:
        head = ms_to_tick(note.time_ms, timing)
        if note.note_type == NoteType.TAP:
            return cls(head, note.col, "tap", None)
        assert note.end_time_ms is not None
        return cls(head, note.col, "hold", ms_to_tick(note.end_time_ms, timing))


def notes_to_sigs(notes: list[ManiaNote], timing: CanonicalTiming) -> set[NoteSig]:
    return {NoteSig.from_note(n, timing) for n in notes}


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
    ms_deltas: list[int] | None = None

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def tap_f1(self) -> float:
        return _f1(self.tap_tp, self.tap_fp, self.tap_fn)

    def hold_f1(self) -> float:
        return _f1(self.hold_tp, self.hold_fp, self.hold_fn)

    def ms_delta_summary(self) -> dict[str, float]:
        if not self.ms_deltas:
            return {"mean": 0.0, "p95": 0.0, "max": 0.0}
        xs = sorted(self.ms_deltas)
        p95_idx = min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))
        return {
            "mean": sum(xs) / len(xs),
            "p95": float(xs[p95_idx]),
            "max": float(xs[-1]),
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["precision"] = self.precision
        d["recall"] = self.recall
        d["f1"] = self.f1
        d["tap_f1"] = self.tap_f1()
        d["hold_f1"] = self.hold_f1()
        d["ms_delta"] = self.ms_delta_summary()
        d.pop("ms_deltas", None)
        return d


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def compare_note_lists(
    predicted: list[ManiaNote],
    expected: list[ManiaNote],
    timing: CanonicalTiming,
    *,
    collect_ms_deltas: bool = True,
) -> NoteMatchStats:
    pred_sigs = notes_to_sigs(predicted, timing)
    exp_sigs = notes_to_sigs(expected, timing)
    tp_sigs = pred_sigs & exp_sigs

    stats = NoteMatchStats(
        tp=len(tp_sigs),
        fp=len(pred_sigs - exp_sigs),
        fn=len(exp_sigs - pred_sigs),
    )
    for sig in tp_sigs:
        if sig.kind == "tap":
            stats.tap_tp += 1
        else:
            stats.hold_tp += 1
    for sig in pred_sigs - exp_sigs:
        if sig.kind == "tap":
            stats.tap_fp += 1
        else:
            stats.hold_fp += 1
    for sig in exp_sigs - pred_sigs:
        if sig.kind == "tap":
            stats.tap_fn += 1
        else:
            stats.hold_fn += 1

    if collect_ms_deltas:
        exp_by_sig = {NoteSig.from_note(n, timing): n for n in expected}
        pred_by_sig = {NoteSig.from_note(n, timing): n for n in predicted}
        deltas: list[int] = []
        for sig in tp_sigs:
            pe, pp = exp_by_sig[sig], pred_by_sig[sig]
            deltas.append(abs(pe.time_ms - pp.time_ms))
            if sig.kind == "hold" and pe.end_time_ms is not None and pp.end_time_ms is not None:
                deltas.append(abs(pe.end_time_ms - pp.end_time_ms))
        stats.ms_deltas = deltas
    return stats


def compare_note_lists_tick_tol(
    predicted: list[ManiaNote],
    expected: list[ManiaNote],
    timing: CanonicalTiming,
    *,
    tick_tolerance: int = 1,
) -> NoteMatchStats:
    """Greedy match allowing ``±tick_tolerance`` on heads (diagnostic)."""
    exp = list(expected)
    used = [False] * len(exp)
    stats = NoteMatchStats()

    for pn in predicted:
        ps = NoteSig.from_note(pn, timing)
        best_j = None
        best_dist = tick_tolerance + 1
        for j, en in enumerate(exp):
            if used[j]:
                continue
            es = NoteSig.from_note(en, timing)
            if ps.col != es.col or ps.kind != es.kind:
                continue
            dist = abs(ps.head_tick - es.head_tick)
            if ps.kind == "hold":
                if ps.end_tick is None or es.end_tick is None:
                    continue
                if abs(ps.end_tick - es.end_tick) > tick_tolerance:
                    continue
            if dist <= tick_tolerance and dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is None:
            stats.fp += 1
            if ps.kind == "tap":
                stats.tap_fp += 1
            else:
                stats.hold_fp += 1
            continue
        used[best_j] = True
        stats.tp += 1
        if ps.kind == "tap":
            stats.tap_tp += 1
        else:
            stats.hold_tp += 1

    for j, en in enumerate(exp):
        if used[j]:
            continue
        stats.fn += 1
        es = NoteSig.from_note(en, timing)
        if es.kind == "tap":
            stats.tap_fn += 1
        else:
            stats.hold_fn += 1
    return stats
