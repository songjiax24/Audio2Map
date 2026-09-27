"""Lane, hand, and row motifs.

A position exists only when that motif's own scope contains a tap, hold head,
or hold tail. Relative time is the tick gap between those positions.
Pooled lane/hand figures are sums of occurrence counts, not frequencies.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from audio2map.grid import CanonicalTiming
from audio2map.osu.schema import ManiaNote
from audio2map.tokens import EVENT_LANE_STATES, LaneState, RowState, chart_event_rows

HAND_12 = (0, 1)
HAND_34 = (2, 3)
MotifKey = tuple


def mirror_hand_symbol(pair: tuple[LaneState, LaneState]) -> tuple[LaneState, LaneState]:
    """Map hand 34 onto hand 12 across the 4K center: ``0↔3``, ``1↔2``.

    A left-to-right pair ``(s2, s3)`` becomes ``(s3, s2)``.
    """
    return (pair[1], pair[0])


def _lane_positions(rows: dict[int, RowState], col: int) -> list[tuple[int, LaneState]]:
    return [
        (tick, row[col])
        for tick, row in sorted(rows.items())
        if row[col] in EVENT_LANE_STATES
    ]


def _hand_positions(
    rows: dict[int, RowState],
    cols: tuple[int, int],
) -> list[tuple[int, tuple[LaneState, LaneState]]]:
    left, right = cols
    found = []
    for tick, row in sorted(rows.items()):
        if row[left] in EVENT_LANE_STATES or row[right] in EVENT_LANE_STATES:
            found.append((tick, (row[left], row[right])))
    return found


def _format_symbol(symbol: LaneState | tuple) -> str:
    if isinstance(symbol, LaneState):
        return symbol.name
    if len(symbol) == 2:
        return f"{symbol[0].name}+{symbol[1].name}"
    return "".join(str(int(state)) for state in symbol)


def motif_key(symbols: Sequence, ticks: Sequence[int], start: int, length: int) -> MotifKey:
    parts: list = [symbols[start]]
    for offset in range(1, length):
        parts.append(ticks[start + offset] - ticks[start + offset - 1])
        parts.append(symbols[start + offset])
    return tuple(parts)


def motif_length(key: MotifKey) -> int:
    return (len(key) + 1) // 2


def format_motif(key: MotifKey) -> str:
    parts = []
    for index, part in enumerate(key):
        parts.append(str(part) if index % 2 == 1 else _format_symbol(part))
    return "/".join(parts)


def parse_motif(text: str) -> MotifKey:
    parts = text.split("/")
    parsed: list = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            parsed.append(int(part))
            continue
        if "+" in part:
            left, right = part.split("+", 1)
            parsed.append((LaneState[left], LaneState[right]))
        elif part.isdigit() and len(part) == 4:
            parsed.append(tuple(LaneState(int(ch)) for ch in part))
        else:
            parsed.append(LaneState[part])
    return tuple(parsed)


class MotifSequences:
    def __init__(self, rows: dict[int, RowState]) -> None:
        self.lane = [_lane_positions(rows, col) for col in range(4)]
        self.hand = {
            "12": _hand_positions(rows, HAND_12),
            "34": _hand_positions(rows, HAND_34),
        }
        self.row = [(tick, row) for tick, row in sorted(rows.items())]

    @classmethod
    def from_notes(cls, notes: list[ManiaNote], timing: CanonicalTiming) -> MotifSequences:
        return cls(chart_event_rows(notes, timing))


def _length_profile(n_positions: int) -> dict[str, int]:
    return {str(length): n_positions - length + 1 for length in range(1, n_positions + 1)}


def _add_profiles(profiles: Iterable[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for profile in profiles:
        for length, count in profile.items():
            total[length] = total.get(length, 0) + count
    return total


def length_occurrence_profile(seqs: MotifSequences) -> dict:
    """How many overlapping spans exist at each length. No motif identity yet."""
    lane = [_length_profile(len(positions)) for positions in seqs.lane]
    hand = {name: _length_profile(len(positions)) for name, positions in seqs.hand.items()}
    return {
        "lane": {f"lane_{col + 1}": lane[col] for col in range(4)},
        "all_lanes": _add_profiles(lane),
        "hand": hand,
        "both_hands": _add_profiles(hand.values()),
        "row": _length_profile(len(seqs.row)),
    }


def _count_spans(
    positions: list[tuple[int, object]],
    vocab: set[MotifKey],
    *,
    mirror: bool = False,
) -> Counter[MotifKey]:
    if mirror:
        positions = [(tick, mirror_hand_symbol(symbol)) for tick, symbol in positions]  # type: ignore[arg-type]
    symbols = [symbol for _tick, symbol in positions]
    ticks = [tick for tick, _symbol in positions]
    lengths = {motif_length(key) for key in vocab}
    counts: Counter[MotifKey] = Counter()
    n = len(symbols)
    for length in lengths:
        if length < 1 or n < length:
            continue
        for start in range(n - length + 1):
            key = motif_key(symbols, ticks, start, length)
            if key in vocab:
                counts[key] += 1
    return counts


class MotifAccumulator:
    """Per-scope occurrence counts. Pooled values are sums, not frequencies."""

    def __init__(self, vocab: dict[str, set[MotifKey]] | None = None) -> None:
        self.charts = 0
        self.vocab = vocab or {"lane": set(), "hand": set(), "row": set()}
        self.lane: list[Counter[MotifKey]] = [Counter() for _ in range(4)]
        self.hand = {"12": Counter(), "34": Counter()}
        self.row: Counter[MotifKey] = Counter()
        self._length = {
            "lane": [Counter() for _ in range(4)],
            "hand": {"12": Counter(), "34": Counter()},
            "row": Counter(),
        }

    def add_chart(self, notes: list[ManiaNote], timing: CanonicalTiming) -> None:
        self.charts += 1
        seqs = MotifSequences.from_notes(notes, timing)
        for col, positions in enumerate(seqs.lane):
            self.lane[col].update(_count_spans(positions, self.vocab.get("lane", set())))
            for length, count in _length_profile(len(positions)).items():
                self._length["lane"][col][length] += count
        self.hand["12"].update(_count_spans(seqs.hand["12"], self.vocab.get("hand", set())))
        self.hand["34"].update(
            _count_spans(seqs.hand["34"], self.vocab.get("hand", set()), mirror=True)
        )
        for name in ("12", "34"):
            for length, count in _length_profile(len(seqs.hand[name])).items():
                self._length["hand"][name][length] += count
        self.row.update(_count_spans(seqs.row, self.vocab.get("row", set())))
        for length, count in _length_profile(len(seqs.row)).items():
            self._length["row"][length] += count

    def to_dict(self) -> dict:
        n = self.charts
        lane_avg = {}
        for col in range(4):
            lane_avg[f"lane_{col + 1}"] = _avg_counter(self.lane[col], self.vocab.get("lane", set()), n)
        pooled_lane: Counter[MotifKey] = Counter()
        for counter in self.lane:
            pooled_lane.update(counter)
        hand_avg = {
            "12": _avg_counter(self.hand["12"], self.vocab.get("hand", set()), n),
            "34": _avg_counter(self.hand["34"], self.vocab.get("hand", set()), n),
        }
        pooled_hand: Counter[MotifKey] = Counter()
        pooled_hand.update(self.hand["12"])
        pooled_hand.update(self.hand["34"])
        return {
            "charts": n,
            "lane": lane_avg,
            "all_lanes": _avg_counter(pooled_lane, self.vocab.get("lane", set()), n),
            "hand": hand_avg,
            "both_hands": _avg_counter(pooled_hand, self.vocab.get("hand", set()), n),
            "row": _avg_counter(self.row, self.vocab.get("row", set()), n),
            "length_occurrences": {
                "lane": {
                    f"lane_{col + 1}": dict(self._length["lane"][col]) for col in range(4)
                },
                "all_lanes": _sum_counters(self._length["lane"]),
                "hand": {name: dict(self._length["hand"][name]) for name in ("12", "34")},
                "both_hands": _sum_counters(self._length["hand"].values()),
                "row": dict(self._length["row"]),
            },
        }


def _avg_counter(counts: Counter[MotifKey], vocab: set[MotifKey], charts: int) -> dict[str, float]:
    keys = vocab or set(counts)
    return {
        format_motif(key): (counts[key] / charts if charts else 0.0)
        for key in sorted(keys, key=format_motif)
    }


def _sum_counters(counters: Iterable[Counter]) -> dict[str, int]:
    total: Counter = Counter()
    for counter in counters:
        total.update(counter)
    return dict(total)
