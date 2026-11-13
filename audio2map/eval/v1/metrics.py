"""Grid endpoint matching, distributions, and independent playability counts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations

from audio2map.eval.adherence import condition_mae
from audio2map.eval.note_match import match_counts as counter_match
from audio2map.features.cond.chart import ChartMeta
from audio2map.features.cond.v1.vec import COND_SOURCE_FIELDS
from audio2map.generate.v1.notes import ALMOST_CONCURRENT_MS, CONCURRENT_MS, V1Note
from audio2map.grid.v1.beatgrid import BeatGrid
from audio2map.tokens.v1.codec import DecodedEvent
from audio2map.tokens.v1.structure import EndpointKind
from audio2map.tokens.vocab import TOKEN_ROW_PREFIX, row_state_from_token


def event_key(event: DecodedEvent) -> tuple:
    return (event.lane, event.section, event.beat_k, event.numer, event.denom, event.kind.value)


def note_pairs(events: Sequence[DecodedEvent]) -> list[tuple[DecodedEvent, DecodedEvent | None]]:
    """Complete notes. An isolated head is an endpoint, but is not a complete hold."""
    heads = {}
    notes = []
    for event in events:
        if event.kind is EndpointKind.TAP:
            notes.append((event, None))
        elif event.kind is EndpointKind.HEAD:
            heads[event.lane] = event
        else:
            notes.append((heads.pop(event.lane), event))
    return notes


def endpoint_match(predicted: Sequence[DecodedEvent], reference: Sequence[DecodedEvent]) -> dict:
    def selected(events, kind=None, lane=None):
        return Counter(
            event_key(event)
            for event in events
            if (kind is None or event.kind is kind) and (lane is None or event.lane == lane)
        )

    aware = selected(predicted)
    expected = selected(reference)

    def without_lane(counter):
        pooled = Counter()
        for key, count in counter.items():
            pooled[key[1:]] += count
        return pooled

    def event_block(pred, gold):
        return {
            "all": counter_match(pred, gold),
            **{
                kind.value: counter_match(
                    Counter({k: n for k, n in pred.items() if k[-1] == kind.value}),
                    Counter({k: n for k, n in gold.items() if k[-1] == kind.value}),
                )
                for kind in EndpointKind
            },
        }

    def note_counter(events):
        return Counter(
            (head.lane, "tap" if tail is None else "hold", event_key(head)[1:-1],
             None if tail is None else event_key(tail)[1:-1])
            for head, tail in note_pairs(events)
        )

    def note_block(pred, gold, type_index):
        return {
            "all": counter_match(pred, gold),
            **{
                kind: counter_match(
                    Counter({k: n for k, n in pred.items() if k[type_index] == kind}),
                    Counter({k: n for k, n in gold.items() if k[type_index] == kind}),
                )
                for kind in ("tap", "hold")
            },
        }

    pred_notes, gold_notes = note_counter(predicted), note_counter(reference)
    return {
        "all": counter_match(selected(predicted), selected(reference)),
        "by_kind": {
            kind.value: counter_match(selected(predicted, kind), selected(reference, kind))
            for kind in EndpointKind
        },
        "by_lane": {
            str(lane): counter_match(selected(predicted, lane=lane), selected(reference, lane=lane))
            for lane in range(4)
        },
        "event": {
            "lane_aware": event_block(aware, expected),
            "lane_agnostic": event_block(without_lane(aware), without_lane(expected)),
        },
        "note": {
            "lane_aware": note_block(pred_notes, gold_notes, 1),
            "lane_agnostic": note_block(without_lane(pred_notes), without_lane(gold_notes), 0),
        },
    }


def condition_error(target: ChartMeta, generated: ChartMeta) -> dict:
    """The scenario's masked meta is the target, including for random scenarios."""
    return condition_mae(target, generated, names=COND_SOURCE_FIELDS)


def event_distribution(
    events: Sequence[DecodedEvent], grid: BeatGrid, tokens: Sequence[str] | None = None
) -> dict:
    ordinal = {
        (point.section, point.beat_k, point.numer, point.denom): index
        for index, point in enumerate(point for beat in grid.points for point in beat)
    }
    denominators: Counter[str] = Counter()
    lanes = {str(lane): {kind.value: 0 for kind in EndpointKind} for lane in range(4)}
    grid_spans: Counter[str] = Counter()
    beat_spans: Counter[str] = Counter()
    rows = {}
    phases = Counter()
    for event in events:
        denominators[str(event.denom)] += 1
        lanes[str(event.lane)][event.kind.value] += 1
        phases[f"{event.numer}/{event.denom}"] += 1
        rows[(event.section, event.beat_k, event.numer, event.denom)] = True
    notes = note_pairs(events)
    for head, event in notes:
        if event is not None:
            here = (event.section, event.beat_k, event.numer, event.denom)
            there = (head.section, head.beat_k, head.numer, head.denom)
            grid_spans[str(ordinal[here] - ordinal[there])] += 1
            span = (
                event.global_b
                - head.global_b
                + Fraction(event.numer, event.denom)
                - Fraction(head.numer, head.denom)
            )
            beat_spans[str(span)] += 1
    return {
        "endpoints": len(events),
        "denominator": dict(denominators),
        "lanes": lanes,
        "hold_grid_span": dict(grid_spans),
        "hold_beat_span": dict(beat_spans),
        "position": dict(phases),
        "notes": len(notes),
        "taps": sum(tail is None for _, tail in notes),
        "holds": sum(tail is not None for _, tail in notes),
        "rows": len(rows),
        "row_pattern": dict(Counter(
            "".join(str(int(state)) for state in row_state_from_token(token))
            for token in tokens if token.startswith(TOKEN_ROW_PREFIX)
        )) if tokens is not None else None,
    }


def plausibility(notes: Sequence[V1Note]) -> dict:
    """§9.1 compares adjacent objects per lane; the two thresholds are independent."""
    concurrent = almost = duplicates = shut = 0
    for lane in range(4):
        ordered = sorted(
            (note for note in notes if note.col == lane),
            key=lambda note: (note.time_ms, _end(note)),
        )
        shut += sum(
            note.end_time_ms is not None and note.end_time_ms <= note.time_ms for note in ordered
        )
        for previous, following in zip(ordered, ordered[1:]):
            gap = following.time_ms - _end(previous)
            concurrent += gap <= CONCURRENT_MS
            almost += abs(gap) < ALMOST_CONCURRENT_MS
            duplicates += following.time_ms == previous.time_ms
    return {
        "concurrent": concurrent,
        "almost_concurrent": almost,
        "duplicate_start": duplicates,
        "nonpositive_hold": shut,
        "has_concurrent": bool(concurrent),
        "has_almost_concurrent": bool(almost),
    }


def diversity(samples: dict[int, Sequence[DecodedEvent]]) -> dict:
    pairs = []
    for left, right in combinations(sorted(samples), 2):
        a = Counter(event_key(event) for event in samples[left])
        b = Counter(event_key(event) for event in samples[right])
        pairs.append({
            "seeds": [left, right], "identical": a == b, **counter_match(a, b),
            "match": endpoint_match(samples[left], samples[right]),
        })
    return {
        "pairs": pairs,
        "identical_pair_rate": sum(row["identical"] for row in pairs) / len(pairs)
        if pairs
        else None,
    }


def _end(note: V1Note) -> int:
    return note.time_ms if note.end_time_ms is None else note.end_time_ms
