"""Reference-onset diagnostics for native POS and ROW decisions, not chart-quality scores."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from audio2map.eval.v1.logits import decision_distribution, token_stats, top_tokens
from audio2map.eval.v1.onset import onset_scores
from audio2map.grid.v1.beatgrid import BeatGrid
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.v1.timing import parse_redlines_file
from audio2map.tokens.v1.codec import encode_chart
from audio2map.tokens.v1.structure import EndpointKind
from audio2map.tokens.v1.vocab import (
    TOKEN_BEAT,
    TOKEN_EOS,
    build_vocab,
    invert_vocab,
    parse_pos_token,
    pos_token,
)
from audio2map.tokens.vocab import LaneState, row_state_from_token

ONSET_TOLERANCE_MS = 2
HEAD_STATES = (LaneState.TAP, LaneState.HOLD_START)


@dataclass(frozen=True)
class ReferenceOnset:
    beat_id: int
    numer: int
    denom: int
    time_ms: float
    heads: int

    @property
    def token(self) -> str:
        return pos_token(self.numer, self.denom)


@dataclass
class ReferenceContext:
    grid: BeatGrid
    by_beat: dict[int, list[ReferenceOnset]]
    onsets_ms: list[float]


def reference_context(path: Path, duration_ms: float) -> ReferenceContext:
    chart = parse_beatmap(path)
    encoded = encode_chart(chart.notes, parse_redlines_file(path).redlines, duration_ms)
    groups = defaultdict(list)
    for i, endpoint in enumerate(encoded.endpoints):
        if endpoint.kind not in (EndpointKind.TAP, EndpointKind.HEAD):
            continue
        key = (
            int(encoded.labels.beat_id[i]),
            int(encoded.labels.numer[i]),
            int(encoded.labels.denom[i]),
        )
        groups[key].append(endpoint.orig_ms)
    by_beat = defaultdict(list)
    for (beat_id, numer, denom), times in groups.items():
        by_beat[beat_id].append(ReferenceOnset(beat_id, numer, denom, min(times), len(times)))
    for entries in by_beat.values():
        entries.sort(key=lambda r: r.numer / r.denom)
    return ReferenceContext(encoded.grid, dict(by_beat), sorted({n.time_ms for n in chart.notes}))


def token_time(token: str, beat_id: int, grid: BeatGrid) -> float:
    numer, denom = parse_pos_token(token)
    beat = grid.beats[beat_id]
    return beat.start_ms + beat.nominal_length_ms * numer / denom


def reference_target(
    row: dict, reference: ReferenceContext, covered: set[tuple]
) -> ReferenceOnset | None:
    previous = row["last_fraction"]
    remaining = [
        r
        for r in reference.by_beat.get(row["beat_id"], ())
        if (r.beat_id, r.numer, r.denom) not in covered
        and (previous is None or r.numer * previous[1] > previous[0] * r.denom)
    ]
    return remaining[0] if remaining else None


def position_stats(
    row: dict, distribution: dict, reference: ReferenceContext, covered: set[tuple]
) -> dict:
    target = reference_target(row, reference, covered)
    token = (
        target.token
        if target
        else (TOKEN_EOS if row["beat_id"] >= len(reference.grid.beats) else TOKEN_BEAT)
    )
    allowed_ids = row["allowed_ids"]
    names = invert_vocab()
    equivalents = (
        [build_vocab()[token]]
        if target is None
        else [
            i
            for i in allowed_ids
            if names[i].startswith("<POS_")
            and abs(
                math.floor(token_time(names[i], row["beat_id"], reference.grid)) - target.time_ms
            )
            <= ONSET_TOLERANCE_MS
        ]
    )
    chosen = row["chosen_token"]
    greedy = distribution["greedy"]
    result = {
        "target": token_stats(distribution, token, allowed_ids),
        "target_time_ms": target.time_ms if target else None,
        "target_equivalent_probability": float(distribution["legal"][equivalents].sum()),
        "greedy_hits_next_reference": build_vocab()[greedy] in equivalents,
        "chosen_hits_next_reference": build_vocab()[chosen] in equivalents,
        "skipped_onsets_ms": [],
        "chosen_has_reference": None,
    }
    if chosen.startswith("<POS_"):
        t = math.floor(token_time(chosen, row["beat_id"], reference.grid))
        previous = row["last_fraction"]
        remaining = [
            r
            for r in reference.by_beat.get(row["beat_id"], ())
            if (r.beat_id, r.numer, r.denom) not in covered
            and (previous is None or r.numer * previous[1] > previous[0] * r.denom)
        ]
        result["chosen_time_ms"] = t
        result["skipped_onsets_ms"] = [
            r.time_ms for r in remaining if r.time_ms < t - ONSET_TOLERANCE_MS
        ]
        result["chosen_has_reference"] = any(
            abs(r.time_ms - t) <= ONSET_TOLERANCE_MS
            for r in reference.by_beat.get(row["beat_id"], ())
        )
    elif chosen == TOKEN_BEAT:
        previous = row["last_fraction"]
        result["skipped_onsets_ms"] = [
            r.time_ms
            for r in reference.by_beat.get(row["beat_id"], ())
            if (r.beat_id, r.numer, r.denom) not in covered
            and (previous is None or r.numer * previous[1] > previous[0] * r.denom)
        ]
    return result


def row_stats(row: dict, distribution: dict, reference: ReferenceContext) -> dict:
    names = invert_vocab()
    probabilities = distribution["legal"]
    sizes = [0.0] * 5
    hold_probability = 0.0
    for i in row["allowed_ids"]:
        lanes = row_state_from_token(names[i])
        heads = sum(state in HEAD_STATES for state in lanes)
        p = float(probabilities[i])
        sizes[heads] += p
        if LaneState.HOLD_START in lanes:
            hold_probability += p
    pending = row["pending_fraction"]
    t = math.floor(token_time(pos_token(*pending), row["beat_id"], reference.grid))
    matched = [
        r
        for r in reference.by_beat.get(row["beat_id"], ())
        if abs(r.time_ms - t) <= ONSET_TOLERANCE_MS
    ]
    lanes = row_state_from_token(row["chosen_token"])
    return {
        "time_ms": t,
        "chosen_heads": sum(state in HEAD_STATES for state in lanes),
        "expected_heads": sum(size * p for size, p in enumerate(sizes)),
        "head_count_probability": sizes,
        "hold_start_probability": hold_probability,
        "reference_heads": matched[0].heads if matched else 0,
        "reference_has_onset": bool(matched),
    }


def state_key(row: dict) -> tuple:
    fraction = row["pending_fraction"] if row["expect_row"] else row["last_fraction"]
    return (
        row["beat_id"],
        row["expect_row"],
        None if fraction is None else tuple(fraction),
        tuple(row["active_lanes"]),
    )


def analyze_trace(
    rows: list[dict], logits: np.ndarray, reference: ReferenceContext, temperature: float
) -> list[dict]:
    results = []
    covered = set()
    for row, raw in zip(rows, logits, strict=True):
        distribution = decision_distribution(raw, row["allowed_ids"], temperature)
        p = distribution["legal"]
        positive = p > 0
        item = {
            **row,
            "legal_mass_raw": distribution["legal_mass_raw"],
            "chosen": token_stats(distribution, row["chosen_token"], row["allowed_ids"]),
            "greedy": distribution["greedy"],
            "entropy": float(-(p[positive] * p[positive].log()).sum()),
            "top": top_tokens(distribution, row["allowed_ids"]),
        }
        item.update(
            row_stats(row, distribution, reference)
            if row["expect_row"]
            else position_stats(row, distribution, reference, covered)
        )
        if row["expect_row"] and item["chosen_heads"] > 0:
            matched = [
                r
                for r in reference.by_beat.get(row["beat_id"], ())
                if (r.beat_id, r.numer, r.denom) not in covered
                and abs(r.time_ms - item["time_ms"]) <= ONSET_TOLERANCE_MS
            ]
            if matched:
                closest = min(matched, key=lambda r: abs(r.time_ms - item["time_ms"]))
                covered.add((closest.beat_id, closest.numer, closest.denom))
        results.append(item)
    return results


def aggregate_trace(decisions: list[dict], reference: ReferenceContext) -> dict:
    actual = [r for r in decisions if not r["forced"]]
    positions = [r for r in actual if not r["expect_row"]]
    note_choices = [r for r in positions if r["chosen_token"].startswith("<POS_")]
    skipping = [r for r in positions if r["skipped_onsets_ms"]]
    extras = [r for r in note_choices if not r["chosen_has_reference"]]
    event_rows = [r for r in actual if r["expect_row"]]
    emitted = [r["time_ms"] for r in event_rows if r["chosen_heads"] > 0]
    cut = min((r["cursor_time_ms"] for r in actual), default=float("inf"))
    refs = [t for t in reference.onsets_ms if t >= cut - ONSET_TOLERANCE_MS]

    def errors(records):
        return {
            "decisions": len(records),
            "greedy_would_hit_next_reference": sum(
                r["greedy_hits_next_reference"] for r in records
            ),
            "chosen_is_greedy": sum(r["chosen_token"] == r["greedy"] for r in records),
            "target_rank_le_5": sum(
                r["target"]["legal_rank"] is not None and r["target"]["legal_rank"] <= 5
                for r in records
            ),
            "median_target_probability": float(
                np.median([r["target_equivalent_probability"] for r in records])
            )
            if records
            else None,
        }

    return {
        "sampled_steps": len(actual),
        "position_steps": len(positions),
        "onset_scores": onset_scores(sorted(set(emitted)), refs),
        "skipping": errors(skipping),
        "extra_positions": errors(extras),
        "skipped_reference_onsets": sum(len(r["skipped_onsets_ms"]) for r in skipping),
        "extra_position_count": len(extras),
        "matched_row_count": sum(r["reference_has_onset"] for r in event_rows),
        "matched_multi_reference_rows": sum(r["reference_heads"] > 1 for r in event_rows),
        "mean_multi_probability_on_multi_reference": float(
            np.mean(
                [
                    sum(r["head_count_probability"][2:])
                    for r in event_rows
                    if r["reference_heads"] > 1
                ]
            )
        )
        if any(r["reference_heads"] > 1 for r in event_rows)
        else None,
        "mean_expected_heads_on_multi_reference": float(
            np.mean([r["expected_heads"] for r in event_rows if r["reference_heads"] > 1])
        )
        if any(r["reference_heads"] > 1 for r in event_rows)
        else None,
    }


def compare_history(actual: list[dict], teacher: list[dict]) -> list[dict]:
    lookup = {state_key(r): r for r in teacher if not r["expect_row"]}
    paired = []
    for row in actual:
        if row["expect_row"] or row["forced"]:
            continue
        other = lookup.get(state_key(row))
        if other is None or other["allowed_ids"] != row["allowed_ids"]:
            continue
        if other["target"]["token"] != row["target"]["token"]:
            continue
        paired.append(
            {
                "position": row["position"],
                "beat_id": row["beat_id"],
                "last_fraction": row["last_fraction"],
                "cursor_time_ms": row["cursor_time_ms"],
                "target": row["target"]["token"],
                "generated_history_probability": row["target_equivalent_probability"],
                "human_history_probability": other["target_equivalent_probability"],
                "skips_reference": bool(row["skipped_onsets_ms"]),
            }
        )
    return paired


def compare_audio(actual: list[dict], muted: list[dict]) -> list[dict]:
    if len(actual) != len(muted):
        raise ValueError("matched token histories must have equal trace length")
    results = []
    for row, other in zip(actual, muted, strict=True):
        if state_key(row) != state_key(other) or row["chosen_token"] != other["chosen_token"]:
            raise ValueError("audio intervention changed the forced token history")
        item = {
            "position": row["position"],
            "expect_row": row["expect_row"],
            "cursor_time_ms": row["cursor_time_ms"],
            "beat_id": row["beat_id"],
        }
        if row["expect_row"]:
            item.update(
                real_expected_heads=row["expected_heads"],
                muted_expected_heads=other["expected_heads"],
            )
        else:
            item.update(
                target=row["target"]["token"],
                real_target_probability=row["target_equivalent_probability"],
                muted_target_probability=other["target_equivalent_probability"],
                skips_reference=bool(row["skipped_onsets_ms"]),
            )
        results.append(item)
    return results
