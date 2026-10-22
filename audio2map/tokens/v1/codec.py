"""Encode a chart to v1 tokens and decode those tokens back to endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field

from audio2map.grid.v1.beatgrid import BeatGrid, build_beatgrid
from audio2map.osu.schema import ManiaNote
from audio2map.osu.v1.timing import Redline, RedlineParse
from audio2map.tokens.errors import IllegalTokenError
from audio2map.tokens.v1.labels import LabelBatch, label_times
from audio2map.tokens.v1.state import initial_cursor, step
from audio2map.tokens.v1.structure import (
    Endpoint,
    EndpointKind,
    check_audio_range,
    check_holds,
    endpoints_from_notes,
    rows_in_order,
)
from audio2map.tokens.v1.vocab import TOKEN_BEAT, TOKEN_BOS, TOKEN_EOS, parse_pos_token, pos_token
from audio2map.tokens.vocab import LaneState, RowState, row_state_from_token, row_state_to_token

__all__ = [
    "DecodedEvent",
    "EncodeResult",
    "Endpoint",
    "EndpointKind",
    "decode_tokens",
    "encode_chart",
    "encode_parsed",
    "endpoints_from_notes",
]


@dataclass(frozen=True, slots=True)
class DecodedEvent:
    lane: int
    kind: EndpointKind
    section: int
    beat_k: int
    numer: int
    denom: int
    time_ms: float
    global_b: int


@dataclass
class EncodeResult:
    tokens: list[str]
    grid: BeatGrid
    labels: LabelBatch
    endpoints: list[Endpoint]
    near_duplicate_pairs: tuple[tuple[float, float], ...] = field(default_factory=tuple)


def encode_chart(
    notes: list[ManiaNote],
    redlines: tuple[Redline, ...] | list[Redline],
    t_audio_ms: float,
) -> EncodeResult:
    endpoints = endpoints_from_notes(notes)
    check_audio_range(endpoints, t_audio_ms)
    grid = build_beatgrid(redlines, t_audio_ms)
    labels = label_times(grid, [endpoint.orig_ms for endpoint in endpoints])
    check_holds(endpoints, labels)
    by_beat: dict[int, list[tuple[int, int, RowState]]] = {}
    for beat_id, numer, denom, row in rows_in_order(endpoints, labels):
        by_beat.setdefault(beat_id, []).append((numer, denom, row))

    tokens = [TOKEN_BOS]
    cursor = initial_cursor()
    for beat_id in range(len(grid.beats)):
        for numer, denom, row in by_beat.get(beat_id, []):
            pos = pos_token(numer, denom)
            step(cursor, pos, grid)
            tokens.append(pos)
            row_token = row_state_to_token(row)
            step(cursor, row_token, grid)
            tokens.append(row_token)
        step(cursor, TOKEN_BEAT, grid)
        tokens.append(TOKEN_BEAT)
    step(cursor, TOKEN_EOS, grid)
    tokens.append(TOKEN_EOS)
    return EncodeResult(
        tokens=tokens,
        grid=grid,
        labels=labels,
        endpoints=endpoints,
        near_duplicate_pairs=grid.near_duplicate_pairs,
    )


def encode_parsed(notes: list[ManiaNote], parsed: RedlineParse, t_audio_ms: float) -> EncodeResult:
    return encode_chart(notes, parsed.redlines, t_audio_ms)


def decode_tokens(tokens: list[str], grid: BeatGrid) -> list[DecodedEvent]:
    """Walk ``tokens`` with the same cursor used at encode time."""
    if not tokens or tokens[0] != TOKEN_BOS:
        raise IllegalTokenError(tokens[0] if tokens else "")
    cursor = initial_cursor()
    events: list[DecodedEvent] = []
    pending: tuple[int, int] | None = None
    for token in tokens[1:]:
        if token.startswith("<POS_"):
            pending = parse_pos_token(token)
        step(cursor, token, grid)
        if not token.startswith("<ROW_") or pending is None:
            continue
        numer, denom = pending
        point = next(
            item
            for item in grid.points[cursor.beat_id]
            if item.numer == numer and item.denom == denom
        )
        events.extend(_events_from_row(token, point, numer, denom))
        pending = None
    if not cursor.finished:
        raise IllegalTokenError(TOKEN_EOS)
    return events


def _events_from_row(token: str, point, numer: int, denom: int) -> list[DecodedEvent]:
    kind_of = {
        LaneState.TAP: EndpointKind.TAP,
        LaneState.HOLD_START: EndpointKind.HEAD,
        LaneState.HOLD_END: EndpointKind.TAIL,
    }
    events = []
    for lane, state in enumerate(row_state_from_token(token)):
        kind = kind_of.get(state)
        if kind is None:
            continue
        events.append(
            DecodedEvent(
                lane=lane,
                kind=kind,
                section=point.section,
                beat_k=point.beat_k,
                numer=numer,
                denom=denom,
                time_ms=point.time_ms,
                global_b=point.global_b,
            )
        )
    return events
