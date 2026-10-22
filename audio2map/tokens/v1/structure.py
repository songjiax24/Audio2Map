"""Chart structure checks that run before the cursor walk."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.tokens.v1.labels import LabelBatch
from audio2map.tokens.vocab import LaneState, RowState
from audio2map.utils.errors import (
    DUPLICATE_ENDPOINT,
    ENDPOINT_OUTSIDE_AUDIO,
    HOLD_NOT_AFTER_HEAD,
    ChartRejected,
)


class EndpointKind(Enum):
    TAP = "tap"
    HEAD = "head"
    TAIL = "tail"


@dataclass(frozen=True, slots=True)
class Endpoint:
    orig_ms: int
    lane: int
    kind: EndpointKind
    hold_index: int | None


def endpoints_from_notes(notes: list[ManiaNote]) -> list[Endpoint]:
    endpoints: list[Endpoint] = []
    hold_index = 0
    for note in notes:
        if note.note_type == NoteType.TAP:
            endpoints.append(Endpoint(note.time_ms, note.col, EndpointKind.TAP, None))
            continue
        if note.end_time_ms is None:
            raise ChartRejected(HOLD_NOT_AFTER_HEAD, "hold is missing an end time")
        endpoints.append(Endpoint(note.time_ms, note.col, EndpointKind.HEAD, hold_index))
        endpoints.append(Endpoint(note.end_time_ms, note.col, EndpointKind.TAIL, hold_index))
        hold_index += 1
    return endpoints


def check_audio_range(endpoints: list[Endpoint], t_audio_ms: float) -> None:
    for endpoint in endpoints:
        if endpoint.orig_ms < 0 or endpoint.orig_ms >= t_audio_ms:
            raise ChartRejected(
                ENDPOINT_OUTSIDE_AUDIO,
                f"lane={endpoint.lane} kind={endpoint.kind.value} orig={endpoint.orig_ms}",
            )


def _later(
    label_beat: int,
    numer: int,
    denom: int,
    other_beat: int,
    other_n: int,
    other_d: int,
) -> bool:
    if label_beat != other_beat:
        return label_beat > other_beat
    return other_n * denom < numer * other_d


def check_holds(endpoints: list[Endpoint], labels: LabelBatch) -> None:
    by_hold: dict[int, dict[EndpointKind, int]] = {}
    for index, endpoint in enumerate(endpoints):
        if endpoint.hold_index is None:
            continue
        by_hold.setdefault(endpoint.hold_index, {})[endpoint.kind] = index
    for hold_index, parts in by_hold.items():
        head = parts.get(EndpointKind.HEAD)
        tail = parts.get(EndpointKind.TAIL)
        if head is None or tail is None:
            raise ChartRejected(HOLD_NOT_AFTER_HEAD, f"hold {hold_index} is incomplete")
        if not _later(
            int(labels.beat_id[tail]),
            int(labels.numer[tail]),
            int(labels.denom[tail]),
            int(labels.beat_id[head]),
            int(labels.numer[head]),
            int(labels.denom[head]),
        ):
            raise ChartRejected(
                HOLD_NOT_AFTER_HEAD,
                f"hold {hold_index} tail is not on a later grid point",
            )


def rows_in_order(
    endpoints: list[Endpoint],
    labels: LabelBatch,
) -> list[tuple[int, int, int, RowState]]:
    """``(beat_id, numer, denom, row)`` in token order.

    Lanes that are already holding, and have no new event in the cell, stay
    ``HOLD_ACTIVE``.
    """
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, _endpoint in enumerate(endpoints):
        key = (int(labels.beat_id[index]), int(labels.numer[index]), int(labels.denom[index]))
        groups.setdefault(key, []).append(index)
    ordered_keys = sorted(groups, key=lambda item: (item[0], item[1] / item[2], item[1], item[2]))
    active = [False, False, False, False]
    ordered: list[tuple[int, int, int, RowState]] = []
    for key in ordered_keys:
        lanes = [LaneState.HOLD_ACTIVE if holding else LaneState.EMPTY for holding in active]
        for index in groups[key]:
            _place(lanes, endpoints[index], key)
        row = (lanes[0], lanes[1], lanes[2], lanes[3])
        for lane, state in enumerate(row):
            if state == LaneState.HOLD_START:
                active[lane] = True
            elif state == LaneState.HOLD_END:
                active[lane] = False
        ordered.append((key[0], key[1], key[2], row))
    return ordered


def _place(lanes: list[LaneState], endpoint: Endpoint, key: tuple[int, int, int]) -> None:
    lane = lanes[endpoint.lane]
    where = f"lane={endpoint.lane} cell={key[1]}/{key[2]} beat={key[0]}"
    if endpoint.kind is EndpointKind.TAP:
        if lane == LaneState.HOLD_ACTIVE:
            raise ChartRejected(DUPLICATE_ENDPOINT, f"tap on an open hold lane={endpoint.lane}")
        if lane is not LaneState.EMPTY:
            raise ChartRejected(DUPLICATE_ENDPOINT, where)
        lanes[endpoint.lane] = LaneState.TAP
        return
    if endpoint.kind is EndpointKind.HEAD:
        if lane is not LaneState.EMPTY:
            raise ChartRejected(DUPLICATE_ENDPOINT, where)
        lanes[endpoint.lane] = LaneState.HOLD_START
        return
    if lane is not LaneState.HOLD_ACTIVE:
        raise ChartRejected(HOLD_NOT_AFTER_HEAD, f"tail without an open hold lane={endpoint.lane}")
    lanes[endpoint.lane] = LaneState.HOLD_END
