"""Bar-position 5-state ROW tokenisation (v2).

Token window::

    <BOS>
    <ROW_initial>
    <BAR>
    <POS_x> <ROW_abcd>
    ...
    <EOS>

Lane states (col0..col3 digits):

- ``0`` empty
- ``1`` tap
- ``2`` hold_start
- ``3`` hold_active
- ``4`` hold_end

Initial ``<ROW_*>`` uses only states ``{0, 3}``. Event rows must include at least
one of ``{1, 2, 4}``; pure ``0/3`` rows are never emitted as events.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from audio2map.osu.bpm import canonicalize_bpm
from audio2map.osu.grid_config import TICKS_PER_BAR, TICKS_PER_BEAT
from audio2map.osu.tick_range import chart_event_bar_range_from_ticks, split_absolute_tick
from audio2map.osu.schema import Beatmap, ManiaNote, NoteType, TimingPoint
from audio2map.osu.timing import beat_length_to_bpm

TOKEN_PAD = "<PAD>"
TOKEN_BOS = "<BOS>"
TOKEN_EOS = "<EOS>"
TOKEN_BAR = "<BAR>"
TOKEN_POS_PREFIX = "<POS_"
TOKEN_ROW_PREFIX = "<ROW_"

REAL_EVENT_STATES = frozenset({1, 2, 4})
INITIAL_LANE_STATES = frozenset({0, 3})


class LaneState(IntEnum):
    EMPTY = 0
    TAP = 1
    HOLD_START = 2
    HOLD_ACTIVE = 3
    HOLD_END = 4


RowState = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CanonicalTiming:
    offset_ms: int
    original_bpm: float
    canonical_bpm: float
    bpm_scale_exp: int

    @property
    def beat_ms(self) -> float:
        return 60_000.0 / self.canonical_bpm

    @property
    def tick_ms(self) -> float:
        return self.beat_ms / TICKS_PER_BEAT

    @classmethod
    def from_timing_points(cls, timing_points: list[TimingPoint]) -> CanonicalTiming:
        for tp in timing_points:
            if tp.uninherited:
                original_bpm = beat_length_to_bpm(tp.beat_length_ms)
                canonical_bpm, scale_exp = canonicalize_bpm(original_bpm)
                return cls(
                    offset_ms=tp.offset_ms,
                    original_bpm=original_bpm,
                    canonical_bpm=canonical_bpm,
                    bpm_scale_exp=scale_exp,
                )
        raise ValueError("no uninherited timing point")

    @classmethod
    def from_beatmap(cls, beatmap: Beatmap) -> CanonicalTiming:
        return cls.from_timing_points(beatmap.timing_points)


def ms_to_tick(time_ms: int, timing: CanonicalTiming) -> int:
    return round((time_ms - timing.offset_ms) / timing.tick_ms)


def tick_to_ms(tick: int, timing: CanonicalTiming) -> int:
    return timing.offset_ms + round(tick * timing.tick_ms)


def split_tick(tick: int) -> tuple[int, int]:
    """``(absolute_bar, pos)``; ``absolute_tick`` may be negative."""
    return split_absolute_tick(tick)


def all_event_ticks(notes: list[ManiaNote], timing: CanonicalTiming) -> list[int]:
    ticks: list[int] = []
    for note in notes:
        ticks.append(ms_to_tick(note.time_ms, timing))
        if note.note_type == NoteType.HOLD and note.end_time_ms is not None:
            ticks.append(ms_to_tick(note.end_time_ms, timing))
    return ticks


def chart_event_bar_range(
    notes: list[ManiaNote],
    timing: CanonicalTiming,
) -> tuple[int, int]:
    return chart_event_bar_range_from_ticks(all_event_ticks(notes, timing))


def row_state_to_token(row: RowState) -> str:
    if any(s < 0 or s > 4 for s in row):
        raise ValueError(f"invalid lane states: {row}")
    return TOKEN_ROW_PREFIX + "".join(str(s) for s in row) + ">"


def row_state_from_token(token: str) -> RowState:
    if not token.startswith(TOKEN_ROW_PREFIX) or not token.endswith(">"):
        raise ValueError(f"invalid ROW token: {token!r}")
    digits = token[len(TOKEN_ROW_PREFIX) : -1]
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError(f"invalid ROW token: {token!r}")
    states = tuple(int(c) for c in digits)
    if any(s > 4 for s in states):
        raise ValueError(f"lane state out of range in {token!r}")
    return states  # type: ignore[return-value]


def pos_token(pos: int) -> str:
    if pos < 0 or pos >= TICKS_PER_BAR:
        raise ValueError(f"pos must be 0..{TICKS_PER_BAR - 1}, got {pos}")
    return f"{TOKEN_POS_PREFIX}{pos}>"


def pos_from_token(token: str) -> int:
    if not token.startswith(TOKEN_POS_PREFIX) or not token.endswith(">"):
        raise ValueError(f"invalid POS token: {token!r}")
    pos = int(token[len(TOKEN_POS_PREFIX) : -1])
    if pos < 0 or pos >= TICKS_PER_BAR:
        raise ValueError(f"pos out of range in {token!r}")
    return pos


def is_initial_row(row: RowState) -> bool:
    return all(s in INITIAL_LANE_STATES for s in row)


def is_event_row(row: RowState) -> bool:
    return any(s in REAL_EVENT_STATES for s in row)


def initial_row_from_active(active_hold: list[bool]) -> RowState:
    return tuple(
        LaneState.HOLD_ACTIVE if active else LaneState.EMPTY for active in active_hold
    )  # type: ignore[return-value]


def active_hold_from_initial_row(row: RowState) -> list[bool]:
    if not is_initial_row(row):
        raise ValueError(f"not an initial ROW: {row}")
    return [s == LaneState.HOLD_ACTIVE for s in row]


def _merge_lane(cells: dict[int, int], col: int, state: LaneState) -> None:
    prev = cells.get(col, LaneState.EMPTY)
    if prev != LaneState.EMPTY and state != LaneState.EMPTY:
        raise ValueError(f"conflicting lane {col}: {prev} vs {state}")
    cells[col] = int(state)


def notes_to_raw_events(
    notes: list[ManiaNote],
    timing: CanonicalTiming,
) -> dict[int, dict[int, LaneState]]:
    """``tick -> {col: real event state}`` for tap / hold_start / hold_end only."""
    events: dict[int, dict[int, LaneState]] = {}

    def add(tick: int, col: int, state: LaneState) -> None:
        row = events.setdefault(tick, {})
        _merge_lane(row, col, state)

    for note in notes:
        head = ms_to_tick(note.time_ms, timing)
        if note.note_type == NoteType.TAP:
            add(head, note.col, LaneState.TAP)
        else:
            assert note.end_time_ms is not None
            add(head, note.col, LaneState.HOLD_START)
            add(ms_to_tick(note.end_time_ms, timing), note.col, LaneState.HOLD_END)
    return events


def active_hold_at_tick(
    raw_events: dict[int, dict[int, LaneState]],
    tick: int,
    active: list[bool] | None = None,
) -> list[bool]:
    """Replay raw events up to (but not including) ``tick`` to get active holds."""
    state = [False, False, False, False] if active is None else list(active)
    for t in sorted(k for k in raw_events if k < tick):
        for col, ev in raw_events[t].items():
            if ev == LaneState.HOLD_START:
                state[col] = True
            elif ev == LaneState.HOLD_END:
                state[col] = False
    return state


def _build_event_row(raw: dict[int, LaneState], active: list[bool]) -> RowState:
    row: list[int] = []
    for col in range(4):
        if col in raw:
            row.append(int(raw[col]))
        elif active[col]:
            row.append(LaneState.HOLD_ACTIVE)
        else:
            row.append(LaneState.EMPTY)
    result = tuple(row)  # type: ignore[assignment]
    if not is_event_row(result):
        raise ValueError(f"event row missing real events: {result}")
    return result


def raw_events_to_bar_rows(
    raw_events: dict[int, dict[int, LaneState]],
    *,
    start_tick: int = 0,
    end_tick: int | None = None,
) -> dict[int, dict[int, RowState]]:
    """Convert raw tick events to ``bar -> {pos -> ROW}`` with hold_active fill-in."""
    if end_tick is None:
        end_tick = max(raw_events) if raw_events else start_tick
    active = active_hold_at_tick(raw_events, start_tick)
    bars: dict[int, dict[int, RowState]] = {}

    for tick in sorted(t for t in raw_events if start_tick <= t < end_tick):
        bar_index, pos = split_tick(tick)
        row = _build_event_row(raw_events[tick], active)
        bars.setdefault(bar_index, {})[pos] = row
        for col, ev in raw_events[tick].items():
            if ev == LaneState.HOLD_START:
                active[col] = True
            elif ev == LaneState.HOLD_END:
                active[col] = False

    return bars


def beatmap_bar_rows(
    beatmap: Beatmap,
    timing: CanonicalTiming | None = None,
) -> dict[int, dict[int, RowState]]:
    timing = timing or CanonicalTiming.from_beatmap(beatmap)
    raw = notes_to_raw_events(beatmap.notes, timing)
    if not raw:
        return {}
    start_bar, end_bar = chart_event_bar_range(beatmap.notes, timing)
    return raw_events_to_bar_rows(
        raw,
        start_tick=start_bar * TICKS_PER_BAR,
        end_tick=end_bar * TICKS_PER_BAR,
    )


def encode_window_tokens(
    bar_rows: dict[int, dict[int, RowState]],
    *,
    start_bar: int,
    end_bar: int,
    initial_row: RowState,
    include_bos_eos: bool = True,
) -> list[str]:
    """Encode ``[start_bar, end_bar)`` bars into a token window."""
    tokens: list[str] = []
    if include_bos_eos:
        tokens.extend([TOKEN_BOS, row_state_to_token(initial_row)])
    for bar in range(start_bar, end_bar):
        tokens.append(TOKEN_BAR)
        if bar in bar_rows:
            for pos in sorted(bar_rows[bar]):
                tokens.append(pos_token(pos))
                tokens.append(row_state_to_token(bar_rows[bar][pos]))
    if include_bos_eos:
        tokens.append(TOKEN_EOS)
    return tokens


def beatmap_to_window_tokens(
    beatmap: Beatmap,
    *,
    start_bar: int = 0,
    window_bars: int | None = None,
    timing: CanonicalTiming | None = None,
) -> list[str]:
    timing = timing or CanonicalTiming.from_beatmap(beatmap)
    raw = notes_to_raw_events(beatmap.notes, timing)
    if window_bars is None:
        chart_start, chart_end = chart_event_bar_range(beatmap.notes, timing)
        start = chart_start if start_bar == 0 else start_bar
        end_bar = chart_end
    else:
        start = start_bar
        end_bar = start_bar + window_bars

    start_tick = start * TICKS_PER_BAR
    end_tick = end_bar * TICKS_PER_BAR
    active = active_hold_at_tick(raw, start_tick)
    initial = initial_row_from_active(active)
    bars = raw_events_to_bar_rows(raw, start_tick=start_tick, end_tick=end_tick)
    return encode_window_tokens(
        bars, start_bar=start, end_bar=end_bar, initial_row=initial
    )


def beatmap_to_row_tokens(beatmap: Beatmap, timing: CanonicalTiming | None = None) -> list[str]:
    """Full-chart token sequence over the bar-aligned event range."""
    timing = timing or CanonicalTiming.from_beatmap(beatmap)
    if not beatmap.notes:
        return [TOKEN_BOS, row_state_to_token((0, 0, 0, 0)), TOKEN_EOS]
    start_bar, end_bar = chart_event_bar_range(beatmap.notes, timing)
    return beatmap_to_window_tokens(
        beatmap,
        start_bar=start_bar,
        window_bars=end_bar - start_bar,
        timing=timing,
    )


# --- decoding constraints ---


def is_legal_row(row: RowState, active_hold: list[bool], *, allow_initial: bool = False) -> bool:
    if allow_initial and is_initial_row(row):
        return True
    if not is_event_row(row):
        return False
    for lane, state in enumerate(row):
        if state == LaneState.EMPTY:
            continue
        if state == LaneState.TAP and active_hold[lane]:
            return False
        if state == LaneState.HOLD_START and active_hold[lane]:
            return False
        if state == LaneState.HOLD_ACTIVE and not active_hold[lane]:
            return False
        if state == LaneState.HOLD_END and not active_hold[lane]:
            return False
        if state not in tuple(LaneState):
            return False
    return True


def apply_row(row: RowState, active_hold: list[bool]) -> None:
    for lane, state in enumerate(row):
        if state == LaneState.HOLD_START:
            active_hold[lane] = True
        elif state == LaneState.HOLD_END:
            active_hold[lane] = False


def validate_token_sequence(tokens: list[str]) -> list[str]:
    errors: list[str] = []
    if not tokens or tokens[0] != TOKEN_BOS:
        errors.append("sequence must start with <BOS>")
        return errors

    idx = 1
    if idx >= len(tokens) or not tokens[idx].startswith(TOKEN_ROW_PREFIX):
        errors.append("expected initial <ROW_*> after <BOS>")
        return errors

    initial = row_state_from_token(tokens[idx])
    if not is_initial_row(initial):
        errors.append(f"initial ROW must use only states 0/3: {tokens[idx]}")
    active = active_hold_from_initial_row(initial)
    idx += 1

    in_bar = False
    last_pos: int | None = None
    expect_row = False

    while idx < len(tokens):
        tok = tokens[idx]
        if tok == TOKEN_EOS:
            if expect_row:
                errors.append(f"token {idx}: <EOS> while expecting ROW")
            break
        if tok == TOKEN_BAR:
            in_bar = True
            last_pos = None
            expect_row = False
            idx += 1
            continue
        if tok.startswith(TOKEN_POS_PREFIX):
            if not in_bar:
                errors.append(f"token {idx}: POS outside BAR")
            pos = pos_from_token(tok)
            if last_pos is not None and pos <= last_pos:
                errors.append(f"token {idx}: POS {pos} not strictly increasing")
            last_pos = pos
            expect_row = True
            idx += 1
            continue
        if tok.startswith(TOKEN_ROW_PREFIX):
            if not in_bar or not expect_row:
                errors.append(f"token {idx}: ROW without POS")
            else:
                row = row_state_from_token(tok)
                if not is_legal_row(row, active):
                    errors.append(f"token {idx}: illegal ROW {tok} active={active}")
                else:
                    apply_row(row, active)
                expect_row = False
            idx += 1
            continue
        errors.append(f"token {idx}: unknown token {tok!r}")
        idx += 1

    if idx >= len(tokens) or tokens[idx] != TOKEN_EOS:
        errors.append("sequence must end with <EOS>")
    return errors


def build_vocab() -> dict[str, int]:
    vocab: dict[str, int] = {
        TOKEN_PAD: 0,
        TOKEN_BOS: 1,
        TOKEN_EOS: 2,
        TOKEN_BAR: 3,
    }
    idx = 4
    for pos in range(TICKS_PER_BAR):
        vocab[pos_token(pos)] = idx
        idx += 1
    for a in range(5):
        for b in range(5):
            for c in range(5):
                for d in range(5):
                    vocab[row_state_to_token((a, b, c, d))] = idx
                    idx += 1
    return vocab


def invert_vocab(vocab: dict[str, int] | None = None) -> dict[int, str]:
    vocab = vocab or build_vocab()
    return {idx: tok for tok, idx in vocab.items()}


def event_row_token_ids(vocab: dict[str, int]) -> set[int]:
    """ROW token ids the model may emit as chart events (excludes initial-only rows)."""
    ids: set[int] = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                for d in range(5):
                    row = (a, b, c, d)
                    if is_event_row(row):
                        ids.add(vocab[row_state_to_token(row)])
    return ids


def initial_row_token_ids(vocab: dict[str, int]) -> set[int]:
    ids: set[int] = set()
    for a in (0, 3):
        for b in (0, 3):
            for c in (0, 3):
                for d in (0, 3):
                    ids.add(vocab[row_state_to_token((a, b, c, d))])
    return ids
