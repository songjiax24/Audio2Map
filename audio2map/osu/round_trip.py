"""Round-trip between chart notes and v2 ROW token windows."""

from __future__ import annotations

from audio2map.osu.grid_config import TICKS_PER_BAR
from audio2map.osu.row_tokens import (
    TOKEN_BAR,
    TOKEN_BOS,
    TOKEN_EOS,
    CanonicalTiming,
    LaneState,
    chart_event_bar_range,
    ms_to_tick,
    row_state_from_token,
    tick_to_ms,
)
from audio2map.osu.schema import Beatmap, ManiaNote, NoteType


def tokens_to_notes(
    tokens: list[str],
    timing: CanonicalTiming,
    *,
    start_bar: int = 0,
) -> list[ManiaNote]:
    """Parse a v2 token window into mania notes (hold_active is not exported)."""
    if not tokens or tokens[0] != TOKEN_BOS:
        raise ValueError("tokens must start with <BOS>")
    if tokens[-1] != TOKEN_EOS:
        raise ValueError("tokens must end with <EOS>")

    notes: list[ManiaNote] = []
    open_holds: dict[int, ManiaNote] = {}
    current_bar = start_bar - 1
    pending_pos: int | None = None

    for tok in tokens[2:-1]:
        if tok == TOKEN_BAR:
            current_bar += 1
            pending_pos = None
            continue
        if tok.startswith("<POS_"):
            from audio2map.osu.row_tokens import pos_from_token

            pending_pos = pos_from_token(tok)
            continue
        if tok.startswith("<ROW_"):
            if pending_pos is None:
                raise ValueError(f"ROW without POS: {tok}")
            row = row_state_from_token(tok)
            tick = current_bar * TICKS_PER_BAR + pending_pos
            time_ms = tick_to_ms(tick, timing)
            for col, state in enumerate(row):
                if state == LaneState.TAP:
                    notes.append(ManiaNote(time_ms=time_ms, col=col, note_type=NoteType.TAP))
                elif state == LaneState.HOLD_START:
                    note = ManiaNote(
                        time_ms=time_ms,
                        col=col,
                        note_type=NoteType.HOLD,
                        end_time_ms=time_ms + 1,
                    )
                    open_holds[col] = note
                    notes.append(note)
                elif state == LaneState.HOLD_END:
                    if col in open_holds:
                        head = open_holds.pop(col)
                        notes.remove(head)
                        notes.append(
                            ManiaNote(
                                time_ms=head.time_ms,
                                col=col,
                                note_type=NoteType.HOLD,
                                end_time_ms=time_ms,
                            )
                        )
                    # hold_end with only initial hold_active: drop orphan tail
            pending_pos = None

    open_holds.clear()

    notes.sort(key=lambda n: (n.time_ms, n.col))
    return notes


def notes_equal(a: list[ManiaNote], b: list[ManiaNote]) -> bool:
    if len(a) != len(b):
        return False
    for na, nb in zip(a, b, strict=True):
        if (
            na.time_ms != nb.time_ms
            or na.col != nb.col
            or na.note_type != nb.note_type
            or na.end_time_ms != nb.end_time_ms
        ):
            return False
    return True


def quantize_notes(notes: list[ManiaNote], timing: CanonicalTiming) -> list[ManiaNote]:
    """Snap notes to the canonical tick grid (Phase 1 training representation)."""
    out: list[ManiaNote] = []
    for note in notes:
        tick = ms_to_tick(note.time_ms, timing)
        time_ms = tick_to_ms(tick, timing)
        if note.note_type == NoteType.TAP:
            out.append(ManiaNote(time_ms=time_ms, col=note.col, note_type=NoteType.TAP))
        else:
            assert note.end_time_ms is not None
            end_tick = ms_to_tick(note.end_time_ms, timing)
            if end_tick < tick:
                continue
            out.append(
                ManiaNote(
                    time_ms=time_ms,
                    col=note.col,
                    note_type=NoteType.HOLD,
                    end_time_ms=tick_to_ms(end_tick, timing),
                )
            )
    out.sort(key=lambda n: (n.time_ms, n.col))
    return out


def round_trip_beatmap(beatmap: Beatmap, timing: CanonicalTiming | None = None) -> bool:
    from audio2map.osu.row_tokens import CanonicalTiming as CT, beatmap_to_window_tokens

    timing = timing or CT.from_beatmap(beatmap)
    expected = quantize_notes(beatmap.notes, timing)
    start_bar, end_bar = chart_event_bar_range(beatmap.notes, timing)
    tokens = beatmap_to_window_tokens(
        beatmap,
        start_bar=start_bar,
        window_bars=end_bar - start_bar,
        timing=timing,
    )
    rebuilt = tokens_to_notes(tokens, timing, start_bar=start_bar)
    return notes_equal(expected, rebuilt)
