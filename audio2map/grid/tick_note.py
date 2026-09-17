"""``ManiaNote`` projected onto the canonical tick grid."""

from __future__ import annotations

from dataclasses import dataclass

from audio2map.grid.timing import CanonicalTiming, ms_to_tick
from audio2map.osu.schema import ManiaNote, NoteType


@dataclass(frozen=True, slots=True)
class TickNote:
    head_tick: int
    col: int
    note_type: NoteType
    end_tick: int | None = None

    @classmethod
    def from_note(cls, note: ManiaNote, timing: CanonicalTiming) -> TickNote:
        end_tick = (
            None if note.end_time_ms is None else ms_to_tick(note.end_time_ms, timing)
        )
        return cls(ms_to_tick(note.time_ms, timing), note.col, note.note_type, end_tick)

    @classmethod
    def from_notes(cls, notes: list[ManiaNote], timing: CanonicalTiming) -> list[TickNote]:
        out = [cls.from_note(n, timing) for n in notes]
        out.sort(key=lambda n: (n.head_tick, n.col))
        return out
