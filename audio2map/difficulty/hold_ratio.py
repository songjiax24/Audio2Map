"""Hold (LN) ratio — fraction of hit objects that are long notes."""

from __future__ import annotations

from audio2map.osu.schema import Beatmap, NoteType


def hold_ratio(beatmap: Beatmap) -> float:
    """Return LN count / total note count (same definition as ManiaMapAnalyser)."""
    n = beatmap.note_count
    if n == 0:
        return 0.0
    ln = sum(1 for note in beatmap.notes if note.note_type == NoteType.HOLD)
    return ln / n
