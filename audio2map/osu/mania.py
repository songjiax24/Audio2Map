"""osu!mania 4K hit object parsing."""

from __future__ import annotations

from audio2map.osu.schema import ManiaNote, NoteType

# Standard 4K column x positions (CircleSize=4).
COL4K_X = (64, 192, 320, 448)

HIT_HOLD = 128  # mania long note (LN) head


def x_to_column(x: int, *, keys: int = 4) -> int | None:
    """Map hit object ``x`` to column index, or ``None`` if unrecognised."""
    if keys != 4:
        raise NotImplementedError("only 4K column mapping is implemented")
    if x in COL4K_X:
        return COL4K_X.index(x)
    # Tolerate small editor rounding errors.
    nearest = min(COL4K_X, key=lambda cx: abs(cx - x))
    if abs(nearest - x) <= 2:
        return COL4K_X.index(nearest)
    return None


def is_mania_4k_sections(sections: dict[str, list[str]]) -> bool:
    """Return True when sections describe mania mode with CircleSize 4."""
    general = _kv(sections.get("[General]", []))
    diff = _kv(sections.get("[Difficulty]", []))
    try:
        if int(general.get("Mode", -1)) != 3:
            return False
        cs = float(diff["CircleSize"])
    except (KeyError, ValueError):
        return False
    return abs(cs - 4.0) < 1e-6 or round(cs) == 4


def parse_hit_object(line: str) -> ManiaNote | None:
    """Parse one ``[HitObjects]`` line into a :class:`ManiaNote`, or skip invalid lines."""
    parts = line.split(",")
    if len(parts) < 5:
        return None

    try:
        x = int(parts[0])
        time_ms = int(parts[2])
        type_bits = int(parts[3])
    except ValueError:
        return None

    col = x_to_column(x)
    if col is None:
        return None

    if type_bits & HIT_HOLD:
        end_time_ms = _parse_hold_end(parts[5:])
        if end_time_ms is None or end_time_ms <= time_ms:
            return None
        return ManiaNote(
            time_ms=time_ms,
            col=col,
            note_type=NoteType.HOLD,
            end_time_ms=end_time_ms,
        )

    # Tap / release / other non-LN objects with a note bit.
    if type_bits & 1 or type_bits == 0:
        return ManiaNote(time_ms=time_ms, col=col, note_type=NoteType.TAP)

    return None


def _parse_hold_end(extras: list[str]) -> int | None:
    if not extras:
        return None
    head = extras[0].split(":")[0]
    try:
        return int(head)
    except ValueError:
        return None


def _kv(section: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in section:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out
