"""Parse .osu beatmap files into structured sections.

Authoritative parser for training, tokens, eligibility, and export.
Pattern-analyser features use ``audio2map.features.cond.pattern_analyser.osu_parser``;
see ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

from pathlib import Path

from audio2map.osu.schema import Beatmap, ChartMetadata, ManiaNote, NoteType, TimingPoint

HIT_CIRCLE = 1
HIT_HOLD = 128


class InvalidHitObjectError(ValueError):
    pass


class InvalidTimingPointError(ValueError):
    pass


def parse_sections(text: str) -> dict[str, list[str]]:
    """Split an .osu file into ``{section_name: [lines]}``."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            current = line
            sections[current] = []
        elif current is not None and line and not line.startswith("//"):
            sections[current].append(line)
    return sections


def read_osu_sections(path: Path | str) -> dict[str, list[str]]:
    return parse_sections(Path(path).read_text(encoding="utf-8", errors="replace"))


def section_kv(section: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in section:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def is_mania_4k_sections(sections: dict[str, list[str]]) -> bool:
    """Return True when sections describe mania mode with CircleSize 4."""
    general = section_kv(sections.get("[General]", []))
    diff = section_kv(sections.get("[Difficulty]", []))
    try:
        if int(general.get("Mode", -1)) != 3:
            return False
        cs = float(diff["CircleSize"])
    except (KeyError, ValueError):
        return False
    return abs(cs - 4.0) < 1e-6


def is_mania_4k(path: Path | str) -> bool:
    return is_mania_4k_sections(read_osu_sections(path))


def audio_filename(path: Path | str) -> str | None:
    general = section_kv(read_osu_sections(path).get("[General]", []))
    name = general.get("AudioFilename", "").strip()
    return name or None


def chart_audio_path(osu_path: Path | str) -> Path:
    """Audio file named by ``AudioFilename``, in the same directory as the ``.osu``."""
    osu_path = Path(osu_path)
    name = audio_filename(osu_path)
    if not name:
        raise FileNotFoundError(f"missing AudioFilename in {osu_path}")
    path = osu_path.parent / name
    if not path.is_file():
        raise FileNotFoundError(f"audio not found: {path}")
    return path


def x_to_column(x: int, *, keys: int = 4) -> int:
    """Map hit object ``x`` to column index (osu!/Prelude: ``int(x / 512 * keys)``)."""
    if keys != 4:
        raise NotImplementedError("only 4K column mapping is implemented")
    col = int(float(x) / 512.0 * float(keys))
    if col < 0:
        col = 0
    if col > keys - 1:
        col = keys - 1
    return col


def parse_hit_object(line: str) -> ManiaNote:
    """Parse one ``[HitObjects]`` line into a :class:`ManiaNote`."""
    parts = line.split(",")
    if len(parts) < 5:
        raise InvalidHitObjectError(f"too few fields: {line}")

    try:
        x = int(parts[0])
        time_ms = int(parts[2])
        type_bits = int(parts[3])
    except ValueError as exc:
        raise InvalidHitObjectError(f"non-integer fields: {line}") from exc

    col = x_to_column(x)

    if type_bits & HIT_HOLD:
        end_time_ms = _parse_hold_end(parts[5:])
        if end_time_ms is None or end_time_ms <= time_ms:
            raise InvalidHitObjectError(f"invalid hold: {line}")
        return ManiaNote(
            time_ms=time_ms,
            col=col,
            note_type=NoteType.HOLD,
            end_time_ms=end_time_ms,
        )

    if type_bits & HIT_CIRCLE:
        return ManiaNote(time_ms=time_ms, col=col, note_type=NoteType.TAP)

    raise InvalidHitObjectError(f"unsupported type {type_bits}: {line}")


def _parse_hold_end(extras: list[str]) -> int | None:
    if not extras:
        return None
    head = extras[0].split(":")[0]
    try:
        return int(head)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def parse_timing_points(lines: list[str]) -> list[TimingPoint]:
    points: list[TimingPoint] = []
    for line in lines:
        parts = line.split(",")
        if len(parts) < 8:
            raise InvalidTimingPointError(f"too few fields: {line}")
        try:
            points.append(
                TimingPoint(
                    offset_ms=int(float(parts[0])),
                    beat_length_ms=float(parts[1]),
                    meter=int(parts[2]),
                    uninherited=parts[6].strip() == "1",
                )
            )
        except (ValueError, IndexError) as exc:
            raise InvalidTimingPointError(f"invalid fields: {line}") from exc
    return points


def _parse_metadata(sections: dict[str, list[str]]) -> ChartMetadata:
    meta = section_kv(sections.get("[Metadata]", []))
    return ChartMetadata(
        title=meta.get("Title", ""),
        artist=meta.get("Artist", ""),
        creator=meta.get("Creator", ""),
        version=meta.get("Version", ""),
        beatmap_id=_parse_int(meta.get("BeatmapID")),
        beatmap_set_id=_parse_int(meta.get("BeatmapSetID")),
    )


def parse_notes(lines: list[str]) -> list[ManiaNote]:
    notes: list[ManiaNote] = []
    seen: set[tuple[int, int, NoteType]] = set()
    for line in lines:
        note = parse_hit_object(line)
        key = (note.time_ms, note.col, note.note_type)
        if key in seen:
            continue
        seen.add(key)
        notes.append(note)
    notes.sort(key=lambda n: (n.time_ms, n.col, n.note_type.value))
    return notes


def parse_beatmap(path: Path | str) -> Beatmap:
    """Parse a mania 4K ``.osu`` file into a :class:`Beatmap`."""
    path = Path(path)
    sections = read_osu_sections(path)

    if not is_mania_4k_sections(sections):
        raise ValueError(f"not a mania 4K beatmap: {path}")

    metadata = _parse_metadata(sections)
    try:
        timing_points = parse_timing_points(sections.get("[TimingPoints]", []))
    except InvalidTimingPointError as exc:
        raise InvalidTimingPointError(f"{path}: {exc}") from exc
    try:
        notes = parse_notes(sections.get("[HitObjects]", []))
    except InvalidHitObjectError as exc:
        raise InvalidHitObjectError(f"{path}: {exc}") from exc

    return Beatmap(
        path=path,
        metadata=metadata,
        timing_points=timing_points,
        notes=notes,
    )
