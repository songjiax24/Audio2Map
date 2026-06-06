"""Parse .osu beatmap files into structured sections."""

from __future__ import annotations

from pathlib import Path

from audio2map.osu.mania import is_mania_4k_sections, parse_hit_object
from audio2map.osu.schema import Beatmap, ChartMetadata, ManiaNote, TimingPoint


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


def _section_kv(section: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in section:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def parse_timing_points(lines: list[str]) -> list[TimingPoint]:
    return _parse_timing_points(lines)


def _parse_timing_points(lines: list[str]) -> list[TimingPoint]:
    points: list[TimingPoint] = []
    for line in lines:
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            points.append(
                TimingPoint(
                    offset_ms=int(float(parts[0])),
                    beat_length_ms=float(parts[1]),
                    meter=int(parts[2]),
                    uninherited=parts[6].strip() == "1",
                )
            )
        except (ValueError, IndexError):
            continue
    return points


def _parse_metadata(sections: dict[str, list[str]]) -> ChartMetadata:
    general = _section_kv(sections.get("[General]", []))
    meta = _section_kv(sections.get("[Metadata]", []))
    diff = _section_kv(sections.get("[Difficulty]", []))

    return ChartMetadata(
        title=meta.get("Title", ""),
        artist=meta.get("Artist", ""),
        creator=meta.get("Creator", ""),
        version=meta.get("Version", ""),
        beatmap_id=_parse_int(meta.get("BeatmapID")),
        beatmap_set_id=_parse_int(meta.get("BeatmapSetID")),
        hp=_parse_float(diff.get("HPDrainRate")),
        circle_size=_parse_float(diff.get("CircleSize")),
        overall_difficulty=_parse_float(diff.get("OverallDifficulty")),
        approach_rate=_parse_float(diff.get("ApproachRate")),
        slider_multiplier=_parse_float(diff.get("SliderMultiplier"), default=1.0),
        audio_filename=general.get("AudioFilename", ""),
    )


def _parse_notes(lines: list[str]) -> list[ManiaNote]:
    notes: list[ManiaNote] = []
    seen: set[tuple[int, int, NoteType]] = set()
    for line in lines:
        note = parse_hit_object(line)
        if note is None:
            continue
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
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = parse_sections(text)

    if not is_mania_4k_sections(sections):
        raise ValueError(f"not a mania 4K beatmap: {path}")

    metadata = _parse_metadata(sections)
    timing_points = parse_timing_points(sections.get("[TimingPoints]", []))
    notes = _parse_notes(sections.get("[HitObjects]", []))

    return Beatmap(
        path=path,
        metadata=metadata,
        timing_points=timing_points,
        notes=notes,
    )
