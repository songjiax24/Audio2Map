"""Export generated notes to ``.osu`` format."""

from __future__ import annotations

from pathlib import Path

from audio2map.osu.mania import COL4K_X
from audio2map.osu.schema import Beatmap, ManiaNote, NoteType

MANIA_Y = 192
HIT_SAMPLE = "0:0:0:0:"


def filter_notes_for_export(notes: list[ManiaNote], *, min_time_ms: int = 0) -> list[ManiaNote]:
    """Drop notes osu! cannot represent (negative times, invalid holds)."""
    out: list[ManiaNote] = []
    for note in notes:
        if note.time_ms < min_time_ms:
            continue
        if note.note_type == NoteType.TAP:
            out.append(note)
            continue
        end = note.end_time_ms
        if end is None or end <= note.time_ms:
            continue
        out.append(note)
    return out


def note_to_hit_object(note: ManiaNote) -> str:
    x = COL4K_X[note.col]
    if note.note_type == NoteType.TAP:
        return f"{x},{MANIA_Y},{note.time_ms},1,0,{HIT_SAMPLE}"
    assert note.end_time_ms is not None
    return f"{x},{MANIA_Y},{note.time_ms},128,0,{note.end_time_ms}:{HIT_SAMPLE}"


def _sanitize_export_line(line: str, *, in_events: bool) -> str | None:
    """Drop template-only assets/metadata that break local editor loads."""
    if line.startswith("BeatmapID:"):
        return "BeatmapID:0"
    if not in_events:
        return line
    if line.startswith("Video,"):
        return None
    if line.startswith("0,0,") and any(ext in line.lower() for ext in (".jpg", ".jpeg", ".png", ".bmp")):
        return None
    return line


def export_beatmap_notes(
    template_path: Path,
    notes: list[ManiaNote],
    output_path: Path,
    *,
    version_suffix: str = " (Audio2Map)",
) -> Path:
    """Clone ``template_path`` and replace ``[HitObjects]`` with ``notes``."""
    text = template_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    notes = filter_notes_for_export(notes)

    out_lines: list[str] = []
    in_hit = False
    in_events = False
    replaced_version = False
    for line in lines:
        if line.startswith("[Events]"):
            in_events = True
            out_lines.append(line)
            continue
        if in_events and line.startswith("["):
            in_events = False
        if line.startswith("[HitObjects]"):
            in_hit = True
            out_lines.append(line)
            for note in sorted(notes, key=lambda n: (n.time_ms, n.col)):
                out_lines.append(note_to_hit_object(note))
            continue
        if in_hit:
            if line.startswith("["):
                in_hit = False
            else:
                continue
        if not replaced_version and line.startswith("Version:"):
            out_lines.append(f"Version:{line.split(':', 1)[1].strip()}{version_suffix}")
            replaced_version = True
            continue
        sanitized = _sanitize_export_line(line, in_events=in_events)
        if sanitized is not None:
            out_lines.append(sanitized)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return output_path


def export_from_beatmap(
    beatmap: Beatmap,
    output_path: Path,
    *,
    version_suffix: str = " (Audio2Map)",
) -> Path:
    return export_beatmap_notes(
        beatmap.path,
        beatmap.notes,
        output_path,
        version_suffix=version_suffix,
    )
