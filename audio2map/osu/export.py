"""Export generated notes to ``.osu`` format."""

from __future__ import annotations

from pathlib import Path

from audio2map.osu.mania import COL4K_X
from audio2map.osu.schema import Beatmap, ManiaNote, NoteType


def note_to_hit_object(note: ManiaNote) -> str:
    x = COL4K_X[note.col]
    if note.note_type == NoteType.TAP:
        return f"{x},0,{note.time_ms},1,0,0:0:0:0:"
    assert note.end_time_ms is not None
    return f"{x},0,{note.time_ms},128,0,{note.end_time_ms}:0:0:0:"


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

    out_lines: list[str] = []
    in_hit = False
    replaced_version = False
    for line in lines:
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
        out_lines.append(line)

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
