"""Write mania 4K ``.osu`` files from notes plus a source chart or BPM/offset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from audio2map.osu.parser import HIT_CIRCLE, HIT_HOLD
from audio2map.osu.schema import ManiaNote, NoteType

COL4K_X = (64, 192, 320, 448)
MANIA_Y = 192
HIT_SAMPLE = "0:0:0:0:"

_ILLEGAL_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_NAME_PART = 80
_MAX_FILENAME = 180


def sanitize_osu_filename_part(raw: str, fallback: str, *, max_len: int = _MAX_NAME_PART) -> str:
    """Strip characters osu! / Windows reject in beatmap filenames."""
    text = _ILLEGAL_FILENAME.sub(" ", raw)
    text = re.sub(r"\s+", " ", text).strip(" .")
    if not text:
        return fallback
    if len(text) > max_len:
        text = text[:max_len].rstrip(" .")
    return text or fallback


@dataclass(frozen=True, slots=True)
class OsuExportNames:
    """osu! set layout: ``Artist - Title.osz`` + difficulty ``.osu``; audio keeps its own name."""

    osu: str
    osz: str
    audio: str


def sanitize_audio_filename(raw: str, *, default_suffix: str = ".mp3") -> str:
    """Keep the uploaded song filename; only strip illegal path characters."""
    name = Path(raw).name
    suffix = name[name.rfind(".") :].lower() if "." in name else ""
    if suffix not in {".mp3", ".wav"}:
        suffix = default_suffix if default_suffix.startswith(".") else f".{default_suffix}"
        suffix = suffix.lower()
    stem = sanitize_osu_filename_part(Path(name).stem, "audio")
    return f"{stem}{suffix}"


def osu_export_names(
    *,
    artist: str,
    title: str,
    creator: str,
    version: str,
    audio_filename: str,
) -> OsuExportNames:
    """Difficulty/set names follow the editor; audio is the sanitized upload name."""
    artist_s = sanitize_osu_filename_part(artist, "Unknown Artist")
    title_s = sanitize_osu_filename_part(title, "Untitled")
    creator_s = sanitize_osu_filename_part(creator, "Audio2Map")
    version_s = sanitize_osu_filename_part(version, "Generated")
    set_stem = f"{artist_s} - {title_s}"
    osu = f"{set_stem} ({creator_s}) [{version_s}].osu"
    osz = f"{set_stem}.osz"
    audio = sanitize_audio_filename(audio_filename)
    if len(osu) > _MAX_FILENAME:
        budget = _MAX_FILENAME - len(f" ({creator_s}) [{version_s}].osu")
        set_stem = set_stem[: max(12, budget)].rstrip(" .-")
        osu = f"{set_stem} ({creator_s}) [{version_s}].osu"
        osz = f"{set_stem}.osz"
    return OsuExportNames(osu=osu, osz=osz, audio=audio)


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
        return f"{x},{MANIA_Y},{note.time_ms},{HIT_CIRCLE},0,{HIT_SAMPLE}"
    assert note.end_time_ms is not None
    return f"{x},{MANIA_Y},{note.time_ms},{HIT_HOLD},0,{note.end_time_ms}:{HIT_SAMPLE}"


def write_osu(
    path: Path,
    notes: list[ManiaNote],
    *,
    source_osu: Path | None = None,
    title: str = "Untitled",
    artist: str = "Unknown Artist",
    creator: str = "Audio2Map",
    version: str = "Generated",
    offset_ms: int = 0,
    original_bpm: float | None = None,
    audio_filename: str = "audio.mp3",
    version_suffix: str = "",
) -> Path:
    """Write one mania 4K ``.osu``.

    ``source_osu`` supplies timing and metadata (HitObjects replaced, Mode/CS
    forced to mania 4K). Omit it to build from ``original_bpm`` / ``offset_ms``.
    """
    if source_osu is None and original_bpm is None:
        raise ValueError("write_osu needs source_osu or original_bpm")
    hit_lines = [
        note_to_hit_object(n)
        for n in sorted(filter_notes_for_export(notes), key=lambda n: (n.time_ms, n.col))
    ]
    if source_osu is not None:
        body = _lines_from_source(source_osu, hit_lines, version_suffix=version_suffix)
    else:
        assert original_bpm is not None
        body = _lines_from_scratch(
            hit_lines,
            title=title,
            artist=artist,
            creator=creator,
            version=f"{version}{version_suffix}",
            offset_ms=offset_ms,
            original_bpm=original_bpm,
            audio_filename=audio_filename,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def _hit_object_lines(hit_lines: list[str]) -> list[str]:
    return ["[HitObjects]", *hit_lines]


def _sanitize_export_line(line: str, *, in_events: bool) -> str | None:
    if line.startswith("BeatmapID:"):
        return "BeatmapID:0"
    if not in_events:
        return line
    if line.startswith("Video,"):
        return None
    if line.startswith("0,0,") and any(ext in line.lower() for ext in (".jpg", ".jpeg", ".png", ".bmp")):
        return None
    return line


def _lines_from_source(
    source_osu: Path,
    hit_lines: list[str],
    *,
    version_suffix: str,
) -> list[str]:
    lines = Path(source_osu).read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    in_hit = False
    in_events = False
    replaced_version = False
    wrote_hits = False
    for line in lines:
        if line.startswith("[Events]"):
            in_events = True
            out.append(line)
            continue
        if in_events and line.startswith("["):
            in_events = False
        if line.startswith("Mode:"):
            out.append("Mode:3")
            continue
        if line.startswith("CircleSize:"):
            out.append("CircleSize:4")
            continue
        if line.startswith("[HitObjects]"):
            in_hit = True
            wrote_hits = True
            out.extend(_hit_object_lines(hit_lines))
            continue
        if in_hit:
            if line.startswith("["):
                in_hit = False
            else:
                continue
        if not replaced_version and line.startswith("Version:"):
            base = line.split(":", 1)[1].strip()
            out.append(f"Version:{base}{version_suffix}")
            replaced_version = True
            continue
        sanitized = _sanitize_export_line(line, in_events=in_events)
        if sanitized is not None:
            out.append(sanitized)
    if not wrote_hits:
        out.append("")
        out.extend(_hit_object_lines(hit_lines))
    return out


def _lines_from_scratch(
    hit_lines: list[str],
    *,
    title: str,
    artist: str,
    creator: str,
    version: str,
    offset_ms: int,
    original_bpm: float,
    audio_filename: str,
) -> list[str]:
    beat_length = 60_000.0 / original_bpm
    return [
        "osu file format v14",
        "",
        "[General]",
        f"AudioFilename: {audio_filename}",
        "AudioLeadIn: 0",
        "PreviewTime: -1",
        "Countdown: 0",
        "SampleSet: Soft",
        "StackLeniency: 0.7",
        "Mode: 3",
        "LetterboxInBreaks: 0",
        "SpecialStyle: 0",
        "WidescreenStoryboard: 0",
        "",
        "[Editor]",
        "DistanceSpacing: 1",
        "BeatDivisor: 4",
        "GridSize: 4",
        "TimelineZoom: 1",
        "",
        "[Metadata]",
        f"Title:{title}",
        f"TitleUnicode:{title}",
        f"Artist:{artist}",
        f"ArtistUnicode:{artist}",
        f"Creator:{creator}",
        f"Version:{version}",
        "Source:",
        "Tags:",
        "BeatmapID:0",
        "BeatmapSetID:-1",
        "",
        "[Difficulty]",
        "HPDrainRate:8",
        "CircleSize:4",
        "OverallDifficulty:8",
        "ApproachRate:5",
        "SliderMultiplier:1.4",
        "SliderTickRate:1",
        "",
        "[Events]",
        "//Background and Video events",
        "",
        "[TimingPoints]",
        f"{int(offset_ms)},{beat_length:.6f},4,2,0,100,1,0",
        "",
        *_hit_object_lines(hit_lines),
    ]
