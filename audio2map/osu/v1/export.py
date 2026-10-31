"""Write a v1 chart. Red lines only. Notes are floored and not dropped.

``filter_notes_for_export`` is the v0 path and is not used. A same-column
millisecond, a hold that floors shut, or an unclosed hold is written and
recorded. The file is still produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from audio2map.osu.parser import read_osu_sections
from audio2map.osu.v1.timing import Redline


@dataclass(frozen=True, slots=True)
class ExportAudit:
    path: Path
    warnings: tuple[str, ...]
    hit_lines: int


def write_v1_osu(
    path: Path,
    hit_lines: list[str],
    warnings: list[str],
    redlines: tuple[Redline, ...] | list[Redline],
    source_osu: Path | None = None,
    *,
    version_suffix: str = " Audio2Map",
    audio_filename: str = "audio.wav",
    title: str = "Untitled",
    artist: str = "Unknown Artist",
    creator: str = "Audio2Map",
    version: str = "Audio2Map",
) -> ExportAudit:
    """Write red lines and the floored hit objects. Nothing is dropped.

    Omit ``source_osu`` to write a new mania 4K file. Pass it only to keep
    that file's other sections and append ``version_suffix`` to ``Version``.
    """
    timing = [_timing_line(line) for line in redlines]
    if source_osu is None:
        body = fresh_mania_lines(
            hit_lines,
            timing,
            audio_filename=audio_filename,
            title=title,
            artist=artist,
            creator=creator,
            version=version,
        )
    else:
        body = _rewrite(Path(source_osu), timing, hit_lines, version_suffix)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return ExportAudit(path, tuple(warnings), len(hit_lines))


def fresh_mania_lines(
    hit_lines: list[str],
    timing_lines: list[str],
    *,
    title: str,
    artist: str,
    creator: str,
    version: str,
    audio_filename: str,
) -> list[str]:
    """The one from-scratch mania 4K file. Callers supply timing lines.

    v0's BPM path uses this too. A second shell does not belong in
    ``osu/export.py``.
    """
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
        *timing_lines,
        "",
        "[HitObjects]",
        *hit_lines,
    ]


def _timing_line(line: Redline) -> str:
    offset = format(float(line.offset_ms), ".12g")
    beat = format(float(line.beat_length_ms), ".12g")
    return f"{offset},{beat},{int(line.meter)},2,0,100,1,0"


def _rewrite(
    source: Path,
    timing: list[str],
    hits: list[str],
    version_suffix: str,
) -> list[str]:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    # Touch the reader so a missing file fails the same way as other osu paths.
    read_osu_sections(source)
    out: list[str] = []
    skip: str | None = None
    wrote_timing = False
    wrote_hits = False
    for line in lines:
        if line.startswith("[TimingPoints]"):
            out.append(line)
            out.extend(timing)
            skip = "timing"
            wrote_timing = True
            continue
        if line.startswith("[HitObjects]"):
            out.append(line)
            out.extend(hits)
            skip = "hits"
            wrote_hits = True
            continue
        if skip is not None and line.startswith("["):
            skip = None
        if skip is not None:
            continue
        if line.startswith("Mode:"):
            out.append("Mode:3")
            continue
        if line.startswith("CircleSize:"):
            out.append("CircleSize:4")
            continue
        if line.startswith("OverallDifficulty:"):
            out.append("OverallDifficulty:8")
            continue
        if line.startswith("Version:"):
            base = line.split(":", 1)[1]
            out.append(f"Version:{base}{version_suffix}")
            continue
        if line.startswith("BeatmapID:"):
            out.append("BeatmapID:0")
            continue
        out.append(line)
    if not wrote_timing:
        out.extend(["", "[TimingPoints]", *timing])
    if not wrote_hits:
        out.extend(["", "[HitObjects]", *hits])
    return out
