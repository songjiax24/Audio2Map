"""End-to-end chart generation for CLI and demo."""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from audio2map.features.cond import canonical_bpm_norm_from_bpm
from audio2map.generate.cond_select import (
    CondCandidate,
    CondSelectionError,
    USER_COND_SOURCE_FIELDS,
    build_final_cond_vec,
    load_cond_candidates,
    select_candidate_from_ranges,
    select_condition,
)
from audio2map.generate.overlap import (
    DecodeConfig,
    GenerationRangeConfig,
    GenerationReport,
    OverlapConfig,
    generate_chart_notes,
)
from audio2map.grid import CanonicalTiming, canonicalize_bpm
from audio2map.model.model import AudioChartModel
from audio2map.osu.export import write_osu
from audio2map.osu.parser import chart_audio_path, parse_timing_points, read_osu_sections
from audio2map.osu.schema import ManiaNote

AUDIO_SUFFIXES = {".mp3", ".ogg", ".wav", ".flac", ".m4a", ".opus"}

__all__ = [
    "CondCandidate",
    "CondSelectionError",
    "DecodeConfig",
    "GenerationResult",
    "USER_COND_SOURCE_FIELDS",
    "build_final_cond_vec",
    "canonical_bpm_norm_from_bpm",
    "generate_chart",
    "load_cond_candidates",
    "pack_osz",
    "select_candidate_from_ranges",
    "select_condition",
    "timing_from_bpm_offset",
    "timing_from_osu",
]


def timing_from_osu(path: Path) -> CanonicalTiming:
    """Canonical timing from any ``.osu`` (e.g. a std map donating TimingPoints)."""
    tps = parse_timing_points(read_osu_sections(path).get("[TimingPoints]", []))
    if not tps:
        raise ValueError(f"no timing points in {path}")
    return CanonicalTiming.from_timing_points(tps)


def timing_from_bpm_offset(*, bpm: float, offset_ms: float) -> CanonicalTiming:
    """Canonical timing from explicit BPM/offset."""
    canonical_bpm, scale_exp = canonicalize_bpm(bpm)
    return CanonicalTiming(
        offset_ms=int(round(offset_ms)),
        original_bpm=float(bpm),
        canonical_bpm=float(canonical_bpm),
        bpm_scale_exp=scale_exp,
    )


def resolve_audio_file(audio: Path) -> Path:
    audio = Path(audio)
    if audio.suffix.lower() in AUDIO_SUFFIXES:
        if not audio.is_file():
            raise FileNotFoundError(f"audio not found: {audio}")
        return audio
    if audio.suffix.lower() == ".osu":
        return chart_audio_path(audio)
    raise FileNotFoundError(f"expected an audio file or .osu, got {audio}")


def pack_osz(
    osu_path: Path,
    audio_path: Path,
    out_osz: Path,
    *,
    extra_files: list[Path] | None = None,
) -> None:
    out_osz.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_osz, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(osu_path, osu_path.name)
        zf.write(audio_path, audio_path.name)
        for extra in extra_files or []:
            if extra.is_file():
                zf.write(extra, extra.name)


@dataclass(slots=True)
class GenerationResult:
    notes: list[ManiaNote]
    report: GenerationReport
    osu_path: Path | None = None
    osz_path: Path | None = None


def generate_chart(
    model: AudioChartModel,
    *,
    audio: Path,
    timing: CanonicalTiming,
    cond_vec: np.ndarray,
    source_osu: Path | None = None,
    reference_notes: list[ManiaNote] | None = None,
    grid_dir: Path | None = None,
    device: torch.device | None = None,
    overlap: OverlapConfig | None = None,
    range_cfg: GenerationRangeConfig | None = None,
    decode: DecodeConfig = DecodeConfig(),
    single_window: bool = False,
    single_window_start_bar: int | None = None,
    build_grid_if_missing: bool = True,
    out_osu: Path | None = None,
    export_version_suffix: str = " (Audio2Map)",
    export_title: str = "Untitled",
    export_artist: str = "Unknown Artist",
    export_creator: str = "Audio2Map",
    export_version: str = "Generated",
    export_audio_filename: str | None = None,
    osz_audio: Path | None = None,
    out_osz: Path | None = None,
    osz_extra_files: list[Path] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> GenerationResult:
    """Generate notes, then optionally ``write_osu`` / pack ``.osz``.

    ``audio`` is an audio file or an ``.osu`` (audio is ``AudioFilename`` beside it).
    ``source_osu`` is only the export template. ``reference_notes`` optionally
    supply chart bar bounds for reports / ``reference_chart`` range.
    """
    audio_file = resolve_audio_file(audio)
    notes, report = generate_chart_notes(
        model,
        audio_path=audio_file,
        timing=timing,
        cond_vec=cond_vec,
        grid_dir=grid_dir,
        overlap=overlap,
        range_cfg=range_cfg,
        device=device,
        decode=decode,
        single_window=single_window,
        single_window_start_bar=single_window_start_bar,
        reference_notes=reference_notes,
        build_grid_if_missing=build_grid_if_missing,
        on_progress=on_progress,
    )

    osu_path: Path | None = None
    if out_osu is not None:
        audio_name = export_audio_filename or audio_file.name
        if source_osu is not None:
            osu_path = write_osu(
                out_osu,
                notes,
                source_osu=source_osu,
                version_suffix=export_version_suffix,
            )
        else:
            osu_path = write_osu(
                out_osu,
                notes,
                title=export_title,
                artist=export_artist,
                creator=export_creator,
                version=export_version,
                offset_ms=timing.offset_ms,
                original_bpm=timing.original_bpm,
                audio_filename=audio_name,
                version_suffix=export_version_suffix,
            )

    osz_path: Path | None = None
    if out_osz is not None and osz_audio is not None:
        if osu_path is None:
            raise ValueError("pack .osz needs out_osu")
        pack_osz(osu_path, osz_audio, out_osz, extra_files=osz_extra_files)
        osz_path = out_osz

    return GenerationResult(notes=notes, report=report, osu_path=osu_path, osz_path=osz_path)
