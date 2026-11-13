"""Verified completed runs used by rescoring and controlled capability probes."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path

from audio2map.eval.v1.conditions import scenario_cases
from audio2map.eval.v1.inputs import file_digest
from audio2map.eval.v1.journal import read_records
from audio2map.features.audio.v1.constants import MEL_NORM_VERSION
from audio2map.features.audio.v1.decode import ffmpeg_version_line


AUDIO_SETTINGS = ("audio_source", "ffmpeg", "soxr", "mel_norm")


def direct_audio_settings() -> dict:
    return {
        "audio_source": "direct_decode",
        "ffmpeg": ffmpeg_version_line(),
        "soxr": version("soxr"),
        "mel_norm": MEL_NORM_VERSION,
    }


def record_audio_settings(run: dict, mode: str) -> dict | None:
    if mode == "teacher":
        if "teacher_audio" in run:
            return run["teacher_audio"]
        if (run.get("rescore") or {}).get("refresh_teacher"):
            return None
    return {key: run.get(key) for key in AUDIO_SETTINGS}


def effective_precision(run: dict, mode: str) -> str | None:
    """Recover the precision of the records, including older rescored runs."""
    teacher = mode == "teacher"
    refresh = run.get("rescore") or {}
    if teacher and refresh.get("refresh_teacher"):
        settings = refresh
    else:
        key = "teacher_precision" if teacher else "generation_precision"
        if run.get(key) is not None:
            return run[key]
        settings = run
    device = settings.get("device")
    precision = settings.get("precision")
    if not device or precision not in ("bf16", "fp32"):
        return None
    cuda = device.split(":")[0] == "cuda"
    return "bf16" if cuda and (not teacher or precision == "bf16") else "fp32"


@dataclass(frozen=True)
class RunArchive:
    source: Path
    run: dict
    report: dict
    rows: list[dict]

    @property
    def root(self) -> Path:
        return Path(self.run["raw_root"])

    def checkpoint(self) -> Path:
        path = Path(self.run["checkpoint"])
        if file_digest(path) != self.run["checkpoint_sha256"]:
            raise ValueError("checkpoint differs from the source run")
        return path


def load_archive(source: Path) -> RunArchive:
    if not (source / "summary.json").is_file():
        raise ValueError("a completed source run is required")
    run = json.loads((source / "run.json").read_text())
    report = json.loads((source / "summary.json").read_text())
    if report.get("run") != run or report["pool"]["accepted"] != run["charts"]:
        raise ValueError("source summary does not match the frozen run")
    root = Path(run["raw_root"])
    for key in ("chart_sha256", "audio_sha256"):
        for relpath, digest in run.get(key, {}).items():
            if file_digest(root / relpath) != digest:
                raise ValueError(f"source input changed: {relpath}")
    records, committed = read_records(source / "records.jsonl")
    if committed != (source / "records.jsonl").stat().st_size:
        raise ValueError("source journal has an incomplete final record")
    expected = {
        (chart, scenario, mode, seed)
        for chart in run["charts"]
        for scenario in run["scenarios"]
        for mode, seed in scenario_cases(run["mode"], scenario, run["seeds"])
    }
    missing, unexpected = expected - records.keys(), records.keys() - expected
    if not expected or missing or unexpected:
        raise ValueError(
            f"source journal does not match the frozen cases "
            f"({len(missing)} missing, {len(unexpected)} unexpected)"
        )
    conditions = json.loads((source / "conditions.json").read_text())
    for (chart, scenario, mode, seed), row in records.items():
        condition = conditions.get(chart, {}).get(scenario)
        if condition is None or row.get("condition") != condition:
            raise ValueError(f"source journal condition differs: {chart}/{scenario}/{mode}/{seed}")
    return RunArchive(source, run, report, list(records.values()))
