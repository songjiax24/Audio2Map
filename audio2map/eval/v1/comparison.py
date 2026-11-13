"""Paired checkpoint comparisons under exactly matched frozen evaluation inputs."""

from __future__ import annotations

import json
from pathlib import Path

from audio2map.eval.v1.archive import (
    AUDIO_SETTINGS, effective_precision, load_archive, record_audio_settings,
)
from audio2map.eval.v1.inputs import file_digest, freeze_json
from audio2map.eval.v1.journal import record_key
from audio2map.eval.v1.statistics import chart_values, paired_difference

MATCHED_SETTINGS = (
    "charts", "chart_sha256", "audio_sha256", "split_sha256", "seed", "seeds", "scenarios",
    "temperature", "keep_frames", "motif_vocab_sha256", "tokenizer", "cond_vec_version",
)


def compare_runs(left: Path, right: Path, out: Path) -> dict:
    if out.resolve() in (left.resolve(), right.resolve()):
        raise ValueError("comparison needs a separate output directory")
    archives = [load_archive(path) for path in (left, right)]
    left_run, right_run = (archive.run for archive in archives)
    for key in MATCHED_SETTINGS:
        if left_run.get(key) != right_run.get(key):
            raise ValueError(f"comparison inputs differ: {key}")
    if min(archive.run["report_version"] for archive in archives) < 2:
        raise ValueError("rescore both runs to report version 2 first")
    if json.loads((left / "conditions.json").read_text()) != json.loads(
        (right / "conditions.json").read_text()
    ):
        raise ValueError("comparison conditions or masks differ")
    provenance = {
        "left": str(left.resolve()), "right": str(right.resolve()),
        "left_summary_sha256": file_digest(left / "summary.json"),
        "right_summary_sha256": file_digest(right / "summary.json"),
        "left_records_sha256": file_digest(left / "records.jsonl"),
        "right_records_sha256": file_digest(right / "records.jsonl"),
        "difference": "right minus left; seeds averaged within chart before chart bootstrap",
    }
    rows = []
    for scenario in archives[0].run["scenarios"]:
        for mode in ("teacher", "constrained", "unconstrained"):
            groups = [[r for r in archive.rows if r["mode"] == mode and r["scenario"] == scenario]
                      for archive in archives]
            if not all(groups):
                continue
            precisions = [effective_precision(archive.run, mode) for archive in archives]
            if None in precisions or precisions[0] != precisions[1]:
                raise ValueError(f"comparison effective precision differs or is missing: {mode}")
            left_audio, right_audio = [
                record_audio_settings(archive.run, mode) for archive in archives
            ]
            if left_audio is None or right_audio is None:
                raise ValueError(f"comparison audio provenance is missing: {mode}; refresh teacher")
            for key in AUDIO_SETTINGS:
                if left_audio.get(key) != right_audio.get(key):
                    raise ValueError(f"comparison audio provenance differs: {mode}/{key}")
            if {record_key(row) for row in groups[0]} != {record_key(row) for row in groups[1]}:
                raise ValueError(f"comparison cases differ: {scenario}/{mode}")
            if mode == "teacher":
                metrics = {
                    "NLL/token": lambda r: r["metrics"]["loss_token"],
                    "ROW accuracy": lambda r: r["metrics"]["row"],
                }
            elif mode == "unconstrained":
                metrics = {"valid": lambda r: float(r["validity"] == "valid")}
            else:
                metrics = {
                    **{
                        f"{unit}/{scope} F1":
                        (lambda r, u=unit, s=scope: r["match"][u][s]["all"]["f1"])
                        for unit in ("event", "note")
                        for scope in ("lane_aware", "lane_agnostic")
                    },
                    "SR absolute error": lambda r: r["condition_error"]["official_sr"],
                    "MSD absolute error": lambda r: r["condition_error"]["msd_overall"],
                    "samples with concurrent": lambda r: float(r["plausibility"]["has_concurrent"]),
                }
            for metric, value in metrics.items():
                values = [chart_values(group, value) for group in groups]
                rows.append({"scenario": scenario, "mode": mode, "metric": metric,
                             **paired_difference(*values)})
    if not rows:
        raise ValueError("the runs have no common evaluation cases")
    out.mkdir(parents=True, exist_ok=True)
    freeze_json(out / "run.json", provenance)
    (out / "comparison.json").write_text(json.dumps({**provenance, "metrics": rows}, indent=2))
    lines = [
        "# Paired capability comparison", "", provenance["difference"], "",
        "Confidence intervals describe chart resampling of these fixed generated samples. "
        "F1 is positive evidence of reference reproduction; no total chart-quality score is used.",
        "", "| Scenario | Mode | Metric | Charts | Right − left | 95% low | 95% high |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        def value(key):
            return "—" if row[key] is None else f"{row[key]:.5g}"
        lines.append(f"| {row['scenario']} | {row['mode']} | {row['metric']} | {row['charts']} | "
                     f"{value('mean')} | {value('low')} | {value('high')} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    return {**provenance, "metrics": rows}
