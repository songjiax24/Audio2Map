"""Controlled audio-alignment and isolated-condition capability experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.metadata import version
import json
import logging
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from audio2map.dataset.v1.example import build_v1_example, load_chart_audio
from audio2map.eval.v1.archive import load_archive
from audio2map.eval.v1.conditions import ScenarioCondition
from audio2map.features.audio.v1.decode import ffmpeg_version_line
from audio2map.eval.v1.inputs import file_digest, freeze_json
from audio2map.eval.v1.journal import Journal, generated_tokens, read_records, record_key
from audio2map.eval.v1.protocol import (
    AUDIO_OFFSETS_S, CONTROL_LEVELS, MAX_SHIFT_FRACTION, load_protocol,
)
from audio2map.eval.v1.reporting.html import write_probe_html
from audio2map.eval.v1.reporting.plots import require_plotting
from audio2map.eval.v1.run import EvalConfig, _device, _generate, _json
from audio2map.eval.v1.statistics import (
    chart_values, describe, mean_interval, paired_difference, regression_slope,
)
from audio2map.eval.v1.summary import summarize
from audio2map.eval.v1.teacher import summarize_teacher, teacher_metrics
from audio2map.features.audio.v1.constants import SAMPLE_RATE
from audio2map.features.cond.chart import compute_chart_meta
from audio2map.features.cond.v1.store import load_v1_chart_meta
from audio2map.features.cond.v1.vec import COND_SOURCE_FIELDS, build_cond
from audio2map.generate.v1.chart import load_v1_model
from audio2map.tokens.v1.codec import decode_tokens, encode_chart

logger = logging.getLogger(__name__)
PROBE_VERSION = 1
CUSTOM_PROBE_VERSION = 2
REUSE_SETTINGS = (
    "source_run_sha256", "checkpoint_sha256", "temperature", "precision", "torch", "gpu",
    "max_shift_fraction", "ffmpeg", "soxr",
)


@dataclass(frozen=True)
class ProbeConfig:
    source: Path
    out: Path
    charts: int = 4
    teacher_charts: int = 16
    seed: int = 0
    seeds: tuple[int, ...] = (0, 1)
    device: str = "cuda"
    precision: str = "bf16"
    resume: bool = False
    html: bool = False
    protocol: Path | None = None
    reuse_runs: tuple[Path, ...] = ()


def shift_audio(audio, offset_s):
    """Circular PCM shift preserves duration and spectrum; never reuse the unshifted mel cache."""
    if not math.isfinite(offset_s):
        raise ValueError("audio shift must be finite")
    amount = min(round(abs(offset_s) * SAMPLE_RATE), int(len(audio.pcm) * MAX_SHIFT_FRACTION))
    amount = amount if offset_s >= 0 else -amount
    if offset_s and not amount:
        raise ValueError("audio is too short for an alignment probe")
    return replace(audio, pcm=np.roll(audio.pcm, amount), audio_key=None), amount


def isolated_condition(meta, field=None, value=None):
    values = dict.fromkeys(COND_SOURCE_FIELDS)
    if field is not None:
        if field not in values or value is None or not math.isfinite(value):
            raise ValueError("unknown or non-finite condition request")
        values[field] = float(value)
    return replace(meta, **values)


def _base(relpath, scenario, meta, **extra):
    mask = tuple(int(value) for value in build_cond(meta).m_real)
    return {
        "relpath": relpath, "scenario": scenario,
        "condition": ScenarioCondition(meta, None, mask).as_dict(), **extra,
    }


def _control_summary(rows, field, requested_levels=None):
    requested_levels = CONTROL_LEVELS[field] if requested_levels is None else requested_levels
    group = [row for row in rows if row.get("field") == field]
    levels = {}
    by_chart = {}
    for level in requested_levels:
        selected = [row for row in group if row["requested"] == level]
        values = chart_values(selected, lambda row: row["generated_meta"].get(field))
        levels[str(level)] = {
            "samples": len(selected),
            "sample_distribution": describe([row["generated_meta"].get(field) for row in selected]),
            "chart_mean": mean_interval(list(values.values())),
        }
        for path, value in values.items():
            by_chart.setdefault(path, {})[str(level)] = value
    responses = {}
    for path, values in by_chart.items():
        complete = all(str(level) in values for level in requested_levels)
        x = list(requested_levels)
        y = [values[str(level)] for level in x] if complete else []
        responses[path] = {
            "values": values, "complete": complete,
            "slope": regression_slope(x, y, center=True) if complete else None,
            "nondecreasing": all(b >= a for a, b in zip(y, y[1:])) if complete else None,
            "high_minus_low": y[-1] - y[0] if complete else None,
        }
    return {"levels": levels, "charts": responses}


def probe_summary(rows, *, offsets=None, controls=None):
    offsets = AUDIO_OFFSETS_S if offsets is None else offsets
    controls = CONTROL_LEVELS if controls is None else controls
    teachers, generations = {}, {}
    for condition in ("matched_full", "none"):
        group = [row for row in rows if row["mode"] == "probe_teacher"
                 and row["condition_name"] == condition]
        if not group:
            continue
        baseline = chart_values([r for r in group if r["offset_s"] == 0],
                                lambda row: row["metrics"]["loss_token"])
        teachers[condition] = {}
        for offset in offsets:
            selected = [row for row in group if row["offset_s"] == offset]
            values = chart_values(selected, lambda row: row["metrics"]["loss_token"])
            teachers[condition][str(offset)] = {
                **summarize_teacher([row["metrics"] for row in selected]),
                "nll_shift_minus_correct": paired_difference(baseline, values),
            }
    gen = [row for row in rows if row["mode"] == "constrained"]
    audio = [row for row in gen if row["kind"] == "audio_shift"]
    baseline = chart_values([r for r in audio if r["offset_s"] == 0],
                            lambda r: r["match"]["event"]["lane_agnostic"]["all"]["f1"])
    for offset in offsets:
        selected = [r for r in audio if r["offset_s"] == offset]
        values = chart_values(selected,
                              lambda r: r["match"]["event"]["lane_agnostic"]["all"]["f1"])
        generations[str(offset)] = {
            "charts": len(values), "f1": describe(list(values.values())),
            "f1_shift_minus_correct": paired_difference(baseline, values),
            "notes": describe([row["notes"] for row in selected]),
            "tokens": describe([row["tokens"] for row in selected]),
            "empty_samples": sum(row["notes"] == 0 for row in selected),
            "meta_error_samples": sum(bool(row["generated_meta"].get("error"))
                                      for row in selected),
        }
    return {
        "audio_teacher": teachers, "audio_generation": generations,
        "actual_offsets_s": {
            str(offset): describe([
                row["shift_samples"] / SAMPLE_RATE for row in rows
                if row.get("offset_s") == offset and "shift_samples" in row
            ]) for offset in offsets
        },
        "control": {field: _control_summary(gen, field, levels)
                    for field, levels in controls.items()},
        "generated": summarize(gen)["constrained"],
    }


class ProbeReuse:
    """Copy only verified compatible cases; the destination journal remains complete."""

    def __init__(self, paths, plan):
        self.rows = {}
        self.used = 0
        self.provenance = []
        for path in paths:
            if not (path / "summary.json").is_file():
                raise ValueError(f"reuse requires a complete probe run: {path}")
            original = json.loads((path / "run.json").read_text())
            for setting in REUSE_SETTINGS:
                if original.get(setting) != plan.get(setting):
                    raise ValueError(f"probe reuse settings differ: {setting}")
            records, committed = read_records(path / "records.jsonl")
            if committed != (path / "records.jsonl").stat().st_size:
                raise ValueError("reuse journal is incomplete")
            for key, row in records.items():
                self.rows[key] = (path, row)
            self.provenance.append({
                "path": str(path.resolve()), "run_sha256": file_digest(path / "run.json"),
                "records_sha256": file_digest(path / "records.jsonl"),
            })

    def get(self, journal, base, mode, seed=None):
        cached = journal.get(base, mode, seed)
        if cached is not None:
            return cached
        candidate = self.rows.get(record_key({**base, "mode": mode, "seed": seed}))
        if candidate is None:
            return None
        path, row = candidate
        if any(row.get(key) != value for key, value in base.items()):
            return None
        if mode == "constrained" and generated_tokens(row) is None:
            return None
        self.used += 1
        return {**row, "reused_from": str(path.resolve())}


def run_probes(cfg: ProbeConfig) -> dict:
    if cfg.html:
        require_plotting()
    if cfg.source.resolve() == cfg.out.resolve():
        raise ValueError("probes require a separate output directory")
    if cfg.charts < 1 or cfg.teacher_charts < 0 or not cfg.seeds:
        raise ValueError("invalid probe chart/seed counts")
    if len(set(cfg.seeds)) != len(cfg.seeds) or cfg.precision not in ("bf16", "fp32"):
        raise ValueError("seeds must be unique and precision must be bf16/fp32")
    started = time.perf_counter()
    protocol = load_protocol(cfg.protocol)
    archive = load_archive(cfg.source)
    device = _device(cfg.device)
    if device.type == "cuda" and cfg.precision != "bf16":
        raise ValueError("production CUDA generation requires bf16")
    order = random.Random(cfg.seed).sample(archive.run["charts"], len(archive.run["charts"]))
    generation_paths = sorted(order[:cfg.charts])
    teacher_paths = sorted(order[:cfg.teacher_charts])
    paths = sorted(set(generation_paths) | set(teacher_paths))
    stored = load_v1_chart_meta(cfg.source / "metadata.jsonl")
    metadata = {path: stored.get(path) or compute_chart_meta(archive.root / path) for path in paths}
    plan = {
        "probe_version": PROBE_VERSION, "source": str(cfg.source.resolve()),
        "source_run_sha256": file_digest(cfg.source / "run.json"),
        "checkpoint": archive.run["checkpoint"],
        "checkpoint_sha256": archive.run["checkpoint_sha256"],
        "teacher_charts": teacher_paths, "generation_charts": generation_paths,
        "metadata": {path: meta.to_dict() for path, meta in metadata.items()},
        "offsets_s": list(protocol.offsets_s), "max_shift_fraction": MAX_SHIFT_FRACTION,
        "controls": {key: list(value) for key, value in protocol.controls.items()},
        "seed": cfg.seed, "seeds": list(cfg.seeds), "temperature": archive.run["temperature"],
        "device": str(device), "precision": cfg.precision, "torch": str(torch.__version__),
        "ffmpeg": ffmpeg_version_line(), "soxr": version("soxr"),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "html": cfg.html,
    }
    if cfg.protocol is not None:
        plan["protocol"] = protocol.provenance
    reuse = ProbeReuse(cfg.reuse_runs, plan)
    if cfg.protocol is not None or cfg.reuse_runs:
        plan["probe_version"] = CUSTOM_PROBE_VERSION
        plan["reuse_runs"] = reuse.provenance
    cfg.out.mkdir(parents=True, exist_ok=True)
    freeze_json(cfg.out / "run.json", plan)
    (cfg.out / "summary.json").unlink(missing_ok=True)
    (cfg.out / "index.html").unlink(missing_ok=True)
    model = load_v1_model(archive.checkpoint(), device)
    generation_cfg = EvalConfig(
        Path(archive.run["checkpoint"]), Path(archive.run["split_manifest"]), archive.run["split"],
        archive.root, cfg.out, temperature=archive.run["temperature"],
        keep_frames=archive.run["keep_frames"],
    )
    with Journal(cfg.out / "records.jsonl", resume=cfg.resume) as journal:
        for index, relpath in enumerate(paths, 1):
            audio = load_chart_audio(archive.root / relpath, use_cache=False)
            encoded = encode_chart(list(audio.notes), audio.redlines, audio.t_audio_ms)
            reference = decode_tokens(encoded.tokens, encoded.grid)
            meta = metadata[relpath]
            for offset in protocol.offsets_s:
                shifted, samples = shift_audio(audio, offset)
                if relpath in teacher_paths:
                    batch = None
                    for condition in ("matched_full", "none"):
                        target = meta if condition == "matched_full" else isolated_condition(meta)
                        base = _base(
                            relpath, f"audio/{condition}/{offset}", target,
                            condition_name=condition, offset_s=offset, shift_samples=samples,
                        )
                        cached = reuse.get(journal, base, "probe_teacher")
                        if cached is None:
                            if batch is None:
                                batch = build_v1_example(shifted, meta=meta, encoded=encoded,
                                                         record_legal=True)
                            cached = {
                                **base, "mode": "probe_teacher",
                                "metrics": teacher_metrics(
                                    model, batch, target, device=device,
                                    use_amp=device.type == "cuda" and cfg.precision == "bf16",
                                ),
                            }
                        journal.add(cached)
                if relpath in generation_paths:
                    base = _base(relpath, f"audio/matched_full/{offset}", meta, kind="audio_shift",
                                 offset_s=offset, shift_samples=samples)
                    row = reuse.get(journal, base, "constrained", cfg.seeds[0])
                    if generated_tokens(row) is None:
                        row, _, _ = _generate(
                            generation_cfg, model, shifted, encoded.grid, reference, meta,
                            base=base, seed=cfg.seeds[0], device=device,
                        )
                    journal.add(row)
            if relpath in generation_paths:
                for field, levels in protocol.controls.items():
                    for value in levels:
                        target = isolated_condition(meta, field, value)
                        base = _base(relpath, f"control/{field}/{value}", target,
                                     kind="condition_control", field=field, requested=value)
                        for seed in cfg.seeds:
                            row = reuse.get(journal, base, "constrained", seed)
                            if generated_tokens(row) is None:
                                row, _, _ = _generate(
                                    generation_cfg, model, audio, encoded.grid, reference, target,
                                    base=base, seed=seed, device=device,
                                )
                            journal.add(row)
            logger.info("v1 probes %s/%s %s", index, len(paths), relpath)
    report = {
        "run": plan, **probe_summary(journal.rows, offsets=protocol.offsets_s,
                                    controls=protocol.controls),
        "elapsed_s": time.perf_counter()-started,
        "reused_records": journal.reused + reuse.used, "external_reused_records": reuse.used,
    }
    if cfg.html:
        write_probe_html(report, cfg.out, rows=journal.rows)
    _json(cfg.out / "summary.json", report)
    return report
