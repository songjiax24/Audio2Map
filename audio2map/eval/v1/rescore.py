"""Re-evaluate saved v1 token artifacts without resampling or altering the source run."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time

import torch

from audio2map.dataset.v1.example import build_v1_example, load_chart_audio
from audio2map.dataset.v1.gate import cached_audio_durations
from audio2map.eval.v1.archive import (
    direct_audio_settings, effective_precision, load_archive, record_audio_settings,
)
from audio2map.eval.v1.inputs import file_digest, freeze_json, verify_training_corpus
from audio2map.eval.v1.journal import Journal, generated_tokens
from audio2map.eval.v1.metrics import endpoint_match, event_distribution, plausibility
from audio2map.eval.v1.motifs import load_vocabulary
from audio2map.eval.v1.reporting.html import write_html
from audio2map.eval.v1.reporting.markdown import markdown
from audio2map.eval.v1.reporting.plots import require_plotting
from audio2map.eval.v1.run import REPORT_VERSION, _device, _json
from audio2map.eval.v1.summary import EvaluationCollection
from audio2map.eval.v1.teacher import teacher_metrics
from audio2map.features.cond.chart import ChartMeta, compute_chart_meta
from audio2map.features.cond.v1.store import load_v1_chart_meta
from audio2map.generate.v1.chart import load_v1_model
from audio2map.generate.v1.notes import notes_from_events
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.v1.timing import parse_redlines_file
from audio2map.tokens.v1.codec import decode_tokens, encode_chart

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RescoreConfig:
    source: Path
    out: Path
    motif_vocab: Path | None = None
    refresh_teacher: bool = False
    device: str = "auto"
    precision: str = "bf16"
    html: bool = False


def rescore(cfg: RescoreConfig) -> dict:
    started = time.perf_counter()
    if cfg.precision not in ("bf16", "fp32"):
        raise ValueError("unknown teacher precision")
    if cfg.html:
        require_plotting()
    if cfg.source.resolve() == cfg.out.resolve():
        raise ValueError("rescoring requires a separate output directory")
    archive = load_archive(cfg.source)
    source, original_report, root, rows = archive.run, archive.report, archive.root, archive.rows
    has_teacher = any(row["mode"] == "teacher" for row in rows)
    if cfg.refresh_teacher and not has_teacher:
        raise ValueError("teacher refresh requires existing teacher cases")
    vocabulary_path = cfg.motif_vocab or (
        Path(source["motif_vocab"]) if source.get("motif_vocab") else None
    )
    vocabulary = None
    if source.get("motif_vocab_sha256"):
        if vocabulary_path is None:
            raise ValueError("legacy run requires --motif-vocab")
        if file_digest(vocabulary_path) != source["motif_vocab_sha256"]:
            raise ValueError("motif vocabulary differs from the source run")
        fitted = json.loads(vocabulary_path.read_text())
        vocabulary = load_vocabulary(fitted)
        verify_training_corpus(fitted["corpus"], Path(source["split_manifest"]), root)
    cfg.out.mkdir(parents=True, exist_ok=True)
    device = _device(cfg.device) if cfg.refresh_teacher else torch.device("cpu")
    use_amp = device.type == "cuda" and cfg.precision == "bf16"
    run = {
        **source, "report_version": REPORT_VERSION,
        "motif_vocab": str(vocabulary_path.resolve()) if vocabulary_path else None,
        "rescore": {
            "source": str(cfg.source.resolve()),
            "source_run_sha256": file_digest(cfg.source / "run.json"),
            "source_records_sha256": file_digest(cfg.source / "records.jsonl"),
            "refresh_teacher": cfg.refresh_teacher,
            "html": cfg.html,
            "device": str(device), "precision": cfg.precision,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    if cfg.refresh_teacher:
        run["teacher_precision"] = "bf16" if use_amp else "fp32"
        run["teacher_audio"] = direct_audio_settings()
    elif has_teacher:
        run["teacher_precision"] = effective_precision(source, "teacher")
        run["teacher_audio"] = record_audio_settings(source, "teacher")
    freeze_json(cfg.out / "run.json", run)
    for name in ("pool.json", "conditions.json", "donors.json"):
        if (cfg.source / name).is_file():
            freeze_json(cfg.out / name, json.loads((cfg.source / name).read_text()))
    for name in ("summary.json", "summary.md", "index.html"):
        (cfg.out / name).unlink(missing_ok=True)
    model = None
    if cfg.refresh_teacher:
        model = load_v1_model(archive.checkpoint(), device)
    collection = EvaluationCollection(vocabulary, source["scenarios"])
    durations = cached_audio_durations(root, source["charts"], gate_dir=cfg.source / "gate")
    saved_references = {row["relpath"]: row for row in original_report.get("references", ())}
    metadata = load_v1_chart_meta(cfg.source / "metadata.jsonl")
    by_chart = {key: [] for key in source["charts"]}
    for row in rows:
        by_chart[row["relpath"]].append(row)
    with Journal(cfg.out / "records.jsonl", resume=False) as journal:
        for index, (relpath, group) in enumerate(by_chart.items(), 1):
            path = root / relpath
            saved = saved_references.get(relpath, {})
            duration = saved.get("audio_duration_ms", durations.get(relpath))
            audio = (
                load_chart_audio(path, use_cache=False)
                if cfg.refresh_teacher or duration is None else None
            )
            if audio is None:
                notes, redlines = parse_beatmap(path).notes, parse_redlines_file(path).redlines
            else:
                notes, redlines = list(audio.notes), audio.redlines
            if duration is None:
                duration = audio.t_audio_ms
            encoded = encode_chart(notes, redlines, duration)
            reference = decode_tokens(encoded.tokens, encoded.grid)
            saved_meta = saved.get("meta")
            meta = (
                ChartMeta.from_dict(saved_meta) if saved_meta is not None
                else metadata.get(relpath) or compute_chart_meta(path)
            )
            collection.add_reference(relpath, encoded.tokens, reference, encoded.grid, meta)
            batch = None
            if cfg.refresh_teacher:
                teacher_encoded = (
                    encoded if duration == audio.t_audio_ms
                    else encode_chart(notes, redlines, audio.t_audio_ms)
                )
                batch = build_v1_example(
                    audio, meta=meta, encoded=teacher_encoded, record_legal=True,
                )
            samples, sample_tokens = {}, {}
            for old in group:
                row = dict(old)
                if row["mode"] == "teacher" and model is not None:
                    row["metrics"] = teacher_metrics(
                        model, batch, ChartMeta.from_dict(row["condition"]["meta"]),
                        device=device, use_amp=use_amp,
                    )
                if row["mode"] == "constrained":
                    tokens = generated_tokens(row)
                    if tokens is None:
                        raise ValueError(f"generated artifact missing or changed: {row['output']}")
                    events = decode_tokens(tokens, encoded.grid)
                    notes, _warnings = notes_from_events(events)
                    row.update(
                        match=endpoint_match(events, reference),
                        distribution=event_distribution(events, encoded.grid, tokens),
                        plausibility=plausibility(notes),
                    )
                    collection.add_generation(row, tokens, encoded.grid)
                    if row["scenario"] == "matched_full":
                        samples[row["seed"]] = events
                        sample_tokens[row["seed"]] = tokens
                journal.add(row)
            if samples:
                collection.add_diversity(relpath, samples, sample_tokens)
            logger.info("v1 rescore %s/%s %s", index, len(by_chart), relpath)
    report = collection.finish(
        journal.rows, run=run, pool=original_report["pool"],
        elapsed_s=time.perf_counter() - started, reused_records=len(rows),
        peak_allocated_gib=None,
    )
    if cfg.html:
        write_html(report, journal.rows, cfg.out)
    (cfg.out / "summary.md").write_text(markdown(report))
    _json(cfg.out / "summary.json", report)
    return report
