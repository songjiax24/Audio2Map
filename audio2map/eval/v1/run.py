"""Run reproducible whole-chart v1 evaluation through the production generator."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from audio2map.dataset.v1.example import build_v1_example, load_chart_audio
from audio2map.dataset.v1.gate import cached_audio_durations
from audio2map.eval.v1.archive import direct_audio_settings
from audio2map.eval.v1.conditions import (
    DIVERSITY_SEEDS,
    SCENARIOS,
    scenario_cases,
    scenario_condition,
)
from audio2map.eval.v1.inputs import (
    file_digest,
    freeze_json,
    prepare_metadata,
    prepare_pool,
    select_charts,
    verify_training_corpus,
)
from audio2map.eval.v1.journal import Journal, generated_tokens, write_playback_pcm
from audio2map.eval.v1.metrics import (
    condition_error,
    endpoint_match,
    event_distribution,
    plausibility,
)
from audio2map.eval.v1.motifs import DEFAULT_MOTIF_K, fit_vocabulary, load_vocabulary
from audio2map.eval.v1.reporting.html import write_html
from audio2map.eval.v1.reporting.markdown import markdown
from audio2map.eval.v1.reporting.plots import require_plotting
from audio2map.eval.v1.summary import EvaluationCollection
from audio2map.eval.v1.teacher import teacher_metrics
from audio2map.features.cond.chart import compute_chart_meta
from audio2map.features.cond.v1.vec import COND_VEC_VERSION
from audio2map.generate.v1.chart import (
    generate_v1_chart,
    generate_v1_unconstrained,
    load_v1_model,
)
from audio2map.generate.v1.notes import hit_lines_from_notes
from audio2map.generate.v1.stream import KEEP_FRAMES
from audio2map.osu.parser import chart_audio_path, parse_beatmap
from audio2map.osu.v1.export import write_v1_osu
from audio2map.osu.v1.timing import parse_redlines_file
from audio2map.tokens.v1.codec import decode_tokens, encode_chart
from audio2map.tokens.v1.vocab import TOKENIZER_VERSION

logger = logging.getLogger(__name__)
MODES = ("teacher", "constrained", "unconstrained", "all", "fit-motifs")
REPORT_VERSION = 2


@dataclass(frozen=True, slots=True)
class EvalConfig:
    checkpoint: Path | None
    split_manifest: Path
    split: str
    root: Path
    out: Path
    mode: str = "all"
    scenarios: tuple[str, ...] = SCENARIOS
    seeds: tuple[int, ...] = DIVERSITY_SEEDS
    seed: int = 0
    temperature: float = 1.0
    device: str = "auto"
    precision: str = "bf16"
    limit: int | None = None
    paths: tuple[Path, ...] = ()
    workers: int = 4
    motif_vocab: Path | None = None
    motif_k: int = DEFAULT_MOTIF_K
    keep_frames: int = KEEP_FRAMES
    resume: bool = False
    html: bool = False


def _json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _check_config(cfg: EvalConfig) -> None:
    if not cfg.split_manifest.is_file() or not cfg.root.is_dir():
        raise ValueError("evaluation requires an existing split manifest and raw directory")
    if cfg.motif_vocab is not None and not cfg.motif_vocab.is_file():
        raise ValueError(f"motif vocabulary does not exist: {cfg.motif_vocab}")
    if cfg.mode not in MODES or cfg.precision not in ("bf16", "fp32"):
        raise ValueError("unknown evaluation mode or precision")
    if not cfg.scenarios or not set(cfg.scenarios) <= set(SCENARIOS):
        raise ValueError("scenarios must be a nonempty subset of the five v1 scenarios")
    if len(set(cfg.scenarios)) != len(cfg.scenarios) or not math.isfinite(cfg.temperature):
        raise ValueError("scenarios must be unique and temperature must be finite")
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("sampling seeds must be nonempty and unique")
    if cfg.mode != "fit-motifs" and not any(
        next(scenario_cases(cfg.mode, scenario, cfg.seeds), None) for scenario in cfg.scenarios
    ):
        raise ValueError("the selected mode and scenarios have no evaluation cases")
    if cfg.workers < 1 or cfg.keep_frames < 1 or cfg.motif_k < 1:
        raise ValueError("workers, keep_frames, and motif_k must be positive")
    if cfg.mode == "fit-motifs":
        if cfg.split != "train":
            raise ValueError("fit-motifs requires --split train")
    elif cfg.checkpoint is None or not cfg.checkpoint.is_file():
        raise ValueError("evaluation requires an existing v1 checkpoint")
    if cfg.mode in ("constrained", "all") and cfg.motif_vocab is None:
        raise ValueError("constrained evaluation requires a frozen v1 --motif-vocab")


def evaluate(cfg: EvalConfig) -> dict:
    _check_config(cfg)
    if cfg.html and cfg.mode != "fit-motifs":
        require_plotting()
    device = _device(cfg.device) if cfg.mode != "fit-motifs" else torch.device("cpu")
    if (
        device.type == "cuda"
        and cfg.mode in ("constrained", "unconstrained", "all")
        and cfg.precision != "bf16"
    ):
        raise ValueError("production v1 CUDA generation uses bf16; fp32 is teacher-only")
    started = time.perf_counter()
    requested = select_charts(
        cfg.split_manifest,
        cfg.root,
        cfg.split,
        limit=cfg.limit,
        seed=cfg.seed,
        paths=cfg.paths,
    )
    cfg.out.mkdir(parents=True, exist_ok=True)
    charts, donors, pool = prepare_pool(
        requested,
        cfg.root,
        cfg.out,
        seed=cfg.seed,
        workers=cfg.workers,
    )
    inputs = {
        "split": cfg.split,
        "split_manifest": str(cfg.split_manifest.resolve()),
        "split_sha256": file_digest(cfg.split_manifest),
        "raw_root": str(cfg.root.resolve()),
        "seed": cfg.seed,
        "subset": cfg.limit is not None or bool(cfg.paths),
        "charts": [chart.relpath for chart in charts],
        "chart_sha256": {chart.relpath: file_digest(cfg.root / chart.relpath) for chart in charts},
        "audio_sha256": {
            str(path.relative_to(cfg.root.resolve())): file_digest(path)
            for path in sorted(
                {chart_audio_path(cfg.root / chart.relpath).resolve() for chart in charts}
            )
        },
    }
    if cfg.mode == "fit-motifs":
        return _fit(cfg, charts, inputs)
    vocabulary = None
    motif_data = None
    if cfg.motif_vocab is not None:
        motif_data = json.loads(cfg.motif_vocab.read_text())
        vocabulary = load_vocabulary(motif_data)
        verify_training_corpus(motif_data["corpus"], cfg.split_manifest, cfg.root)
    saved_run = cfg.out / "run.json"
    if saved_run.exists():
        previous = json.loads(saved_run.read_text())
        if any(previous.get(key) != value for key, value in inputs.items()):
            raise ValueError(f"frozen inputs differ: {saved_run}; choose another output directory")
    freeze_json(cfg.out / "source_inputs.json", inputs)
    use_amp = device.type == "cuda" and cfg.precision == "bf16"
    metadata = prepare_metadata(charts, cfg.root, cfg.out, workers=cfg.workers)
    conditions = {
        chart.relpath: {
            scenario: scenario_condition(
                metadata[chart.relpath],
                metadata[donors[chart.relpath]],
                scenario=scenario,
                seed=cfg.seed,
                key=chart.relpath,
                donor_path=donors[chart.relpath],
            )
            for scenario in cfg.scenarios
        }
        for chart in charts
    }
    freeze_json(
        cfg.out / "conditions.json",
        {
            key: {name: value.as_dict() for name, value in values.items()}
            for key, values in conditions.items()
        },
    )
    run = {
        "report_version": REPORT_VERSION,
        **inputs,
        "mode": cfg.mode,
        "checkpoint": str(cfg.checkpoint.resolve()),
        "checkpoint_sha256": file_digest(cfg.checkpoint),
        "tokenizer": TOKENIZER_VERSION,
        "cond_vec_version": COND_VEC_VERSION,
        "torch": str(torch.__version__),
        **direct_audio_settings(),
        "device": str(device),
        "precision": cfg.precision,
        "teacher_precision": ("bf16" if use_amp else "fp32")
        if cfg.mode in ("teacher", "all")
        else None,
        "generation_precision": ("bf16" if device.type == "cuda" else "fp32")
        if cfg.mode in ("constrained", "unconstrained", "all")
        else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "scenarios": list(cfg.scenarios),
        "seeds": list(cfg.seeds),
        "temperature": cfg.temperature,
        "keep_frames": cfg.keep_frames,
        "html": cfg.html,
        "motif_vocab_sha256": file_digest(cfg.motif_vocab) if cfg.motif_vocab else None,
        "motif_vocab": str(cfg.motif_vocab.resolve()) if cfg.motif_vocab else None,
        "subset": cfg.limit is not None or bool(cfg.paths),
        "full_diversity_seeds": cfg.seeds == DIVERSITY_SEEDS,
        "motif_training_subset": motif_data["corpus"].get("subset", False) if motif_data else None,
    }
    freeze_json(cfg.out / "run.json", run)
    for name in ("summary.json", "summary.md", "index.html"):
        (cfg.out / name).unlink(missing_ok=True)
    model = load_v1_model(cfg.checkpoint, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    collection = EvaluationCollection(vocabulary, cfg.scenarios)
    with Journal(cfg.out / "records.jsonl", resume=cfg.resume) as journal:
        for index, chart in enumerate(charts, start=1):
            tick = time.perf_counter()
            path = cfg.root / chart.relpath
            audio = load_chart_audio(path, use_cache=False)
            encoded = encode_chart(list(audio.notes), audio.redlines, audio.t_audio_ms)
            reference = decode_tokens(encoded.tokens, encoded.grid)
            collection.add_reference(
                chart.relpath, encoded.tokens, reference, encoded.grid, metadata[chart.relpath],
            )
            batch = None
            if cfg.mode in ("teacher", "all"):
                batch = build_v1_example(
                    audio,
                    meta=metadata[chart.relpath],
                    encoded=encoded,
                    record_legal=True,
                )
            samples = {}
            sample_tokens = {}
            for scenario, condition in conditions[chart.relpath].items():
                base = {
                    "relpath": chart.relpath,
                    "scenario": scenario,
                    "condition": condition.as_dict(),
                }
                for mode, seed in scenario_cases(cfg.mode, scenario, cfg.seeds):
                    if mode == "teacher":
                        cached = journal.get(base, mode)
                        journal.add(
                            cached
                            if cached is not None
                            else {
                                **base,
                                "mode": "teacher",
                                "metrics": teacher_metrics(
                                    model,
                                    batch,
                                    condition.meta,
                                    device=device,
                                    use_amp=use_amp,
                                ),
                            },
                        )
                    elif mode == "constrained":
                        row = journal.get(base, mode, seed)
                        tokens = generated_tokens(row)
                        if tokens is None:
                            row, events, tokens = _generate(
                                cfg,
                                model,
                                audio,
                                encoded.grid,
                                reference,
                                condition.meta,
                                base=base,
                                seed=seed,
                                device=device,
                            )
                        else:
                            events = decode_tokens(tokens, encoded.grid)
                        journal.add(row)
                        collection.add_generation(row, tokens, encoded.grid)
                        if scenario == "matched_full":
                            samples[seed] = events
                            sample_tokens[seed] = tokens
                    else:
                        cached = journal.get(base, mode, seed)
                        if cached is None:
                            result = generate_v1_unconstrained(
                                model,
                                audio.pcm,
                                audio.redlines,
                                condition.meta,
                                device=device,
                                seed=seed,
                                temperature=cfg.temperature,
                                keep_frames=cfg.keep_frames,
                            )
                            cached = {
                                **base,
                                "mode": "unconstrained",
                                "seed": seed,
                                "validity": result.validity,
                                "position": result.position,
                                "illegal_token": result.illegal_token,
                                "tokens": len(result.tokens),
                            }
                        journal.add(cached)
            if samples:
                collection.add_diversity(chart.relpath, samples, sample_tokens)
            logger.info(
                "v1 eval chart %s/%s %s tokens=%s elapsed_s=%.2f",
                index,
                len(charts),
                chart.relpath,
                len(encoded.tokens),
                time.perf_counter() - tick,
            )
    report = collection.finish(
        journal.rows, run=run, pool=pool, elapsed_s=time.perf_counter() - started,
        reused_records=journal.reused,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda" else None,
    )
    if cfg.html:
        write_html(report, journal.rows, cfg.out)
    (cfg.out / "summary.md").write_text(markdown(report), encoding="utf-8")
    _json(cfg.out / "summary.json", report)
    logger.info("v1 eval complete: %s elapsed_s=%.2f", cfg.out, report["elapsed_s"])
    return report


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this execution environment")
    return device


def _fit(cfg: EvalConfig, charts, inputs: dict) -> dict:
    durations = cached_audio_durations(
        cfg.root, [chart.relpath for chart in charts], gate_dir=cfg.out / "gate"
    )

    def token_sequences():
        for index, chart in enumerate(charts, start=1):
            duration = durations.get(chart.relpath)
            if duration is None:
                audio = load_chart_audio(cfg.root / chart.relpath, use_cache=False)
                duration = audio.t_audio_ms
                notes, redlines = audio.notes, audio.redlines
            else:
                path = cfg.root / chart.relpath
                notes = parse_beatmap(path).notes
                redlines = parse_redlines_file(path).redlines
            yield encode_chart(list(notes), redlines, duration).tokens
            if index % 100 == 0 or index == len(charts):
                logger.info("v1 motif corpus %s/%s", index, len(charts))

    fitted = fit_vocabulary(token_sequences(), k=cfg.motif_k, corpus=inputs)
    freeze_json(cfg.out / "motifs.json", fitted)
    return fitted


def _generate(cfg, model, audio, grid, reference, meta, *, base, seed, device):
    started = time.perf_counter()
    chart = generate_v1_chart(
        model,
        audio.pcm,
        audio.redlines,
        meta,
        device=device,
        seed=seed,
        temperature=cfg.temperature,
        keep_frames=cfg.keep_frames,
    )
    events = decode_tokens(list(chart.tokens), grid)
    identifier = hashlib.sha256(base["relpath"].encode()).hexdigest()[:20]
    folder = cfg.out / "generated" / identifier / base["scenario"] / str(seed)
    output = folder / "chart.osu"
    source_audio = chart_audio_path(cfg.root / base["relpath"]).resolve()
    shifted = bool(base.get("shift_samples"))
    playback = folder / ("shifted.wav" if shifted else source_audio.name)
    write_v1_osu(
        output,
        hit_lines_from_notes(chart.notes),
        list(chart.warnings),
        audio.redlines,
        audio_filename=playback.name,
    )
    audio_record = {}
    if shifted:
        write_playback_pcm(playback, audio.pcm)
        audio_record = {"playback_audio": playback.name, "playback_sha256": file_digest(playback)}
    elif not playback.is_symlink() or playback.resolve() != source_audio:
        playback.unlink(missing_ok=True)
        playback.symlink_to(source_audio)
    _json(folder / "tokens.json", list(chart.tokens))
    generated_meta = compute_chart_meta(output)
    return (
        {
            **base,
            **audio_record,
            "mode": "constrained",
            "seed": seed,
            "output": str(output),
            "osu_sha256": file_digest(output),
            "tokens_sha256": file_digest(folder / "tokens.json"),
            "tokens": len(chart.tokens),
            "notes": len(chart.notes),
            "warnings": list(chart.warnings),
            "match": endpoint_match(events, reference),
            "condition_error": condition_error(meta, generated_meta),
            "generated_meta": generated_meta.to_dict(),
            "distribution": event_distribution(events, grid, chart.tokens),
            "plausibility": plausibility(chart.notes),
            "elapsed_s": time.perf_counter() - started,
        },
        events,
        chart.tokens,
    )
