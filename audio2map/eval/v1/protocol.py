"""Validated probe requests and train-only calibration of pattern request levels."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

import numpy as np

from audio2map.eval.v1.archive import load_archive
from audio2map.eval.v1.inputs import file_digest, freeze_json, verify_training_corpus
from audio2map.eval.v1.motifs import load_vocabulary
from audio2map.eval.v1.statistics import describe
from audio2map.features.cond.v1.store import ensure_v1_chart_meta, load_v1_chart_meta
from audio2map.features.cond.v1.vec import COND_SOURCE_FIELDS

AUDIO_OFFSETS_S = (-5.0, 0.0, 5.0)
FINE_AUDIO_OFFSETS_S = (-5.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 5.0)
MAX_SHIFT_FRACTION = 0.25
CONTROL_LEVELS = {
    "official_sr": (2.0, 4.0, 6.0),
    "analyzer_ln_percent": (0.0, 0.5, 1.0),
}
PATTERN_FIELDS = tuple(name for name in COND_SOURCE_FIELDS if name.startswith("pattern_"))
POSITIVE_QUANTILES = (0.25, 0.75)
PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class ProbeProtocol:
    offsets_s: tuple[float, ...]
    controls: dict[str, tuple[float, ...]]
    provenance: dict


def load_protocol(path: Path | None) -> ProbeProtocol:
    if path is None:
        return ProbeProtocol(AUDIO_OFFSETS_S, CONTROL_LEVELS, {})
    data = json.loads(path.read_text())
    if data.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported probe protocol version")
    offsets = tuple(float(value) for value in data["audio_offsets_s"])
    if not offsets or not all(math.isfinite(value) for value in offsets):
        raise ValueError("audio offsets must be nonempty and finite")
    if len(set(offsets)) != len(offsets) or 0.0 not in offsets:
        raise ValueError("audio offsets must be unique and include zero")
    controls = {}
    for name, requested in data["controls"].items():
        if name not in COND_SOURCE_FIELDS:
            raise ValueError(f"unknown condition field: {name}")
        values = tuple(float(value) for value in requested)
        if len(values) < 2 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"need at least two finite levels for {name}")
        if tuple(sorted(set(values))) != values:
            raise ValueError(f"levels must be strictly increasing for {name}")
        if name in PATTERN_FIELDS and not all(0 <= value <= 1 for value in values):
            raise ValueError("pattern shares must lie in [0, 1]")
        controls[name] = values
    return ProbeProtocol(offsets, controls, {
        "path": str(path.resolve()), "sha256": file_digest(path), "definition": data,
    })


def pattern_levels(values) -> dict:
    finite = [float(value) for value in values
              if value is not None and math.isfinite(value) and 0 <= value <= 1]
    positive = [value for value in finite if value > 0]
    levels = sorted({0.0, *np.quantile(positive, POSITIVE_QUANTILES).tolist()}) if positive else []
    return {
        "distribution": describe(finite), "positive_count": len(positive),
        "zero_count": len(finite) - len(positive), "levels": levels,
        "selection": (
            "zero plus q25/q75 among positive train observations; duplicate levels removed"
        ),
    }


@dataclass(frozen=True)
class CalibrationConfig:
    source: Path
    training_vocab: Path
    out: Path
    charts: int = 512
    seed: int = 0
    workers: int = 4


def calibrate_patterns(cfg: CalibrationConfig) -> dict:
    if cfg.charts < 1 or cfg.workers < 1:
        raise ValueError("calibration chart and worker counts must be positive")
    archive = load_archive(cfg.source)
    fitted = json.loads(cfg.training_vocab.read_text())
    load_vocabulary(fitted)
    corpus = fitted["corpus"]
    if corpus["split_sha256"] != archive.run["split_sha256"]:
        raise ValueError("training corpus and evaluation use different split manifests")
    available = sorted(corpus["charts"])
    manifest_path = Path(archive.run["split_manifest"])
    verify_training_corpus(corpus, manifest_path, archive.root)
    selected = sorted(random.Random(cfg.seed).sample(available, min(cfg.charts, len(available))))
    digests = {path: file_digest(archive.root / path) for path in selected}
    if any(digest != corpus["chart_sha256"][path] for path, digest in digests.items()):
        raise ValueError("a calibration chart differs from the frozen training corpus")
    cfg.out.mkdir(parents=True, exist_ok=True)
    inputs = {
        "split": "train", "training_corpus_sha256": file_digest(cfg.training_vocab),
        "split_sha256": corpus["split_sha256"], "raw_root": str(archive.root),
        "available_charts": len(available), "selected": selected, "chart_sha256": digests,
        "seed": cfg.seed, "subset": len(selected) < len(available),
    }
    freeze_json(cfg.out / "inputs.json", inputs)
    meta_path = cfg.out / "metadata.jsonl"
    ensure_v1_chart_meta(selected, archive.root, workers=cfg.workers, path=meta_path)
    metadata = load_v1_chart_meta(meta_path)
    statistics = {
        name: pattern_levels([getattr(metadata[path], name) for path in selected])
        for name in PATTERN_FIELDS
    }
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "audio_offsets_s": list(FINE_AUDIO_OFFSETS_S),
        "controls": {name: row["levels"] for name, row in statistics.items() if row["levels"]},
        "calibration": {
            **inputs, "metadata_sha256": file_digest(meta_path), "statistics": statistics,
            "errors": {path: meta.error for path, meta in metadata.items() if meta.error},
            "pattern_semantics": (
                "Per-class interval-union time share; no cross-class renormalization"
            ),
        },
    }
    freeze_json(cfg.out / "protocol.json", protocol)
    return protocol
