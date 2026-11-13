#!/usr/bin/env python3
"""Trace v1 raw logits through native generation, forced replay, or prefix continuation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import time

import soundfile as sf
import torch
import numpy as np

from audio2map.eval.v1.inputs import file_digest, freeze_json, matches_digest
from audio2map.eval.v1.journal import write_playback_pcm
from audio2map.eval.v1.logits import LogitRecorder
from audio2map.features.audio.v1.constants import SAMPLE_RATE
from audio2map.features.cond.chart import ChartMeta
from audio2map.generate.v1.chart import generate_v1_chart, load_v1_model
from audio2map.generate.v1.notes import hit_lines_from_notes
from audio2map.osu.v1.export import write_v1_osu
from audio2map.osu.v1.timing import parse_redlines_file

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "audio2map/generate/v1/sample.py",
    "audio2map/generate/v1/chart.py",
    "audio2map/generate/v1/trace.py",
    "audio2map/eval/v1/logits.py",
    "audio2map/eval/v1/journal.py",
    "audio2map/model/v1/kv.py",
    "audio2map/generate/v1/stream.py",
    "tools/analyze_generation_logits.py",
)


def run(args) -> None:
    manifest = json.loads(args.manifest.read_text())
    protocol = {
        "manifest_sha256": file_digest(args.manifest),
        "checkpoint_sha256": file_digest(args.checkpoint),
        "device": str(torch.device(args.device)),
        "torch": str(torch.__version__),
        "code_sha256": {p: file_digest(ROOT / p) for p in SOURCES},
        "logits": "float32 CPU copies before mask/temperature; native sampler and RNG unchanged",
    }
    freeze_json(args.output / "protocol.json", protocol)
    protocol_digest = file_digest(args.output / "protocol.json")
    for name in SOURCES:
        dest = args.output / "executed_source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    model = load_v1_model(args.checkpoint, torch.device(args.device))
    results = []
    for case in manifest["rows"]:
        target = args.output / case["key"]
        target.mkdir(exist_ok=True)
        saved = target / "result.json"
        audio_path = Path(case["audio"])
        if not matches_digest(audio_path, case["audio_sha256"]):
            raise ValueError(f"audio input changed: {audio_path}")
        reference = Path(case["reference"])
        if not matches_digest(reference, case["reference_sha256"]):
            raise ValueError(f"reference input changed: {reference}")
        lines = parse_redlines_file(reference).redlines
        prefix = None
        if case.get("prefix"):
            path = Path(case["prefix"])
            if not matches_digest(path, case["prefix_sha256"]):
                raise ValueError(f"prefix input changed: {path}")
            prefix = json.loads(path.read_text())
        expected = None
        if case.get("expected_tokens"):
            expected_path = Path(case["expected_tokens"])
            if not matches_digest(expected_path, case["expected_tokens_sha256"]):
                raise ValueError(f"expected tokens changed: {expected_path}")
            expected = json.loads(expected_path.read_text())
        if saved.exists():
            try:
                result = json.loads(saved.read_text())
            except json.JSONDecodeError:
                logger.warning("rebuilding incomplete result: %s", saved)
            else:
                if result["protocol_sha256"] != protocol_digest:
                    raise ValueError(f"result belongs to a different protocol: {saved}")
                artifacts = ("trace", "tokens", "chart", "playback") if case.get("silent") else (
                    "trace", "tokens", "chart",
                )
                if all(
                    matches_digest(Path(result[key]), result.get(key + "_sha256"))
                    for key in artifacts
                ):
                    results.append(result)
                    continue
        pcm, sr = sf.read(audio_path, dtype="float32")
        if sr != SAMPLE_RATE or pcm.ndim != 1:
            raise ValueError("trace input must be native-rate mono PCM")
        if case.get("silent"):
            pcm = np.zeros_like(pcm)
        recorder = LogitRecorder()
        logger.info("START %s", case["key"])
        start = time.monotonic()
        chart = generate_v1_chart(
            model,
            pcm,
            lines,
            ChartMeta.from_dict(case["condition"]),
            device=torch.device(args.device),
            seed=case["seed"],
            temperature=case["temperature"],
            prefix=prefix,
            history_start=case.get("history_start", 0),
            observer=recorder,
        )
        parity = None
        if expected is not None:
            parity = list(chart.tokens) == expected
            if not parity:
                raise RuntimeError(f"trace altered or failed to reproduce tokens: {case['key']}")
        trace_path = target / "logits.npz"
        recorder.save(trace_path)
        tokens_path = target / "tokens.json"
        tokens_path.write_text(json.dumps(chart.tokens))
        chart_path = target / "chart.osu"
        playback = {}
        if case.get("silent"):
            silent_path = target / "silence.wav"
            write_playback_pcm(silent_path, pcm)
            playback = {
                "playback": str(silent_path), "playback_sha256": file_digest(silent_path),
            }
        write_v1_osu(
            chart_path,
            hit_lines_from_notes(chart.notes),
            list(chart.warnings),
            lines,
            audio_filename="silence.wav" if playback else audio_path.name,
        )
        result = {
            **case,
            **playback,
            "protocol_sha256": protocol_digest,
            "trace": str(trace_path),
            "trace_sha256": file_digest(trace_path),
            "tokens": str(tokens_path),
            "tokens_sha256": file_digest(tokens_path),
            "chart": str(chart_path),
            "chart_sha256": file_digest(chart_path),
            "token_parity": parity,
            "steps": len(recorder.rows),
            "elapsed_s": time.monotonic() - start,
        }
        temporary = saved.with_suffix(".part")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        temporary.replace(saved)
        results.append(result)
        logger.info("DONE %s %.1fs parity=%s", case["key"], result["elapsed_s"], parity)
    (args.output / "results.json").write_text(json.dumps({"rows": results}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
