#!/usr/bin/env python3
"""Cache frozen v1 audio features and fit small physical-onset readouts.

The input JSON lists WAVs, known PCM onsets, redlines, and whole-song split groups.
It must be prepared independently of probe fitting. No reference notes or target
conditions enter feature extraction or the readout.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf
import torch

from audio2map.eval.v1.inputs import file_digest, freeze_json
from audio2map.eval.v1.onset import capture_features, frame_targets
from audio2map.eval.v1.onset_fit import fit_diagnostics
from audio2map.features.audio.v1.constants import SAMPLE_RATE
from audio2map.generate.v1.chart import load_v1_model
from audio2map.osu.v1.timing import Redline

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "audio2map/eval/v1/onset.py",
    "audio2map/model/v1/onset.py",
    "audio2map/generate/v1/stream.py",
    "audio2map/model/v1/encoder.py",
    "audio2map/features/audio/v1/mel.py",
)


def check_splits(rows: list[dict]) -> None:
    assignments = defaultdict(set)
    for row in rows:
        assignments[row["group"]].add(row["split"])
    if any(len(splits) != 1 for splits in assignments.values()):
        raise ValueError("one whole-song group occurs in multiple probe splits")
    if {r["split"] for r in rows} != {"train", "dev", "test"}:
        raise ValueError("probe inputs require train, dev, test")
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError("probe keys must be unique")


def capture(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["rows"]
    check_splits(rows)
    protocol = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_digest(args.manifest),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_digest(args.checkpoint),
        "code_sha256": {p: file_digest(ROOT / p) for p in SOURCE_FILES},
        "cache_dtype": "float16",
        "weights_frozen": True,
        "target_clock": "physical PCM insertion samples, including input delay",
    }
    freeze_json(args.output / "capture_protocol.json", protocol)
    for name in SOURCE_FILES:
        snapshot = args.output / "executed_source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, snapshot)
    model = load_v1_model(args.checkpoint, torch.device(args.device)).requires_grad_(False)
    cache = args.output / "features"
    cache.mkdir(exist_ok=True)
    index = []
    for number, row in enumerate(rows, 1):
        audio = Path(row["audio"])
        if file_digest(audio) != row["audio_sha256"]:
            raise ValueError(f"changed audio: {audio}")
        pcm, sr = sf.read(audio, dtype="float32")
        if sr != SAMPLE_RATE or pcm.ndim != 1:
            raise ValueError("probe input must be mono native-rate audio")
        redlines = [Redline(**r) for r in row["redlines"]]
        variants = (False, True) if row["kind"] == "review" else (False,)
        for silent in variants:
            key = row["key"] + ("_silence" if silent else "")
            target = cache / (key + ".npz")
            if not target.exists():
                logger.info("capture %s/%s %s", number, len(rows), key)
                features, frame_lo = capture_features(
                    model,
                    np.zeros_like(pcm) if silent else pcm,
                    redlines,
                    device=torch.device(args.device),
                )
                labels = frame_targets(row["onsets_ms"], frame_lo, len(features["encoder"]))
                if any(not np.isfinite(x).all() for x in features.values()):
                    raise ValueError(f"nonfinite features: {key}")
                np.savez(
                    target,
                    **{k: v.astype(np.float16) for k, v in features.items()},
                    frame_lo=np.array(frame_lo),
                    present=labels.present,
                    fraction=labels.fraction,
                )
            index.append(
                {
                    **row,
                    "key": key,
                    "silent": silent,
                    "cache": str(target),
                    "cache_sha256": file_digest(target),
                }
            )
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("original model must stay frozen and gradient-free")
    freeze_json(args.output / "feature_index.json", {"rows": index})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=("capture", "fit", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage in ("capture", "all"):
        capture(args)
    if args.stage in ("fit", "all"):
        fit_diagnostics(args.output, device=torch.device(args.device), epochs=args.epochs)


if __name__ == "__main__":
    main()
