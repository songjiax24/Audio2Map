"""Fit diagnostic onset readouts; select epochs and thresholds on development only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from audio2map.eval.v1.inputs import file_digest, freeze_json
from audio2map.eval.v1.onset import (
    PRIMARY_TOLERANCE_MS,
    STAGES,
    match_stats,
    onset_match,
    onset_scores,
    predicted_onsets,
)
from audio2map.features.audio.v1.constants import DT_MS
from audio2map.model.v1.onset import OnsetProbe
from audio2map.utils.time import MS_PER_SECOND

logger = logging.getLogger(__name__)
STD_FLOOR = 1e-4
BATCH_FRAMES = 4096
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0001
VALIDATE_EVERY = 5
THRESHOLDS = tuple(float(x) for x in np.linspace(0.1, 0.9, 9))
ROLL_FRAMES = 137
SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class ProbeConfig:
    stage: str
    hidden: int
    seed: int
    epochs: int

    @property
    def key(self) -> str:
        return f"{self.stage}_h{self.hidden}_s{self.seed}"


def load_stage(index: list[dict], stage: str) -> list[dict]:
    rows = []
    for record in index:
        with np.load(record["cache"]) as cache:
            rows.append(
                {
                    **record,
                    "features": cache[stage].astype(np.float32),
                    "frame_lo": int(cache["frame_lo"]),
                    "present": cache["present"],
                    "fraction": cache["fraction"],
                }
            )
    return rows


def infer(probe, features, mean, scale, device) -> tuple[np.ndarray, np.ndarray]:
    x = torch.as_tensor(features, device=device)
    with torch.no_grad():
        logit, fraction = probe((x - mean) / scale)
    return logit.sigmoid().cpu().numpy(), fraction.cpu().numpy()


def evaluate(probe, rows, mean, scale, device, threshold: float) -> list[dict]:
    scored = []
    for row in rows:
        probability, fraction = infer(probe, row["features"], mean, scale, device)
        times = predicted_onsets(probability, fraction, row["frame_lo"], threshold)
        positive = row["present"].astype(bool)
        errors = np.abs(fraction[positive] - row["fraction"][positive]) * DT_MS
        scored.append(
            {
                "key": row["key"],
                "kind": row["kind"],
                "silent": row["silent"],
                "predicted_ms": times.tolist(),
                "scores": onset_scores(times, [] if row["silent"] else row["onsets_ms"]),
                "oracle_presence_offset_mae_ms": float(errors.mean())
                if errors.size and not row["silent"]
                else None,
                "oracle_presence_offset_p95_ms": float(np.quantile(errors, 0.95))
                if errors.size and not row["silent"]
                else None,
                "false_positives_per_minute": len(times)
                / (len(probability) * DT_MS / MS_PER_SECOND)
                * SECONDS_PER_MINUTE
                if row["silent"]
                else None,
            }
        )
    return scored


def aggregate(rows: list[dict], tolerance: str = str(PRIMARY_TOLERANCE_MS)) -> dict:
    counts = [r["scores"][tolerance] for r in rows]
    return match_stats(*(sum(c[k] for c in counts) for k in ("tp", "fp", "fn")))


def choose_threshold(predictions: list[dict]) -> tuple[float, dict]:
    choices = []
    for threshold in THRESHOLDS:
        totals = [0, 0, 0]
        for row in predictions:
            times = predicted_onsets(row["probability"], row["offset"], row["frame_lo"], threshold)
            score = onset_match(times, row["onsets_ms"], PRIMARY_TOLERANCE_MS)
            for index, key in enumerate(("tp", "fp", "fn")):
                totals[index] += score[key]
        score = match_stats(*totals)
        choices.append((threshold, score))
    return max(choices, key=lambda x: (x[1]["f1"], x[1]["precision"]))


def fit_one(rows: list[dict], config: ProbeConfig, device: torch.device) -> tuple:
    torch.manual_seed(config.seed)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    x = torch.as_tensor(np.concatenate([r["features"] for r in train]), device=device)
    target = torch.as_tensor(np.concatenate([r["present"] for r in train]), device=device)
    offset = torch.as_tensor(np.concatenate([r["fraction"] for r in train]), device=device)
    mean = x.mean(dim=0)
    scale = x.std(dim=0).clamp_min(STD_FLOOR)
    x = (x - mean) / scale
    probe = OnsetProbe(x.shape[-1], config.hidden).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    positive_weight = (1 - target).sum() / target.sum()
    presence_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    best = None
    history = []
    for epoch in range(1, config.epochs + 1):
        order = torch.randperm(len(x), device=device)
        losses = []
        for batch in order.split(BATCH_FRAMES):
            logit, fraction = probe(x[batch])
            active = target[batch].bool()
            loss = presence_loss(logit, target[batch])
            if active.any():
                loss = loss + nn.functional.mse_loss(fraction[active], offset[batch][active])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % VALIDATE_EVERY and epoch != config.epochs:
            continue
        predictions = []
        for row in dev:
            probability, fraction = infer(probe, row["features"], mean, scale, device)
            predictions.append({**row, "probability": probability, "offset": fraction})
        threshold, score = choose_threshold(predictions)
        history.append(
            {"epoch": epoch, "threshold": threshold, "dev": score, "loss": float(np.mean(losses))}
        )
        logger.info(
            "fit %s epoch=%s dev5=%.4f threshold=%.1f", config.key, epoch, score["f1"], threshold
        )
        if best is None or score["f1"] > best["dev"]["f1"]:
            best = {
                "epoch": epoch,
                "threshold": threshold,
                "dev": score,
                "state": {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()},
            }
    probe.load_state_dict(best["state"])
    return probe, mean, scale, best, history


def fit_diagnostics(output: Path, *, device: torch.device, epochs: int) -> None:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    index_path = output / "feature_index.json"
    index = json.loads(index_path.read_text())["rows"]
    for row in index:
        if file_digest(Path(row["cache"])) != row["cache_sha256"]:
            raise ValueError(f"changed cache: {row['cache']}")
    configs = [
        ProbeConfig(stage, hidden, seed, epochs)
        for stage in STAGES
        for hidden in (0, 64)
        for seed in (0, 1)
    ]
    protocol = {
        "feature_index_sha256": file_digest(index_path),
        "code_sha256": file_digest(Path(__file__)),
        "configs": [asdict(c) for c in configs],
        "thresholds": THRESHOLDS,
        "batch_frames": BATCH_FRAMES,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "std_floor": STD_FLOOR,
        "primary_tolerance_ms": PRIMARY_TOLERANCE_MS,
        "selection": "maximum development micro F1 at 5ms, no test selection",
    }
    freeze_json(output / "fit_protocol.json", protocol)
    source_copy = output / "executed_source" / "audio2map/eval/v1/onset_fit.py"
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, source_copy)
    fits = output / "fits"
    fits.mkdir(exist_ok=True)
    results = []
    for stage in STAGES:
        rows = load_stage(index, stage)
        for config in (c for c in configs if c.stage == stage):
            path = fits / (config.key + ".json")
            if path.exists():
                result = json.loads(path.read_text())
                if result["fit_protocol_sha256"] != file_digest(output / "fit_protocol.json"):
                    raise ValueError(f"changed fitting protocol: {path}")
                if file_digest(Path(result["checkpoint"])) != result["checkpoint_sha256"]:
                    raise ValueError(f"changed probe checkpoint: {path}")
                results.append(result)
                continue
            probe, mean, scale, best, history = fit_one(rows, config, device)
            test = [r for r in rows if r["split"] == "test"]
            scores = evaluate(probe, test, mean, scale, device, best["threshold"])
            shuffled = []
            for row in test:
                if row["silent"]:
                    continue
                features = np.roll(row["features"], ROLL_FRAMES, axis=0)
                if stage == "phase":
                    features = row["features"]
                if stage == "context":
                    with np.load(row["cache"]) as cache:
                        features = np.roll(
                            cache["encoder"].astype(np.float32), ROLL_FRAMES, axis=0
                        ) + cache["phase"].astype(np.float32)
                shuffled.append({**row, "features": features})
            shuffled_scores = evaluate(
                probe,
                shuffled,
                mean,
                scale,
                device,
                best["threshold"],
            )
            result = {
                "config": asdict(config),
                "best_epoch": best["epoch"],
                "threshold": best["threshold"],
                "development": best["dev"],
                "history": history,
                "test": scores,
                "rolled_audio_features": shuffled_scores,
                "fit_protocol_sha256": file_digest(output / "fit_protocol.json"),
            }
            torch.save(
                {
                    "state": best["state"],
                    "mean": mean.cpu(),
                    "scale": scale.cpu(),
                    "config": asdict(config),
                    "threshold": best["threshold"],
                },
                fits / (config.key + ".pt"),
            )
            result["checkpoint"] = str(fits / (config.key + ".pt"))
            result["checkpoint_sha256"] = file_digest(Path(result["checkpoint"]))
            path.write_text(json.dumps(result, indent=2))
            results.append(result)
    (output / "results.json").write_text(json.dumps({"fits": results}, indent=2))
