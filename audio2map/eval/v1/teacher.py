"""Use the training validator for whole-chart teacher evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, replace

import torch

from audio2map.features.cond.chart import ChartMeta
from audio2map.features.cond.v1.vec import build_cond
from audio2map.model.v1.batch import V1Batch
from audio2map.model.v1.model import V1ChartModel
from audio2map.tokens.v1.vocab import invert_vocab
from audio2map.train.v1.validation import validate_charts

TOKEN_TYPES = ("pos", "row", "beat", "eos")


def teacher_metrics(
    model: V1ChartModel, batch: V1Batch, meta: ChartMeta, *, device: torch.device, use_amp: bool
) -> dict:
    vector = build_cond(meta)
    conditioned = replace(
        batch,
        cond_x=torch.from_numpy(vector.values).unsqueeze(0),
        cond_m=torch.from_numpy(vector.m_real).unsqueeze(0),
    )
    readout = validate_charts(model, [conditioned], device, use_amp=use_amp)
    names = invert_vocab()
    counts = {kind: 0 for kind in TOKEN_TYPES}
    for token_id in batch.token_ids[0, 1:].tolist():
        token = names[token_id]
        kind = token[1:-1].split("_", 1)[0].lower()
        if kind in counts:
            counts[kind] += 1
    values = asdict(readout)
    values["tokens"] = sum(counts.values())
    values["type_counts"] = counts
    values["perplexity"] = _perplexity(readout.loss_token)
    return values


def summarize_teacher(rows: list[dict]) -> dict:
    if not rows:
        return {"charts": 0, "tokens": 0}
    tokens = sum(row["tokens"] for row in rows)

    def weighted(name: str, counts: list[int]) -> float | None:
        total = sum(counts)
        return sum(row[name] * count for row, count in zip(rows, counts)) / total if total else None

    sizes = [row["tokens"] for row in rows]
    result = {"charts": len(rows), "tokens": tokens}
    for name in ("loss_chart", "acc_chart", "legal_top1_chart", "legal_prob_chart"):
        result[name] = sum(row[name] for row in rows) / len(rows)
    for name in ("loss_token", "acc_token", "legal_top1_token", "legal_prob_token"):
        result[name] = weighted(name, sizes)
    result["perplexity"] = _perplexity(result["loss_token"])
    result["by_type"] = {}
    for kind in TOKEN_TYPES:
        counts = [row["type_counts"][kind] for row in rows]
        loss = weighted(f"loss_{kind}", counts)
        result["by_type"][kind] = {
            "tokens": sum(counts),
            "nll": loss,
            "perplexity": _perplexity(loss),
            "accuracy": weighted(kind, counts),
            "legal_top1": weighted(f"legal_top1_{kind}", counts),
            "legal_probability": weighted(f"legal_prob_{kind}", counts),
        }
    detailed = [row["row_counts"] for row in rows if row.get("row_counts") is not None]
    result["row_details"] = {
        "available_charts": len(detailed),
        "metrics": {
            name: {
                "correct": sum(row[name][0] for row in detailed),
                "total": sum(row[name][1] for row in detailed),
                "rate": (
                    sum(row[name][0] for row in detailed) / sum(row[name][1] for row in detailed)
                )
                if sum(row[name][1] for row in detailed) else None,
            }
            for name in detailed[0]
        } if detailed else {},
    }
    return result


def _perplexity(loss: float | None) -> float | None:
    if loss is None:
        return None
    try:
        return math.exp(loss)
    except OverflowError:
        return None
