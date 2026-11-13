"""Observe native generation and inspect raw versus actual legal-token distributions."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import torch

from audio2map.generate.v1.sample import sampling_distribution
from audio2map.generate.v1.trace import SamplingStep
from audio2map.tokens.v1.vocab import TOKENIZER_VERSION, build_vocab, invert_vocab

BASE_TEMPERATURE = 1.0
DISPLAY_TOP = 5


class LogitRecorder:
    """A sampling observer with no random draws or model-state access."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.logits: list[np.ndarray] = []

    def __call__(self, step: SamplingStep) -> None:
        record = {
            f.name: getattr(step, f.name)
            for f in fields(step)
            if f.name not in ("raw_logits", "allowed_tokens")
        }
        record["allowed_ids"] = sorted(build_vocab()[t] for t in step.allowed_tokens)
        self.rows.append(record)
        self.logits.append(step.raw_logits.numpy().copy())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            logits=np.stack(self.logits),
            metadata_json=np.array(
                json.dumps(
                    {
                        "tokenizer_version": TOKENIZER_VERSION,
                        "rows": self.rows,
                    }
                )
            ),
        )


def load_trace(path: Path) -> tuple[list[dict], np.ndarray]:
    with np.load(path) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        logits = archive["logits"]
    if metadata["tokenizer_version"] != TOKENIZER_VERSION:
        raise ValueError("trace vocabulary version does not match")
    if logits.shape != (len(metadata["rows"]), len(build_vocab())):
        raise ValueError("trace metadata and logits do not align")
    return metadata["rows"], logits


def decision_distribution(logits: np.ndarray, allowed_ids: list[int], temperature: float) -> dict:
    scores = torch.as_tensor(np.array(logits, copy=True), dtype=torch.float32)
    names = invert_vocab()
    allowed = frozenset(names[i] for i in allowed_ids)
    raw = scores.softmax(dim=-1)
    legal = sampling_distribution(scores, allowed, build_vocab(), BASE_TEMPERATURE)
    policy = sampling_distribution(scores, allowed, build_vocab(), temperature)
    return {
        "logits": scores,
        "raw": raw,
        "legal": legal,
        "policy": policy,
        "legal_mass_raw": float(raw[allowed_ids].sum()),
        "greedy": names[int(legal.argmax())],
    }


def token_stats(distribution: dict, token: str, allowed_ids: list[int]) -> dict:
    token_id = build_vocab()[token]
    logit = distribution["logits"][token_id]
    legal = token_id in allowed_ids
    return {
        "token": token,
        "logit": float(logit),
        "raw_rank": int((distribution["logits"] > logit).sum()) + 1,
        "legal_rank": int((distribution["logits"][allowed_ids] > logit).sum()) + 1
        if legal
        else None,
        "legal": legal,
        "raw_probability": float(distribution["raw"][token_id]),
        "legal_probability": float(distribution["legal"][token_id]),
        "policy_probability": float(distribution["policy"][token_id]),
    }


def top_tokens(distribution: dict, allowed_ids: list[int], count: int = DISPLAY_TOP) -> list[dict]:
    names = invert_vocab()
    order = sorted(allowed_ids, key=lambda i: -float(distribution["legal"][i]))[:count]
    return [token_stats(distribution, names[i], allowed_ids) for i in order]
