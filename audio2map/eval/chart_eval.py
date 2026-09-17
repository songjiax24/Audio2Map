"""Teacher-forcing token accuracy and full-chart inference note F1."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from audio2map.dataset.sample import build_sample
from audio2map.eval.note_match import NoteMatchStats, compare_note_lists
from audio2map.eval.token_accuracy import SplitTokenAccuracy, split_token_accuracy
from audio2map.generate.overlap import (
    DecodeConfig,
    GenerationRangeConfig,
    OverlapConfig,
    generate_chart_notes,
)
from audio2map.generate.service import resolve_audio_file
from audio2map.grid import CanonicalTiming
from audio2map.model.model import AudioChartModel
from audio2map.osu.parser import parse_beatmap
from audio2map.tokens import invert_vocab

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AggregateStats:
    charts: int = 0
    token_correct: int = 0
    token_total: int = 0
    split_token: SplitTokenAccuracy = field(default_factory=SplitTokenAccuracy)

    def to_dict(self) -> dict:
        return {
            "charts": self.charts,
            "token_acc": self.token_correct / self.token_total if self.token_total else 0.0,
            "split_token_acc": self.split_token.to_dict(),
        }


def eval_teacher_forcing(
    model: AudioChartModel,
    paths: list[Path],
    *,
    device: torch.device,
    seed: int = 0,
    build_grid_if_missing: bool = False,
    fixed_start_bar: int | None = None,
    grid_dir: Path | None = None,
) -> AggregateStats:
    """Token accuracy with GT prefix (diagnostic upper bound)."""
    agg = AggregateStats()
    rng = random.Random(seed)
    id_to_token = invert_vocab()
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for path in paths:
            sample = build_sample(
                path,
                rng=rng,
                build_grid_if_missing=build_grid_if_missing,
                start_bar=fixed_start_bar,
                grid_dir=grid_dir,
            )
            if sample is None:
                logger.warning("skip teacher eval, cannot build window: %s", path)
                continue
            agg.charts += 1
            token_ids = torch.tensor([sample.token_ids], dtype=torch.long, device=device)
            loss_mask = torch.tensor([sample.loss_mask], dtype=torch.float32, device=device)
            audio = torch.tensor([sample.audio_features], dtype=torch.float32, device=device)
            cond = torch.tensor([sample.cond_vec], dtype=torch.float32, device=device)
            attn = torch.ones(1, token_ids.shape[1], dtype=torch.bool, device=device)

            logits = model(audio, cond, token_ids, attn_mask=attn)
            targets = token_ids[:, 1:]
            mask = loss_mask[:, 1:]
            pred = logits.argmax(dim=-1)
            correct = ((pred == targets) & mask.bool()).sum().item()
            total = mask.sum().item()
            agg.token_correct += int(correct)
            agg.token_total += int(total)
            agg.split_token.merge(
                split_token_accuracy(pred[0], targets[0], mask[0], id_to_token=id_to_token)
            )

    return agg


def eval_inference_chart(
    model: AudioChartModel,
    path: Path,
    *,
    device: torch.device,
    cond_vec: np.ndarray,
    overlap: OverlapConfig | None = None,
    decode: DecodeConfig | None = None,
    build_grid_if_missing: bool = True,
    range_mode: Literal["audio_full", "reference_chart"] = "audio_full",
    post_margin_bars: int = 4,
    grid_dir: Path | None = None,
) -> NoteMatchStats:
    beatmap = parse_beatmap(path)
    timing = CanonicalTiming.from_beatmap(beatmap)
    pred, _ = generate_chart_notes(
        model,
        audio_path=resolve_audio_file(path),
        timing=timing,
        cond_vec=cond_vec,
        grid_dir=grid_dir,
        overlap=overlap or OverlapConfig(),
        range_cfg=GenerationRangeConfig(mode=range_mode, post_margin_bars=post_margin_bars),
        device=device,
        decode=decode or DecodeConfig(),
        reference_notes=beatmap.notes,
        build_grid_if_missing=build_grid_if_missing,
    )
    return compare_note_lists(pred, beatmap.notes, timing)
