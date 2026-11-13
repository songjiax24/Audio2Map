"""Frozen v1 feature capture and millisecond onset diagnostics.

Unlike chart-quality metrics, this task has known physical input events. The
probe receives no note history, target condition vector, or reference tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from audio2map.features.audio.v1.constants import DT_MS
from audio2map.generate.v1.stream import KEEP_FRAMES, encode_context
from audio2map.grid.v1.beatgrid import build_beatgrid
from audio2map.model.v1.model import V1ChartModel
from audio2map.osu.v1.timing import Redline
from audio2map.utils.time import MS_PER_SECOND, SAMPLE_RATE

STAGES = ("mel", "stem", "encoder", "context", "phase")
PRIMARY_TOLERANCE_MS = 5
TOLERANCES_MS = (1, 2, PRIMARY_TOLERANCE_MS, 10)


@dataclass(frozen=True)
class FrameTargets:
    present: np.ndarray
    fraction: np.ndarray


def frame_targets(onsets_ms: Sequence[float], frame_lo: int, count: int) -> FrameTargets:
    """Floor-bin actual PCM onsets, retaining subframe timing and rejecting collisions."""
    onsets = np.asarray(onsets_ms, dtype=np.float64)
    if not np.isfinite(onsets).all() or len(np.unique(onsets)) != len(onsets):
        raise ValueError("onsets must be finite and unique")
    absolute = np.floor(onsets / DT_MS).astype(np.int64)
    local = absolute - frame_lo
    if np.any(local < 0) or np.any(local >= count):
        raise ValueError("onsets outside captured frame coverage")
    if len(np.unique(local)) != len(local):
        raise ValueError("multiple independent onsets in one probe frame")
    present = np.zeros(count, dtype=np.float32)
    fraction = np.zeros(count, dtype=np.float32)
    present[local] = 1
    fraction[local] = onsets / DT_MS - absolute
    return FrameTargets(present, fraction)


def onset_match(predicted, reference, tolerance_ms: float) -> dict:
    """Maximum cardinality sorted one-to-one matching with inclusive tolerance."""
    if tolerance_ms < 0 or not np.isfinite(tolerance_ms):
        raise ValueError("tolerance must be finite and nonnegative")
    pred = np.sort(np.asarray(predicted, dtype=np.float64))
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    if not np.isfinite(pred).all() or not np.isfinite(ref).all():
        raise ValueError("onsets must be finite")
    i = j = tp = 0
    while i < len(pred) and j < len(ref):
        if pred[i] < ref[j] - tolerance_ms:
            i += 1
        elif ref[j] < pred[i] - tolerance_ms:
            j += 1
        else:
            tp += 1
            i += 1
            j += 1
    return match_stats(tp, len(pred) - tp, len(ref) - tp)


def match_stats(tp: int, fp: int, fn: int) -> dict:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
    }


def onset_scores(predicted, reference) -> dict:
    return {str(t): onset_match(predicted, reference, t) for t in TOLERANCES_MS}


def predicted_onsets(probability, fraction, frame_lo: int, threshold: float) -> np.ndarray:
    selected = np.flatnonzero(np.asarray(probability) >= threshold)
    return (selected + frame_lo + np.asarray(fraction)[selected]) * DT_MS


def capture_features(
    model: V1ChartModel,
    pcm: np.ndarray,
    redlines: Sequence[Redline],
    *,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], int]:
    """Hook the native single-segment inference path; never reimplement its encoder.

    This diagnostic intentionally rejects multi-segment songs rather than joining
    overlapping hook outputs incorrectly. Each stored row has its real global g.
    """
    if model.training:
        raise ValueError("feature capture requires eval mode")
    duration = len(pcm) / SAMPLE_RATE * MS_PER_SECOND
    grid = build_beatgrid(redlines, duration)
    captured = {}

    def stem_hook(_module, inputs, output):
        count = output.shape[1]
        captured["mel"] = inputs[0][0, :, 2 : 2 * count + 2 : 2].T.float().cpu().numpy()
        captured["stem"] = output[0].float().cpu().numpy()

    def encoder_hook(_module, _inputs, output):
        if "encoder" in captured:
            raise ValueError("probe capture supports one encoder segment per song")
        captured["encoder"] = output[0].float().cpu().numpy()

    def phase_hook(_module, _inputs, output):
        captured["phase"] = output[0].float().cpu().numpy()

    handles = [
        model.encoder.stem.register_forward_hook(stem_hook),
        model.encoder.register_forward_hook(encoder_hook),
        model.phase.register_forward_hook(phase_hook),
    ]
    try:
        with torch.inference_mode():
            context, g, segments = encode_context(
                model,
                pcm,
                grid,
                redlines,
                device=device,
                keep_frames=KEEP_FRAMES,
            )
        if segments != 1:
            raise ValueError("probe capture supports one encoder segment per song")
        captured["context"] = context[0].float().cpu().numpy()
        frame_lo = int(g[0, 0])
        if any(len(values) != g.shape[1] for values in captured.values()):
            raise RuntimeError("hook rows do not align with native context")
        return captured, frame_lo
    finally:
        for handle in handles:
            handle.remove()
