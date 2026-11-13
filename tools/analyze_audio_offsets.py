#!/usr/bin/env python3
"""Measure audio attack / chart-head lag without loading a generative model.

This is a diagnostic, not an automatic retiming tool. Positive lag means that
audio changes follow chart heads. A spectral-change maximum is a proxy for an
attack, not a perceptual ground truth. Unresolved charts stay in the denominator.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
from pathlib import Path

import numpy as np

from audio2map.eval.v1.inputs import file_digest as digest, freeze_json, matches_digest
from audio2map.eval.v1.journal import read_records
from audio2map.features.audio.v1.decode import decode_audio
from audio2map.grid.v1.timefeat import MS_PER_MINUTE
from audio2map.osu.parser import chart_audio_path, parse_beatmap
from audio2map.osu.schema import NoteType
from audio2map.utils.time import MS_PER_SECOND, SAMPLE_RATE

logger = logging.getLogger(__name__)
HOP_SAMPLES = 22
FFT_SIZES = (256, 512)
FFT_BATCH = 2048
LOW_HZ = 80.0
HIGH_HZ = 10_000.0
LAGS_MS = np.arange(-80, 81, dtype=np.float64)
MIN_HEAD_GAP_MS = 80.0
MIN_ANCHORS = 80
MAX_ANCHORS = 1200
TIME_BLOCKS = 6
MIN_BLOCK_ANCHORS = 10
MIN_BLOCKS = 4
BOOTSTRAPS = 256
CI_MAX_WIDTH_MS = 20.0
WINDOW_AGREE_MS = 8.0
MIN_CONTRAST = 0.05
MIN_PROMINENCE = 0.01
PEAK_EXCLUSION_MS = 12.0
EDGE_GUARD_MS = 10.0
NEAR20_MS = (10.0, 30.0)
STRICT20_MS = (15.0, 25.0)
NUMERIC_FLOOR = np.finfo(np.float64).eps
CALIBRATION_SEED = 1707


def attack_curve(y: np.ndarray, fft_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Positive spectral changes, timed halfway between adjacent window centres.

    Centred Hann windows, explicit PCM padding, and a roughly 1 ms hop. FFTs
    are batched so memory does not grow as song length times FFT size.
    """
    padded = np.pad(y, (fft_size // 2, fft_size // 2))
    windows = np.lib.stride_tricks.sliding_window_view(padded, fft_size)[::HOP_SAMPLES]
    frequency = np.fft.rfftfreq(fft_size, 1.0 / SAMPLE_RATE)
    keep = (frequency >= LOW_HZ) & (frequency <= HIGH_HZ)
    taper = np.hanning(fft_size)
    previous = np.zeros(int(keep.sum()), dtype=np.float64)
    curves = []
    for start in range(0, len(windows), FFT_BATCH):
        spectrum = np.abs(np.fft.rfft(windows[start : start + FFT_BATCH] * taper))
        compressed = np.sqrt(spectrum[:, keep])
        changes = np.diff(compressed, axis=0, prepend=previous[None])
        curves.append(np.maximum(changes, 0.0).sum(axis=1))
        previous = compressed[-1]
    flux = np.concatenate(curves)
    times = (np.arange(len(flux)) - 0.5) * HOP_SAMPLES / SAMPLE_RATE * MS_PER_SECOND
    return times, flux


def calibration_signals() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Several exact digital attacks; never derive calibration from a real chart."""
    rng = np.random.default_rng(CALIBRATION_SEED)
    starts = np.asarray([400, 953, 1511, 2123, 2737], dtype=np.float64)
    tail_seconds = 0.18
    t = np.arange(round(tail_seconds * SAMPLE_RATE)) / SAMPLE_RATE
    noise = rng.standard_normal(len(t))
    fade_seconds = 0.005
    fade = np.minimum(t / fade_seconds, 1.0)
    voices = {
        "noise": noise * np.exp(-t / 0.025),
        "snare": (noise + np.sin(2 * np.pi * 180 * t)) * np.exp(-t / 0.04),
        "click": np.concatenate(([1.0], np.zeros(len(t) - 1))),
        "tone": np.sin(2 * np.pi * 880 * t) * np.exp(-t / 0.04),
        "ramp5ms": noise * fade * np.exp(-t / 0.04),
    }
    results = []
    for name, voice in voices.items():
        y = np.zeros(round(3.2 * SAMPLE_RATE), dtype=np.float32)
        actual = []
        for start in starts:
            sample = round(start / MS_PER_SECOND * SAMPLE_RATE)
            y[sample : sample + len(voice)] += voice
            actual.append(sample / SAMPLE_RATE * MS_PER_SECOND)
        results.append((name, y, np.asarray(actual)))
    return results


def calibrate() -> dict:
    details = {}
    for size in FFT_SIZES:
        rows = []
        for name, y, actual in calibration_signals():
            times, flux = attack_curve(y, size)
            for onset in actual:
                mask = np.abs(times - onset) <= PEAK_EXCLUSION_MS
                lag = float(times[mask][np.argmax(flux[mask])] - onset)
                rows.append({"voice": name, "lag_ms": lag})
        sharp = [r["lag_ms"] for r in rows if r["voice"] != "ramp5ms"]
        details[str(size)] = {"correction_ms": float(np.median(sharp)), "attacks": rows}
    return details


def head_times(chart) -> np.ndarray:
    return np.unique([note.time_ms for note in chart.notes]).astype(np.float64)


def anchors_for(heads: np.ndarray, duration_ms: float) -> np.ndarray:
    previous = np.diff(heads, prepend=-np.inf)
    following = np.diff(heads, append=np.inf)
    keep = (previous >= MIN_HEAD_GAP_MS) & (following >= MIN_HEAD_GAP_MS)
    keep &= (heads > -LAGS_MS[0]) & (heads < duration_ms - LAGS_MS[-1])
    anchors = heads[keep]
    if len(anchors) > MAX_ANCHORS:
        anchors = anchors[np.linspace(0, len(anchors) - 1, MAX_ANCHORS).astype(int)]
    return anchors


def lag_fit(
    times: np.ndarray,
    flux: np.ndarray,
    anchors: np.ndarray,
    correction_ms: float,
) -> dict:
    """Equal-weight head profiles; block bootstrap preserves local dependence."""
    if len(anchors) < MIN_ANCHORS:
        return {"status": "too_few_anchors", "anchors": len(anchors)}
    samples = anchors[:, None] + LAGS_MS[None] + correction_ms
    profiles = np.interp(samples.ravel(), times, flux).reshape(samples.shape)
    scale = profiles.max(axis=1)
    audible = scale > NUMERIC_FLOOR
    profiles = profiles[audible] / np.maximum(scale[audible, None], NUMERIC_FLOOR)
    anchors = anchors[audible]
    if len(anchors) < MIN_ANCHORS:
        return {"status": "too_few_audible_anchors", "anchors": len(anchors)}
    block_id = np.minimum(
        ((anchors - anchors[0]) / (anchors[-1] - anchors[0]) * TIME_BLOCKS).astype(int),
        TIME_BLOCKS - 1,
    )
    blocks = [profiles[block_id == b] for b in range(TIME_BLOCKS)]
    blocks = [b.mean(axis=0) for b in blocks if len(b) >= MIN_BLOCK_ANCHORS]
    if len(blocks) < MIN_BLOCKS:
        return {"status": "too_few_time_blocks", "anchors": len(anchors)}
    blocks = np.stack(blocks)
    curve = blocks.mean(axis=0)
    best = int(np.argmax(curve))
    lag = float(LAGS_MS[best])
    outside = np.abs(LAGS_MS - lag) > PEAK_EXCLUSION_MS
    contrast = float(curve[best] - np.median(curve))
    prominence = float(curve[best] - curve[outside].max())
    rng = np.random.default_rng(CALIBRATION_SEED)
    draws = rng.integers(0, len(blocks), (BOOTSTRAPS, len(blocks)))
    boot = LAGS_MS[blocks[draws].mean(axis=1).argmax(axis=1)]
    ci = np.quantile(boot, (0.025, 0.975)).tolist()
    reasons = []
    if abs(lag) >= LAGS_MS[-1] - EDGE_GUARD_MS:
        reasons.append("search_edge")
    if contrast < MIN_CONTRAST:
        reasons.append("weak_alignment")
    if prominence < MIN_PROMINENCE:
        reasons.append("competing_peak")
    if ci[1] - ci[0] > CI_MAX_WIDTH_MS:
        reasons.append("unstable_time_blocks")
    return {
        "status": "resolved" if not reasons else "unresolved",
        "reasons": reasons,
        "anchors": len(anchors),
        "lag_ms": lag,
        "ci_ms": ci,
        "contrast": contrast,
        "prominence": prominence,
        "curve": curve.tolist(),
        "block_lags_ms": LAGS_MS[blocks.argmax(axis=1)].tolist(),
        "score_at_zero": float(curve[np.flatnonzero(LAGS_MS == 0)[0]]),
        "score_at_20": float(curve[np.flatnonzero(LAGS_MS == 20)[0]]),
    }


def chart_measurement(chart, curves: dict, calibration: dict, duration_ms: float) -> dict:
    heads = head_times(chart)
    anchors = anchors_for(heads, duration_ms)
    fits = {
        str(size): lag_fit(*curves[size], anchors, calibration[str(size)]["correction_ms"])
        for size in FFT_SIZES
    }
    main, check = (fits[str(size)] for size in FFT_SIZES)
    resolved = all(f["status"] == "resolved" for f in fits.values())
    agreement = None
    if "lag_ms" in main and "lag_ms" in check:
        agreement = abs(main["lag_ms"] - check["lag_ms"])
        resolved &= agreement <= WINDOW_AGREE_MS
    points = [p for p in chart.timing_points if p.uninherited and p.beat_length_ms > 0]
    bpms = [MS_PER_MINUTE / p.beat_length_ms for p in points]
    return {
        "title": chart.metadata.title,
        "version": chart.metadata.version,
        "beatmap_id": chart.metadata.beatmap_id,
        "notes": len(chart.notes),
        "unique_heads": len(heads),
        "anchors": len(anchors),
        "ln_fraction": sum(n.note_type == NoteType.HOLD for n in chart.notes) / len(chart.notes),
        "bpm_min": min(bpms) if bpms else None,
        "bpm_max": max(bpms) if bpms else None,
        "redlines": len(points),
        "fits": fits,
        "window_difference_ms": agreement,
        "status": "resolved" if resolved else "unresolved",
        "lag_ms": main.get("lag_ms"),
        "near20": bool(resolved and NEAR20_MS[0] <= main["lag_ms"] <= NEAR20_MS[1]),
        "strict20": bool(resolved and STRICT20_MS[0] <= main["lag_ms"] <= STRICT20_MS[1]),
    }


def measure_group(task: dict) -> dict:
    result = {k: v for k, v in task.items() if k not in ("calibration", "root")}
    try:
        audio = Path(task["audio"])
        result["audio_sha256"] = digest(audio)
        decoded = decode_audio(audio)
        if decoded.timeline_broken or decoded.timeline_spliced:
            raise ValueError("audio timeline rejected by v1 decoder")
        result["duration_ms"] = decoded.t_audio_ms
        result["ffmpeg_version"] = decoded.ffmpeg_version
        curves = {size: attack_curve(decoded.y, size) for size in FFT_SIZES}
        charts = []
        for relpath in task["charts"]:
            path = Path(task["root"]) / relpath
            try:
                chart = parse_beatmap(path)
                row = chart_measurement(chart, curves, task["calibration"], decoded.t_audio_ms)
                charts.append({"relpath": relpath, "sha256": digest(path), **row})
            except Exception as exc:
                charts.append({"relpath": relpath, "status": "error", "error": str(exc)})
        result["results"] = charts
    except Exception as exc:
        result["error"] = str(exc)
        result["results"] = []
    return result


def select_groups(args, calibration: dict) -> tuple[list[dict], dict]:
    split = json.loads(args.split.read_text())
    population = {}
    for chart in split["charts"]:
        if chart["split"] not in ("train", "val"):
            continue
        population.setdefault(chart["group_id"], []).append(chart)
    rng = np.random.default_rng(args.seed)
    chosen = rng.permutation(sorted(population))[: args.groups]
    tasks = []
    for group in chosen:
        candidates = population[str(group)]
        order = rng.permutation(len(candidates))
        by_audio = {}
        for index in order:
            row = candidates[int(index)]
            path = args.root / row["relpath"]
            try:
                audio = chart_audio_path(path).resolve()
                by_audio.setdefault(str(audio), []).append(row["relpath"])
            except (OSError, ValueError):
                continue
        # One random audio within the group; up to three random difficulty charts.
        if not by_audio:
            tasks.append(
                {
                    "group_id": str(group),
                    "split": candidates[0]["split"],
                    "audio": "",
                    "charts": [],
                    "root": str(args.root),
                    "calibration": calibration,
                }
            )
            continue
        audio = list(by_audio)[int(rng.integers(len(by_audio)))]
        tasks.append(
            {
                "group_id": str(group),
                "split": candidates[0]["split"],
                "audio": audio,
                "charts": by_audio[audio][: args.charts_per_group],
                "root": str(args.root),
                "calibration": calibration,
            }
        )
    return tasks, {
        "groups": len(population),
        "charts": sum(len(rows) for rows in population.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".local-data/raw"))
    parser.add_argument(
        "--split", type=Path, default=Path(".local-data/splits/beatmapset_v1_drop2509435.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=400)
    parser.add_argument("--charts-per-group", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    calibration = calibrate()
    tasks, population = select_groups(args, calibration)
    protocol = {
        "population": population,
        "split_sha256": digest(args.split),
        "seed": args.seed,
        "code_sha256": digest(Path(__file__)),
        "calibration": calibration,
        "sign": "positive: audio spectral changes follow chart heads",
        "method": "positive sqrt-magnitude spectral flux; centred Hann; calibrated exact attacks",
        "lags_ms": LAGS_MS.tolist(),
        "hop_samples": HOP_SAMPLES,
        "fft_sizes": FFT_SIZES,
        "min_head_gap_ms": MIN_HEAD_GAP_MS,
        "min_anchors": MIN_ANCHORS,
        "time_blocks": TIME_BLOCKS,
        "min_block_anchors": MIN_BLOCK_ANCHORS,
        "max_anchors": MAX_ANCHORS,
        "min_blocks": MIN_BLOCKS,
        "bootstraps": BOOTSTRAPS,
        "ci_max_width_ms": CI_MAX_WIDTH_MS,
        "window_agree_ms": WINDOW_AGREE_MS,
        "min_contrast": MIN_CONTRAST,
        "min_prominence": MIN_PROMINENCE,
        "peak_exclusion_ms": PEAK_EXCLUSION_MS,
        "near20_ms": NEAR20_MS,
        "strict20_ms": STRICT20_MS,
        "selection": [{k: v for k, v in t.items() if k != "calibration"} for t in tasks],
        "limitations": [
            "attack proxy, not perceptual onset ground truth",
            "set-uniform sample; chart rows are not a chart-uniform sample",
            "80ms-isolated heads only; dense unmeasurable charts remain unresolved",
            "no test split; no model; no dataset or checkpoint mutation",
        ],
    }
    protocol_path = args.output / "protocol.json"
    freeze_json(protocol_path, protocol)
    path = args.output / "groups.jsonl"
    completed = {}
    if path.exists():
        completed, committed = read_records(path, key=lambda row: row["group_id"])
        if any(r["code_sha256"] != protocol["code_sha256"] for r in completed.values()):
            raise RuntimeError("existing results were produced by different diagnostic code")
        for task in tasks:
            row = completed.get(task["group_id"])
            if row is None:
                continue
            # Failed or incomplete groups are retried, never treated as verified measurements.
            if row.get("error") or any(not r.get("sha256") for r in row["results"]):
                del completed[task["group_id"]]
                continue
            audio = Path(task["audio"])
            if not matches_digest(audio, row.get("audio_sha256")):
                raise ValueError(f"frozen audio input changed: {audio}")
            if [r["relpath"] for r in row["results"]] != task["charts"]:
                raise ValueError(f"saved charts differ for group {task['group_id']}")
            for result in row["results"]:
                chart = Path(task["root"]) / result["relpath"]
                if not matches_digest(chart, result["sha256"]):
                    raise ValueError(f"frozen chart input changed: {chart}")
        if committed != path.stat().st_size:
            with path.open("r+b") as stream:
                stream.truncate(committed)
    pending = [t for t in tasks if t["group_id"] not in completed]
    with (
        ProcessPoolExecutor(max_workers=args.workers) as pool,
        path.open("a", encoding="utf-8") as output,
    ):
        for row in pool.map(measure_group, pending):
            row["code_sha256"] = protocol["code_sha256"]
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            completed[row["group_id"]] = row
            resolved = sum(r["status"] == "resolved" for r in row["results"])
            logger.info(
                "%d/%d %s charts=%d resolved=%d %s",
                len(completed),
                len(tasks),
                row["group_id"],
                len(row["results"]),
                resolved,
                row.get("error", ""),
            )


if __name__ == "__main__":
    main()
