#!/usr/bin/env python3
"""Statistics for 16-bar window ROW token lengths (formal max_decoder_len sizing).

Default mode (~2–4 min on 12k charts): random windows only.
Use ``--scan-chart-range`` for per-chart max over note bars (medium).
Use ``--scan-all-starts`` for full audio span (very slow).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from audio2map.data.audio_grid import AudioGridMeta, grid_cache_paths, is_valid_audio_grid_cache
from audio2map.data.audio_grid import resolve_grid_stem
from audio2map.data.eligible_charts import list_eligible_osu_paths
from audio2map.data.window_sampler import audio_bar_range_from_duration
from audio2map.osu.grid_config import WINDOW_BARS
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.row_tokens import (
    CanonicalTiming,
    beatmap_to_window_tokens,
    build_vocab,
    chart_event_bar_range,
)
from audio2map.training.config import DECODER_LEN_REPORT_LIMITS
from audio2map.utils.paths import audio_grid_dir, raw_dir


def _audio_bar_range(path: Path, timing: CanonicalTiming, *, grid_dir: Path) -> tuple[int, int] | None:
    from audio2map.audio.loader import find_audio_file
    import json

    try:
        audio_path = find_audio_file(path.parent)
        stem = resolve_grid_stem(audio_path, timing)
        npy, json_path = grid_cache_paths(grid_dir, stem)
        if is_valid_audio_grid_cache(npy, json_path):
            meta = AudioGridMeta.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
            return audio_bar_range_from_duration(meta.duration_ms, timing)
    except Exception:
        pass
    return None


def _window_token_len(beatmap, timing: CanonicalTiming, start_bar: int) -> int:
    return len(
        beatmap_to_window_tokens(
            beatmap,
            start_bar=start_bar,
            window_bars=WINDOW_BARS,
            timing=timing,
        )
    )


def _chart_range_starts(
    beatmap,
    timing: CanonicalTiming,
    audio_start: int,
    audio_end: int,
) -> range:
    """Bar starts where a 16-bar window can overlap chart notes (much smaller than full audio)."""
    if not beatmap.notes:
        return range(0)
    chart_start, chart_end = chart_event_bar_range(beatmap.notes, timing)
    lo = max(audio_start, chart_start - WINDOW_BARS + 1)
    hi = min(audio_end - WINDOW_BARS, chart_end)
    if lo > hi:
        return range(0)
    return range(lo, hi + 1)


def _percentiles(values: list[int], ps: tuple[float, ...]) -> dict[str, float]:
    if not values:
        return {f"p{p:g}": float("nan") for p in ps}
    arr = np.array(values, dtype=np.float64)
    return {f"p{p:g}": float(np.percentile(arr, p)) for p in ps}


def _overflow_ratio(values: list[int], limit: int) -> float:
    if not values:
        return float("nan")
    return sum(1 for x in values if x <= limit) / len(values)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit-charts", type=int, default=None)
    p.add_argument(
        "--samples-per-chart",
        type=int,
        default=8,
        help="random windows per chart (matches train samples_per_chart default)",
    )
    p.add_argument(
        "--scan-chart-range",
        action="store_true",
        help="scan all starts overlapping chart notes (medium; good max estimate)",
    )
    p.add_argument(
        "--scan-all-starts",
        action="store_true",
        help="scan every bar-aligned start over full audio (very slow)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    paths = list_eligible_osu_paths(raw_dir())
    if args.limit_charts:
        paths = paths[: args.limit_charts]

    rng = random.Random(args.seed)
    grid_dir = audio_grid_dir()
    sampled_lens: list[int] = []
    max_per_chart: list[int] = []
    charts_ok = 0
    windows_scanned = 0

    for path in tqdm(paths, desc="window_token_len", unit="chart"):
        try:
            beatmap = parse_beatmap(path)
            timing = CanonicalTiming.from_beatmap(beatmap)
            bar_range = _audio_bar_range(path, timing, grid_dir=grid_dir)
            if bar_range is None:
                continue
            audio_start, audio_end = bar_range
            if audio_end - audio_start < WINDOW_BARS:
                continue
            charts_ok += 1

            chart_max = 0
            if args.scan_all_starts:
                scan = range(audio_start, audio_end - WINDOW_BARS + 1)
            elif args.scan_chart_range:
                scan = _chart_range_starts(beatmap, timing, audio_start, audio_end)
            else:
                scan = range(0)

            for start in scan:
                n = _window_token_len(beatmap, timing, start)
                chart_max = max(chart_max, n)
                windows_scanned += 1
            if scan:
                max_per_chart.append(chart_max)

            valid_starts = range(audio_start, audio_end - WINDOW_BARS + 1)
            picks = min(args.samples_per_chart, len(valid_starts))
            for start in rng.sample(list(valid_starts), picks):
                sampled_lens.append(_window_token_len(beatmap, timing, start))
                windows_scanned += 1
                if not scan:
                    chart_max = max(chart_max, sampled_lens[-1])
            if not scan:
                max_per_chart.append(chart_max)
        except Exception:
            continue

    pct = _percentiles(sampled_lens, (50, 90, 95, 99, 99.5))
    max_stats = _percentiles(max_per_chart, (50, 90, 95, 99, 99.5))

    report = {
        "charts_scanned": charts_ok,
        "window_bars": WINDOW_BARS,
        "vocab_size": len(build_vocab()),
        "samples_per_chart": args.samples_per_chart,
        "scan_chart_range": args.scan_chart_range,
        "scan_all_starts": args.scan_all_starts,
        "sampled_windows": len(sampled_lens),
        "windows_scanned_total": windows_scanned,
        "sampled_token_len": {**pct, "max": max(sampled_lens) if sampled_lens else 0},
        "per_chart_max_token_len": {**max_stats, "max": max(max_per_chart) if max_per_chart else 0},
        "overflow_coverage_sampled": {
            str(limit): _overflow_ratio(sampled_lens, limit) for limit in DECODER_LEN_REPORT_LIMITS
        },
        "overflow_coverage_per_chart_max": {
            str(limit): _overflow_ratio(max_per_chart, limit) for limit in DECODER_LEN_REPORT_LIMITS
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    basis = max_per_chart if max_per_chart else sampled_lens
    recommended = max(DECODER_LEN_REPORT_LIMITS)
    for limit in DECODER_LEN_REPORT_LIMITS:
        if _overflow_ratio(basis, limit) >= 0.995:
            recommended = limit
            break
    print(
        f"\nrecommended MAX_DECODER_LEN={recommended} "
        f"(covers {100 * _overflow_ratio(basis, recommended):.2f}% of per-chart max basis)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
