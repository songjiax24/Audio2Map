#!/usr/bin/env python3
"""Phase 1 v2 dataset statistics."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from audio2map.dataset.filter import FilterReason, REQUIRED_METER, check_osu_path
from audio2map.osu.parser import (
    InvalidTimingPointError,
    is_mania_4k_sections,
    parse_beatmap,
    parse_timing_points,
    read_osu_sections,
)
from audio2map.tokens import (
    build_vocab,
    encode_notes,
)
from audio2map.osu.timing import summarize_timing
from audio2map.utils.paths import raw_dir


def _pct(values: list[float], ps=(50, 90, 95, 99)) -> dict[str, float]:
    if not values:
        return {f"p{int(p)}": float("nan") for p in ps}
    arr = np.array(values, dtype=np.float64)
    return {f"p{int(p)}": float(np.percentile(arr, p)) for p in ps}


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze Phase 1 v2 dataset")
    p.add_argument("--root", type=Path, default=raw_dir())
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--window-bars", type=int, default=16)
    args = p.parse_args()

    paths = sorted(args.root.rglob("*.osu"))
    if args.limit:
        paths = paths[: args.limit]

    total = len(paths)
    mania_4k = constant_bpm = eligible = 0
    removed = Counter()
    meter_not_4_hist: Counter[int] = Counter()
    orig_bpm_hist: Counter[float] = Counter()
    canon_bpm_hist: Counter[float] = Counter()
    scale_exp_hist: Counter[int] = Counter()
    token_lens: list[int] = []
    note_counts: list[int] = []
    hold_counts: list[int] = []
    durations_bars: list[int] = []
    encode_err = 0

    for path in tqdm(paths, desc="analyze", unit="file"):
        try:
            sections = read_osu_sections(path)
        except OSError:
            removed["read_error"] += 1
            continue
        if not is_mania_4k_sections(sections):
            removed["not_4k"] += 1
            continue
        mania_4k += 1

        try:
            timing = summarize_timing(parse_timing_points(sections.get("[TimingPoints]", [])))
        except InvalidTimingPointError:
            removed[FilterReason.INVALID_TIMING_POINTS.value] += 1
            continue
        if timing.constant_bpm:
            constant_bpm += 1

        result = check_osu_path(path)
        if not result.eligible:
            key = result.reason.value if result.reason else "unknown"
            removed[key] += 1
            if result.reason == FilterReason.METER_NOT_4 and timing.meter is not None:
                meter_not_4_hist[timing.meter] += 1
            continue

        eligible += 1
        try:
            bm = parse_beatmap(path)
            from audio2map.grid import CanonicalTiming
            from audio2map.osu.schema import NoteType

            ct = CanonicalTiming.from_beatmap(bm)
            orig_bpm_hist[round(ct.original_bpm)] += 1
            canon_bpm_hist[round(ct.canonical_bpm)] += 1
            scale_exp_hist[ct.bpm_scale_exp] += 1
            tokens = encode_notes(bm.notes, ct)
            token_lens.append(len(tokens))
            note_counts.append(bm.note_count)
            hold_counts.append(sum(1 for n in bm.notes if n.note_type == NoteType.HOLD))
            if bm.notes:
                from audio2map.grid import (
                    all_event_ticks,
                    chart_event_bar_range,
                )

                ticks = all_event_ticks(bm.notes, ct)
                start_bar, end_bar = chart_event_bar_range(bm.notes, ct)
                durations_bars.append(end_bar - start_bar)
        except Exception:
            encode_err += 1

    w = 56
    print(f"\n{'Phase 1 v2 dataset analysis':^{w}}\n{'-' * w}")
    print(f"  total charts                 {total:>6}")
    print(f"  mania 4K                     {mania_4k:>6}")
    print(f"  constant BPM                 {constant_bpm:>6}")
    print(f"  eligible (meter=4, etc.)     {eligible:>6}")
    print(f"  tokenize errors              {encode_err:>6}")
    print()
    print(f"  removed: variable BPM        {removed[FilterReason.VARIABLE_BPM.value]:>6}")
    print(f"  removed: variable meter      {removed[FilterReason.VARIABLE_METER.value]:>6}")
    print(f"  removed: meter != {REQUIRED_METER}            {removed[FilterReason.METER_NOT_4.value]:>6}  <--")
    print(f"  removed: no timing           {removed[FilterReason.NO_TIMING.value]:>6}")
    print(f"  removed: no notes            {removed[FilterReason.NO_NOTES.value]:>6}")
    print(f"  removed: invalid hit objects {removed[FilterReason.INVALID_HIT_OBJECTS.value]:>6}")
    print(f"  removed: invalid timing      {removed[FilterReason.INVALID_TIMING_POINTS.value]:>6}")
    print(f"  removed: invalid grid        {removed[FilterReason.INVALID_GRID.value]:>6}")

    if meter_not_4_hist:
        print("\n  meter != 4 breakdown:")
        for m, c in sorted(meter_not_4_hist.items()):
            print(f"    meter={m}: {c}")

    print(f"\n  vocab size                   {len(build_vocab()):>6}")

    if orig_bpm_hist:
        print("\n  original BPM (top 6):")
        for bpm, c in orig_bpm_hist.most_common(6):
            print(f"    {bpm:>6.0f}: {c}")
    if canon_bpm_hist:
        print("\n  canonical BPM (top 6):")
        for bpm, c in canon_bpm_hist.most_common(6):
            print(f"    {bpm:>6.0f}: {c}")
    if scale_exp_hist:
        print("\n  bpm_scale_exp:")
        for e, c in sorted(scale_exp_hist.items()):
            print(f"    {e:+d}: {c}")

    def _print_dist(name: str, values: list[float]) -> None:
        if not values:
            return
        pct = _pct(values)
        print(f"\n  {name} (n={len(values)}):")
        print(f"    min   {min(values):>10.4f}")
        print(f"    max   {max(values):>10.4f}")
        print(f"    mean  {sum(values)/len(values):>10.4f}")
        for k, v in pct.items():
            print(f"    {k}  {v:>10.4f}")

    _print_dist("token length", [float(x) for x in token_lens])
    _print_dist("note count", [float(x) for x in note_counts])
    _print_dist("hold count", [float(x) for x in hold_counts])
    _print_dist("chart duration (bars)", [float(x) for x in durations_bars])

    wb = args.window_bars
    cov = {
        1024: sum(1 for x in token_lens if x <= 1024),
        2048: sum(1 for x in token_lens if x <= 2048),
        3072: sum(1 for x in token_lens if x <= 3072),
        4096: sum(1 for x in token_lens if x <= 4096),
    }
    if token_lens:
        n = len(token_lens)
        print(f"\n  max_token_len coverage (full chart tokens, n={n}):")
        for lim, cnt in cov.items():
            print(f"    <={lim:<5} {cnt:>6} ({100*cnt/n:.1f}%)")
    print()


if __name__ == "__main__":
    main()
