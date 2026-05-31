#!/usr/bin/env python3
"""Validate .osu parsing and sparse event tokenisation."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from audio2map.osu import (
    beatmap_to_events,
    count_event_types,
    event_ar_tokens,
    events_to_frames,
    parse_beatmap,
    summarize_beatmap_timing,
)
from audio2map.osu.events import EventType
from audio2map.utils.paths import raw_dir


def iter_osu_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.osu"))


def validate(paths: list[Path], *, sample_print: int = 3, seed: int = 0) -> int:
    ok = fail = 0
    ok_paths: list[Path] = []
    fail_reasons: Counter[str] = Counter()
    type_totals: Counter[str] = Counter()
    total_events = 0
    total_dense_cells = 0
    constant_bpm = variable_bpm = 0

    for path in paths:
        try:
            bm = parse_beatmap(path)
            events = beatmap_to_events(bm)
            counts = count_event_types(events)
            for k, v in counts.items():
                if k != "total":
                    type_totals[k] += v
            total_events += counts["total"]
            grid = events_to_frames(events, duration_ms=bm.duration_ms + 500)
            total_dense_cells += grid.num_frames * 4
            ts = summarize_beatmap_timing(bm)
            if ts.constant_bpm:
                constant_bpm += 1
            elif ts.n_uninherited > 0:
                variable_bpm += 1
            ok += 1
            ok_paths.append(path)
        except Exception as exc:
            fail += 1
            fail_reasons[type(exc).__name__ + ": " + str(exc)[:80]] += 1

    print(f"\n{'osu + sparse event validation':^44}\n{'-' * 44}")
    print(f"  files total       {len(paths):>6}")
    print(f"  parsed ok         {ok:>6}")
    print(f"  failed            {fail:>6}")
    if fail_reasons:
        print("  failure samples:")
        for reason, count in fail_reasons.most_common(5):
            print(f"    {count:>5}x  {reason}")
    print(f"\n  sparse events")
    print(f"    tap             {type_totals['tap']:>12}")
    print(f"    hold            {type_totals['hold']:>12}")
    print(f"    total           {total_events:>12}")
    if total_dense_cells:
        sparsity = 100.0 * (1 - total_events / total_dense_cells)
        print(f"    vs dense cells  {total_dense_cells:>12}  ({sparsity:.1f}% saved)")
    if ok:
        print(f"\n  timing (of parsed)")
        print(f"    constant BPM    {constant_bpm:>6}")
        print(f"    variable BPM    {variable_bpm:>6}")

    if ok_paths and sample_print > 0:
        rng = random.Random(seed)
        samples = rng.sample(ok_paths, min(sample_print, len(ok_paths)))
        print("\n  random samples:")
        for p in samples:
            bm = parse_beatmap(p)
            events = beatmap_to_events(bm)
            ts = summarize_beatmap_timing(bm)
            print(f"    {p.parent.name}/{p.name}")
            print(
                f"      events={len(events)} od={bm.metadata.overall_difficulty}"
                f" bpm={ts.bpm_primary} const={ts.constant_bpm}"
            )
            for e in events[:4]:
                tok = event_ar_tokens(e)
                if e.type_id == EventType.HOLD:
                    print(f"      frame={tok[0]} col={tok[1]} HOLD end={tok[3]}")
                else:
                    print(f"      frame={tok[0]} col={tok[1]} TAP")

    return 0 if fail == 0 else 1


def main() -> None:
    p = argparse.ArgumentParser(description="Validate mania 4K parse + sparse events")
    p.add_argument("--root", type=Path, default=raw_dir())
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample-print", type=int, default=3)
    args = p.parse_args()

    paths = iter_osu_files(args.root)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        print(f"no .osu files under {args.root}", file=sys.stderr)
        raise SystemExit(1)

    raise SystemExit(validate(paths, sample_print=args.sample_print))


if __name__ == "__main__":
    main()
