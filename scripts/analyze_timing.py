#!/usr/bin/env python3
"""Analyze BPM / timing characteristics of the raw dataset."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from audio2map.osu.timing import parse_timing_points_only, summarize_timing
from audio2map.utils.paths import raw_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze timing / BPM in raw .osu files")
    p.add_argument("--root", type=Path, default=raw_dir())
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    paths = sorted(args.root.rglob("*.osu"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no .osu under {args.root}", file=sys.stderr)
        raise SystemExit(1)

    constant = variable = no_tp = 0
    seg_hist: Counter[int] = Counter()
    bpm_hist: Counter[float] = Counter()
    errors = 0

    for path in tqdm(paths, desc="timing", unit="file"):
        try:
            tps = parse_timing_points_only(path)
            s = summarize_timing(tps)
            if s.n_uninherited == 0:
                no_tp += 1
            elif s.constant_bpm:
                constant += 1
                if s.bpm_primary is not None:
                    bpm_hist[round(s.bpm_primary)] += 1
            else:
                variable += 1
                seg_hist[len(s.bpm_segments)] += 1
        except Exception:
            errors += 1

    total = len(paths)
    w = 44
    print(f"\n{'timing analysis':^{w}}\n{'-' * w}")
    print(f"  files scanned       {total:>6}")
    print(f"  constant BPM        {constant:>6}  ({100*constant/total:.1f}%)")
    print(f"  variable BPM        {variable:>6}  ({100*variable/total:.1f}%)")
    print(f"  no timing points    {no_tp:>6}  ({100*no_tp/total:.1f}%)")
    print(f"  errors              {errors:>6}")
    if seg_hist:
        print(f"\n  variable BPM segment counts (top):")
        for n, c in seg_hist.most_common(5):
            print(f"    {n} distinct BPM(s): {c} maps")
    if bpm_hist:
        print(f"\n  common constant BPM (top 8):")
        for bpm, c in bpm_hist.most_common(8):
            print(f"    {bpm:>6.0f} BPM: {c} maps")
    print()


if __name__ == "__main__":
    main()
