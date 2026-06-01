#!/usr/bin/env python3
"""ROW token / hold state frequency over eligible charts (stats only, no training changes)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audio2map.data.eligible_charts import list_eligible_osu_paths
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.row_tokens import (
    TOKEN_ROW_PREFIX,
    beatmap_to_row_tokens,
    CanonicalTiming,
    row_state_from_token,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    paths = list_eligible_osu_paths()
    import random

    rng = random.Random(args.seed)
    if args.limit < len(paths):
        paths = rng.sample(paths, args.limit)

    row_tokens = 0
    state_counts: Counter[int] = Counter()
    rows_with: Counter[str] = Counter()

    for path in paths:
        bm = parse_beatmap(path)
        timing = CanonicalTiming.from_beatmap(bm)
        toks = beatmap_to_row_tokens(bm, timing=timing)
        for t in toks:
            if not t.startswith(TOKEN_ROW_PREFIX) or t.count("_") < 2:
                continue
            if "initial" in t.lower():
                continue
            row_tokens += 1
            row = row_state_from_token(t)
            has = {1: False, 2: False, 3: False, 4: False}
            for s in row:
                if s in has:
                    has[s] = True
                    state_counts[s] += 1
            if has[1]:
                rows_with["tap"] += 1
            if has[2]:
                rows_with["hold_start"] += 1
            if has[3]:
                rows_with["hold_active"] += 1
            if has[4]:
                rows_with["hold_end"] += 1

    out = {
        "charts": len(paths),
        "row_event_tokens": row_tokens,
        "lane_state_token_occurrences": dict(state_counts),
        "rows_with_state": dict(rows_with),
        "rows_with_hold_start_pct": rows_with["hold_start"] / row_tokens if row_tokens else 0,
        "rows_with_hold_end_pct": rows_with["hold_end"] / row_tokens if row_tokens else 0,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
