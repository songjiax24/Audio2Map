#!/usr/bin/env python3
"""Batch-compute chart metadata: official SR, Etterna MSD, hold ratio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from audio2map.difficulty.chart_meta import compute_chart_meta
from audio2map.osu.mania import is_mania_4k_sections
from audio2map.osu.parser import parse_sections
from audio2map.data.eligible_charts import list_eligible_osu_paths
from audio2map.utils.paths import chart_meta_dir, raw_dir


def _iter_mania_4k_osu(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*.osu")):
        try:
            sections = parse_sections(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if is_mania_4k_sections(sections):
            paths.append(path)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Compute per-chart difficulty metadata")
    p.add_argument("--root", type=Path, default=raw_dir(), help="raw beatmap root")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output manifest.jsonl (default: DATA/chart_meta/manifest.jsonl)",
    )
    p.add_argument("--limit", type=int, default=None, help="max charts to process")
    p.add_argument(
        "--constant-bpm-only",
        action="store_true",
        help="skip variable-BPM charts (recommended for v1 training pool)",
    )
    p.add_argument("--skip-msd", action="store_true", help="only compute official SR + hold ratio")
    p.add_argument("--resume", action="store_true", help="skip paths already in output manifest")
    p.add_argument(
        "--eligible-only",
        action="store_true",
        help="only Phase-1 eligible charts (matches train_v2 pool)",
    )
    args = p.parse_args()

    out_path = args.out or (chart_meta_dir() / "manifest.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.resume and out_path.is_file():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    done.add(rec["osu_path"])
                except (json.JSONDecodeError, KeyError):
                    continue

    paths = list_eligible_osu_paths(args.root) if args.eligible_only else _iter_mania_4k_osu(args.root)
    if args.limit:
        paths = paths[: args.limit]

    n_ok = n_err = n_skip = 0
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as out_f:
        for path in tqdm(paths, desc="chart_meta", unit="chart"):
            path_str = str(path.resolve())
            if path_str in done:
                n_skip += 1
                continue

            meta = compute_chart_meta(path, skip_msd=args.skip_msd)
            if args.constant_bpm_only and not meta.constant_bpm:
                n_skip += 1
                continue

            out_f.write(json.dumps(meta.to_dict(), ensure_ascii=False) + "\n")
            out_f.flush()
            if meta.error:
                n_err += 1
            else:
                n_ok += 1

    print(f"\nmanifest: {out_path}")
    print(f"  ok={n_ok}  errors={n_err}  skipped={n_skip}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
