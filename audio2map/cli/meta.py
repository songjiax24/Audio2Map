"""``audio2map-meta``: batch-compute chart metadata (official SR, MSD, pattern stats)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from audio2map.cli.common import parse_args_with_config, setup_logging
from audio2map.dataset.filter import list_eligible_osu_paths
from audio2map.features.cond import CondVecError, build_cond_vec, compute_chart_meta
from audio2map.osu.parser import is_mania_4k_sections, parse_beatmap, read_osu_sections
from audio2map.osu.timing import summarize_beatmap_timing
from audio2map.utils.paths import chart_meta_dir, raw_dir


def _iter_mania_4k_osu(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*.osu")):
        try:
            sections = read_osu_sections(path)
        except OSError:
            continue
        if is_mania_4k_sections(sections):
            paths.append(path)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Compute per-chart difficulty metadata")
    p.add_argument("--config", type=str, default=None, help="YAML config (CLI flags win)")
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
        help="skip variable-BPM charts (recommended for the training pool)",
    )
    p.add_argument("--resume", action="store_true", help="skip paths already in output manifest")
    p.add_argument(
        "--eligible-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="only Phase-1 eligible charts (default; matches the training pool)",
    )
    p.add_argument(
        "--keep-errors",
        action="store_true",
        help="write rows even when cond_vec cannot be built (debug)",
    )
    args = parse_args_with_config(p)
    setup_logging()

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
                    done.add(str(Path(rec["osu_path"]).resolve()))
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

            if args.constant_bpm_only and not args.eligible_only:
                try:
                    if not summarize_beatmap_timing(parse_beatmap(path)).constant_bpm:
                        n_skip += 1
                        continue
                except Exception:
                    n_err += 1
                    continue

            meta = compute_chart_meta(path)
            if not args.keep_errors:
                try:
                    build_cond_vec(meta)
                except CondVecError:
                    n_err += 1
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
