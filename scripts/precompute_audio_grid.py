#!/usr/bin/env python3
"""Precompute v2 tick-grid audio features → processed_v2/audio_grid/."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from audio2map.data.audio_grid import (
    cleanup_stale_grid_parts,
    grid_cache_paths,
    is_valid_audio_grid_cache,
    precompute_chart_audio_grid,
    remove_audio_grid_cache,
    resolve_grid_stem,
)
from audio2map.data.chart_filter import check_osu_path
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.row_tokens import CanonicalTiming
from audio2map.utils.paths import audio_grid_dir, raw_dir


def iter_set_timing_jobs(raw_root: Path) -> list[tuple[Path, CanonicalTiming, str]]:
    """One job per unique ``(audio_hash, bpm, offset)`` discovered from eligible charts."""
    from audio2map.audio.loader import find_audio_file

    jobs: list[tuple[Path, CanonicalTiming, str]] = []
    seen_stems: set[str] = set()

    for set_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        osu_paths = sorted(set_dir.glob("*.osu"))
        if not osu_paths:
            continue
        for osu_path in osu_paths:
            if not check_osu_path(osu_path).eligible:
                continue
            try:
                timing = CanonicalTiming.from_beatmap(parse_beatmap(osu_path))
                audio_path = find_audio_file(set_dir)
                stem = resolve_grid_stem(audio_path, timing)
                if stem in seen_stems:
                    break
                seen_stems.add(stem)
                jobs.append((set_dir, timing, stem))
                break
            except Exception:
                continue
    return jobs


def _ensure_disk_headroom(path: Path, *, min_gb: float = 2.0) -> None:
    free_gb = shutil.disk_usage(path).free / (1 << 30)
    if free_gb < min_gb:
        raise OSError(28, f"insufficient disk space: {free_gb:.1f}GB free, need >= {min_gb}GB")


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute v2 audio tick-grid caches")
    p.add_argument("--raw", type=str, default=None, help="raw dataset root")
    p.add_argument("--out", type=str, default=None, help="audio_grid output dir")
    p.add_argument("--limit", type=int, default=None, help="max unique audio/timing grids")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument(
        "--min-free-gb",
        type=float,
        default=2.0,
        help="abort new grid writes when data disk free space drops below this (default 2GB)",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    raw_root = raw_dir() if args.raw is None else Path(args.raw)
    out = audio_grid_dir() if args.out is None else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stale = cleanup_stale_grid_parts(out)
    if stale:
        logging.info("removed %d stale .part files from %s", stale, out)

    jobs = iter_set_timing_jobs(raw_root)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    ok = failed = skipped = invalid = 0
    for set_dir, timing, stem in tqdm(jobs, desc="audio_grid"):
        npy_path, json_path = grid_cache_paths(out, stem)
        if not args.no_skip_existing:
            if is_valid_audio_grid_cache(npy_path, json_path):
                skipped += 1
                continue
            if npy_path.is_file() or json_path.is_file():
                invalid += 1
                remove_audio_grid_cache(npy_path, json_path)
        try:
            _ensure_disk_headroom(out, min_gb=args.min_free_gb)
            result = precompute_chart_audio_grid(
                set_dir,
                timing,
                out,
                skip_existing=not args.no_skip_existing,
            )
            if result is None:
                failed += 1
            else:
                ok += 1
        except Exception as exc:
            logging.warning("%s: %s", set_dir, exc)
            failed += 1

    stats = {
        "ok": ok,
        "skipped_existing": skipped,
        "invalid_removed": invalid,
        "failed": failed,
        "unique_jobs": len(jobs),
        "out": str(out),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
