"""``audio2map-grid``: precompute and audit tick-grid audio feature caches."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from audio2map.cli.common import parse_args_with_config, setup_logging
from audio2map.dataset.filter import check_osu_path
from audio2map.features.audio.grid import (
    cleanup_stale_grid_parts,
    grid_cache_paths,
    is_valid_audio_grid_cache,
    iter_set_timing_jobs,
    precompute_chart_audio_grid,
    remove_audio_grid_cache,
)
from audio2map.utils.paths import audio_grid_dir, raw_dir

logger = logging.getLogger(__name__)


def _ensure_disk_headroom(path: Path, *, min_gb: float = 2.0) -> None:
    free_gb = shutil.disk_usage(path).free / (1 << 30)
    if free_gb < min_gb:
        raise OSError(28, f"insufficient disk space: {free_gb:.1f}GB free, need >= {min_gb}GB")


def _run_precompute(args: argparse.Namespace) -> None:
    setup_logging()

    raw_root = raw_dir() if args.raw is None else Path(args.raw)
    out = audio_grid_dir() if args.out is None else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stale = cleanup_stale_grid_parts(out)
    if stale:
        logger.info("removed %d stale .part files from %s", stale, out)

    scan = iter_set_timing_jobs(raw_root, is_eligible=lambda p: check_osu_path(p).eligible)
    jobs = scan.jobs
    if args.limit is not None:
        jobs = jobs[: args.limit]

    ok = failed = skipped = invalid = 0
    for set_dir, timing, stem, audio_path in tqdm(jobs, desc="audio_grid"):
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
            precompute_chart_audio_grid(
                audio_path,
                timing,
                out,
                skip_existing=not args.no_skip_existing,
            )
            ok += 1
        except Exception as exc:
            logger.warning("%s: %s", set_dir, exc)
            failed += 1

    stats = {
        "ok": ok,
        "skipped_existing": skipped,
        "invalid_removed": invalid,
        "failed": failed,
        "enum_failed": scan.enum_failed,
        "unique_jobs": len(jobs),
        "out": str(out),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    raise SystemExit(1 if failed or scan.enum_failed else 0)


def audit_grid_dir(out_dir: Path, *, fix: bool = False) -> dict[str, int | str | list[str]]:
    stale = cleanup_stale_grid_parts(out_dir)
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for npy in sorted(out_dir.glob("*.npy")):
        stem = npy.stem
        seen.add(stem)
        json_path = npy.with_suffix(".json")
        if is_valid_audio_grid_cache(npy, json_path):
            valid.append(stem)
            continue
        invalid.append(stem)
        if fix:
            remove_audio_grid_cache(npy, json_path)

    for json_path in sorted(out_dir.glob("*.json")):
        stem = json_path.stem
        if stem in seen:
            continue
        invalid.append(stem)
        if fix:
            remove_audio_grid_cache(json_path.with_suffix(".npy"), json_path)

    return {
        "valid": len(valid),
        "invalid": len(invalid),
        "stale_parts_removed": stale,
        "invalid_stems": invalid[:20],
        "out": str(out_dir),
    }


def _run_audit(args: argparse.Namespace) -> None:
    out = audio_grid_dir() if args.out is None else Path(args.out)
    stats = audit_grid_dir(out, fix=args.fix)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    raise SystemExit(1 if stats["invalid"] else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None, help="YAML config (CLI flags win)")
    sub = p.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("precompute", help="build tick-grid caches → processed/audio_grid/")
    pre.add_argument("--raw", type=str, default=None, help="raw dataset root")
    pre.add_argument("--out", type=str, default=None, help="audio_grid output dir")
    pre.add_argument("--limit", type=int, default=None, help="max unique audio/timing grids")
    pre.add_argument("--no-skip-existing", action="store_true")
    pre.add_argument(
        "--min-free-gb",
        type=float,
        default=2.0,
        help="abort new grid writes when data disk free space drops below this (default 2GB)",
    )
    pre.set_defaults(func=_run_precompute)

    aud = sub.add_parser("audit", help="audit audio_grid cache integrity")
    aud.add_argument("--out", type=str, default=None, help="audio_grid directory")
    aud.add_argument(
        "--fix",
        action="store_true",
        help="remove invalid caches and stale .part files",
    )
    aud.set_defaults(func=_run_audit)

    args = parse_args_with_config(p)
    args.func(args)


if __name__ == "__main__":
    main()
