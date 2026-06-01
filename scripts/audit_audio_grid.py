#!/usr/bin/env python3
"""Audit processed_v2/audio_grid caches for integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio2map.data.audio_grid import (
    cleanup_stale_grid_parts,
    is_valid_audio_grid_cache,
    remove_audio_grid_cache,
)
from audio2map.utils.paths import audio_grid_dir


def audit_grid_dir(out_dir: Path, *, fix: bool = False) -> dict[str, int | list[str]]:
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


def main() -> None:
    p = argparse.ArgumentParser(description="Audit v2 audio_grid cache integrity")
    p.add_argument("--out", type=str, default=None, help="audio_grid directory")
    p.add_argument(
        "--fix",
        action="store_true",
        help="remove invalid caches and stale .part files",
    )
    args = p.parse_args()

    out = audio_grid_dir() if args.out is None else Path(args.out)
    stats = audit_grid_dir(out, fix=args.fix)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    raise SystemExit(1 if stats["invalid"] else 0)


if __name__ == "__main__":
    main()
