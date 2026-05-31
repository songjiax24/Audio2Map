#!/usr/bin/env python3
"""Preprocess raw beatmap sets → mel + sparse events + manifest."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from audio2map.data.preprocess import PreprocessConfig, run_preprocess
from audio2map.utils.paths import processed_dir, raw_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess raw osu!mania 4K data for training")
    p.add_argument("--raw", type=str, default=None, help="raw dataset root (default: AUDIO2MAP_DATA_ROOT/raw)")
    p.add_argument("--out", type=str, default=None, help="output dir (default: .../processed)")
    p.add_argument("--hop-ms", type=int, default=10, help="chart frame / mel hop (ms)")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--mel-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--limit", type=int, default=None, help="process at most N sets")
    p.add_argument("--no-skip-existing", action="store_true", help="reprocess even if .npz exists")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = PreprocessConfig(
        hop_ms=args.hop_ms,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        mel_dtype=args.mel_dtype,
    )
    raw_root = raw_dir() if args.raw is None else __import__("pathlib").Path(args.raw)
    out = processed_dir() if args.out is None else __import__("pathlib").Path(args.out)

    stats = run_preprocess(
        raw_root,
        out,
        cfg,
        limit=args.limit,
        skip_existing=not args.no_skip_existing,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    raise SystemExit(1 if stats["failed"] else 0)


if __name__ == "__main__":
    main()
