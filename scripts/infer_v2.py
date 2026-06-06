#!/usr/bin/env python3
"""Run v2 inference and export ``.osu``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from audio2map.data.cond_vec import build_cond_vec
from audio2map.difficulty.chart_meta import compute_chart_meta
from audio2map.eval.degeneracy import analyze_chart_degeneracy
from audio2map.osu.export import export_beatmap_notes
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.row_tokens import CanonicalTiming
from audio2map.training.config import MAX_DECODER_LEN
from audio2map.training.inference import (
    GenerationRangeConfig,
    OverlapConfig,
    generate_chart_notes,
)
from audio2map.training.model import load_checkpoint
from audio2map.utils.paths import audio_grid_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Infer v2 chart and export .osu")
    p.add_argument("--osu", type=str, required=True, help="template .osu (timing/metadata source)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--grid-dir",
        type=str,
        default=None,
        help="audio_grid cache dir (must match training; default: processed_v2/audio_grid)",
    )
    p.add_argument("--out", type=str, default=None, help="output .osu path")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--greedy", action="store_true", help="temperature=0 greedy (debug only)")
    p.add_argument("--window-bars", type=int, default=16)
    p.add_argument("--context-bars", type=int, default=8)
    p.add_argument("--keep-bars", type=int, default=4)
    p.add_argument("--future-bars", type=int, default=4)
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_DECODER_LEN,
        help="max decoder tokens per window (default: training MAX_DECODER_LEN)",
    )
    p.add_argument(
        "--range-mode",
        choices=("audio_full", "reference_chart"),
        default="audio_full",
        help="audio_full=whole mp3 (default); reference_chart=eval-only chart_end+margin",
    )
    p.add_argument("--post-margin-bars", type=int, default=4)
    p.add_argument(
        "--single-window",
        action="store_true",
        help="debug: one window, no overlap stitch",
    )
    p.add_argument("--start-bar", type=int, default=None, help="with --single-window")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    osu_path = Path(args.osu)
    out_path = Path(args.out) if args.out else osu_path.with_name(osu_path.stem + " [Audio2Map].osu")

    beatmap = parse_beatmap(osu_path)
    timing = CanonicalTiming.from_beatmap(beatmap)
    meta = compute_chart_meta(osu_path)
    cond = build_cond_vec(meta)

    model = load_checkpoint(Path(args.checkpoint), device)
    if args.context_bars + args.keep_bars + args.future_bars != args.window_bars:
        raise SystemExit(
            "context_bars + keep_bars + future_bars must equal window_bars "
            f"({args.context_bars}+{args.keep_bars}+{args.future_bars}!={args.window_bars})"
        )
    overlap = OverlapConfig(
        window_bars=args.window_bars,
        context_bars=args.context_bars,
        keep_bars=args.keep_bars,
        future_bars=args.future_bars,
        max_seq_len=args.max_seq_len,
    )
    range_cfg = GenerationRangeConfig(
        mode=args.range_mode,
        post_margin_bars=args.post_margin_bars,
    )
    temperature = 0.0 if args.greedy else args.temperature

    grid_dir = Path(args.grid_dir) if args.grid_dir else audio_grid_dir()

    notes, report = generate_chart_notes(
        model,
        audio_path=osu_path,
        timing=timing,
        cond_vec=cond,
        grid_dir=grid_dir,
        overlap=overlap,
        range_cfg=range_cfg,
        device=device,
        temperature=temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        single_window=args.single_window,
        single_window_start_bar=args.start_bar,
        return_report=True,
    )

    export_beatmap_notes(osu_path, notes, out_path)
    degeneracy = analyze_chart_degeneracy(notes, timing)
    summary = {
        "input": str(osu_path),
        "output": str(out_path),
        "note_count": len(notes),
        "checkpoint": args.checkpoint,
        "grid_dir": str(grid_dir),
        "architecture": getattr(model, "architecture", "unknown"),
        "audio_pooling": getattr(model, "audio_pooling", "unknown"),
        "generation": report.to_dict(),
        "degeneracy": degeneracy.to_dict(),
        "decode": {
            "temperature": temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
