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
from audio2map.osu.export import export_beatmap_notes
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.row_tokens import CanonicalTiming
from audio2map.training.inference import OverlapConfig, generate_chart_notes
from audio2map.training.model import AudioChartModel


def main() -> None:
    p = argparse.ArgumentParser(description="Infer v2 chart and export .osu")
    p.add_argument("--osu", type=str, required=True, help="template .osu (timing/metadata source)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default=None, help="output .osu path")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--window-bars", type=int, default=8)
    p.add_argument("--context-bars", type=int, default=4)
    p.add_argument("--keep-bars", type=int, default=4)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    osu_path = Path(args.osu)
    out_path = Path(args.out) if args.out else osu_path.with_name(osu_path.stem + " [Audio2Map].osu")

    timing = CanonicalTiming.from_beatmap(parse_beatmap(osu_path))
    meta = compute_chart_meta(osu_path, skip_msd=True)
    cond = build_cond_vec(meta)

    model = AudioChartModel.load_checkpoint(Path(args.checkpoint), device)
    overlap = OverlapConfig(
        window_bars=args.window_bars,
        context_bars=args.context_bars,
        keep_bars=args.keep_bars,
        future_bars=args.window_bars - args.context_bars - args.keep_bars,
    )

    notes = generate_chart_notes(
        model,
        audio_path=osu_path,
        timing=timing,
        cond_vec=cond,
        overlap=overlap,
        device=device,
        temperature=args.temperature,
    )

    export_beatmap_notes(osu_path, notes, out_path)
    summary = {
        "input": str(osu_path),
        "output": str(out_path),
        "note_count": len(notes),
        "checkpoint": args.checkpoint,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
