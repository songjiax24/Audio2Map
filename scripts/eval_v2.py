#!/usr/bin/env python3
"""Evaluate v2 tokenization, teacher forcing, or full inference."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from audio2map.data.eligible_charts import list_eligible_osu_paths
from audio2map.eval.chart_eval import (
    eval_generate_window,
    eval_inference_chart,
    eval_roundtrip,
    eval_teacher_forcing,
    eval_window_notes,
)
from audio2map.eval.note_match import compare_note_lists_tick_tol


def _pick_paths(limit: int | None, seed: int) -> list[Path]:
    paths = list_eligible_osu_paths()
    if limit is not None and limit < len(paths):
        import random

        rng = random.Random(seed)
        paths = rng.sample(paths, limit)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Audio2Map v2 evaluation")
    p.add_argument(
        "--mode",
        choices=("roundtrip", "window", "teacher", "infer", "generate"),
        default="roundtrip",
    )
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--build-grid", action="store_true", help="build audio grid on the fly (slow)")
    p.add_argument("--osu", type=str, default=None, help="single chart path")
    p.add_argument("--start-bar", type=int, default=None, help="fixed window start (teacher/generate)")
    p.add_argument("--tick-tol", type=int, default=None, help="also report ±N tick tolerant F1")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.mode == "infer":
        if not args.checkpoint or not args.osu:
            raise SystemExit("infer mode requires --checkpoint and --osu")
        import torch

        from audio2map.data.cond_vec import build_cond_vec
        from audio2map.difficulty.chart_meta import compute_chart_meta
        from audio2map.training.model import load_checkpoint

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        path = Path(args.osu)
        model = load_checkpoint(Path(args.checkpoint), device)
        cond = build_cond_vec(compute_chart_meta(path, skip_msd=True))
        stats = eval_inference_chart(
            model,
            path,
            device=device,
            cond_vec=cond,
            build_grid_if_missing=args.build_grid,
        )
        out = {"charts": 1, "notes": stats.to_dict()}
        if args.tick_tol is not None:
            from audio2map.osu.parser import parse_beatmap
            from audio2map.osu.round_trip import quantize_notes
            from audio2map.osu.row_tokens import CanonicalTiming
            from audio2map.training.inference import generate_chart_notes, OverlapConfig

            timing = CanonicalTiming.from_beatmap(parse_beatmap(path))
            exp = quantize_notes(parse_beatmap(path).notes, timing)
            pred = generate_chart_notes(
                model,
                audio_path=path,
                timing=timing,
                cond_vec=cond,
                overlap=OverlapConfig(),
                device=device,
            )
            tol = compare_note_lists_tick_tol(pred, exp, timing, tick_tolerance=args.tick_tol)
            out[f"notes_tol{args.tick_tol}"] = tol.to_dict()
        print(json.dumps(out, indent=2))
        return

    if args.mode == "generate":
        if not args.checkpoint or not args.osu or args.start_bar is None:
            raise SystemExit("generate mode requires --checkpoint, --osu, --start-bar")
        import torch

        from audio2map.training.model import load_checkpoint

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = load_checkpoint(Path(args.checkpoint), device)
        agg, notes = eval_generate_window(
            model,
            Path(args.osu),
            device=device,
            start_bar=args.start_bar,
            build_grid_if_missing=args.build_grid,
        )
        print(json.dumps({**agg.to_dict(), "window_start_bar": args.start_bar}, indent=2))
        return

    paths = _pick_paths(args.limit, args.seed)
    if args.osu:
        paths = [Path(args.osu)]
    logging.info("evaluating %d charts mode=%s", len(paths), args.mode)

    if args.mode == "roundtrip":
        agg = eval_roundtrip(paths)
    elif args.mode == "window":
        agg = eval_window_notes(paths, seed=args.seed, build_grid_if_missing=args.build_grid)
    elif args.mode == "teacher":
        if not args.checkpoint:
            raise SystemExit("teacher mode requires --checkpoint")
        import torch

        from audio2map.training.model import load_checkpoint

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = load_checkpoint(Path(args.checkpoint), device)
        agg = eval_teacher_forcing(
            model,
            paths,
            device=device,
            seed=args.seed,
            build_grid_if_missing=args.build_grid,
            fixed_start_bar=args.start_bar,
        )
    else:
        raise SystemExit(f"unknown mode {args.mode}")

    print(json.dumps(agg.to_dict(), indent=2))


if __name__ == "__main__":
    main()
