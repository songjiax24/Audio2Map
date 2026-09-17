"""``audio2map-infer``: regenerate a chart from its own audio + cond, export ``.osu``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audio2map.cli.common import parse_args_with_config, setup_logging
from audio2map.features.cond import build_cond_vec, compute_chart_meta
from audio2map.generate.overlap import DecodeConfig, GenerationRangeConfig, OverlapConfig
from audio2map.generate.service import generate_chart
from audio2map.grid import CanonicalTiming
from audio2map.model.model import load_checkpoint
from audio2map.osu.parser import parse_beatmap
from audio2map.utils.paths import audio_grid_dir

_DEFAULT_OVERLAP = OverlapConfig()
_DEFAULT_DECODE = DecodeConfig()
_LOCKED_OVERLAP = {
    "window_bars": _DEFAULT_OVERLAP.window_bars,
    "context_bars": _DEFAULT_OVERLAP.context_bars,
    "keep_bars": _DEFAULT_OVERLAP.keep_bars,
    "future_bars": _DEFAULT_OVERLAP.future_bars,
    "max_seq_len": _DEFAULT_OVERLAP.max_seq_len,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Infer chart and export .osu")
    p.add_argument("--config", type=str, default=None, help="YAML config (e.g. configs/data/v2.yaml)")
    p.add_argument("--osu", type=str, required=True, help=".osu for timing, metadata, cond, and audio folder")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--grid-dir",
        type=str,
        default=None,
        help="audio_grid cache dir (must match training; default: processed/audio_grid)",
    )
    p.add_argument("--out", type=str, default=None, help="output .osu path")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--temperature", type=float, default=_DEFAULT_DECODE.temperature)
    p.add_argument("--top-p", type=float, default=_DEFAULT_DECODE.top_p)
    p.add_argument("--top-k", type=int, default=_DEFAULT_DECODE.top_k)
    p.add_argument("--greedy", action="store_true", help="temperature=0 greedy (debug only)")
    p.add_argument(
        "--range-mode",
        choices=("audio_full", "reference_chart"),
        default="audio_full",
        help="audio_full=whole audio (default); reference_chart=eval-only chart_end+margin",
    )
    p.add_argument("--post-margin-bars", type=int, default=GenerationRangeConfig().post_margin_bars)
    p.add_argument(
        "--single-window",
        action="store_true",
        help="debug: one window, no overlap stitch",
    )
    p.add_argument("--start-bar", type=int, default=None, help="with --single-window")
    return p


def _check_args(args: argparse.Namespace) -> None:
    if args.start_bar is not None and not args.single_window:
        raise SystemExit("--start-bar requires --single-window")


def main() -> None:
    args = parse_args_with_config(build_parser(), locked=_LOCKED_OVERLAP)
    _check_args(args)

    setup_logging()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    osu_path = Path(args.osu)
    out_path = Path(args.out) if args.out else osu_path.with_name(osu_path.stem + " [Audio2Map].osu")

    beatmap = parse_beatmap(osu_path)
    timing = CanonicalTiming.from_beatmap(beatmap)
    meta = compute_chart_meta(osu_path)
    cond = build_cond_vec(meta)

    model = load_checkpoint(Path(args.checkpoint), device)

    grid_dir = Path(args.grid_dir) if args.grid_dir else audio_grid_dir()
    temperature = 0.0 if args.greedy else args.temperature
    result = generate_chart(
        model,
        audio=osu_path,
        source_osu=osu_path,
        reference_notes=beatmap.notes,
        timing=timing,
        cond_vec=cond,
        grid_dir=grid_dir,
        overlap=OverlapConfig(),
        range_cfg=GenerationRangeConfig(
            mode=args.range_mode,
            post_margin_bars=args.post_margin_bars,
        ),
        device=device,
        decode=DecodeConfig(
            temperature=temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        ),
        single_window=args.single_window,
        single_window_start_bar=args.start_bar,
        out_osu=out_path,
    )

    summary = {
        "input": str(osu_path),
        "output": str(out_path),
        "note_count": len(result.notes),
        "checkpoint": args.checkpoint,
        "grid_dir": str(grid_dir),
        "generation": result.report.to_dict(),
        "decode": {
            "temperature": temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
