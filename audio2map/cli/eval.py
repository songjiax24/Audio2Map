"""``audio2map-eval``: teacher-forcing / inference eval."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import torch

from audio2map.cli.common import parse_args_with_config, setup_logging
from audio2map.dataset.filter import list_eligible_osu_paths
from audio2map.eval.chart_eval import eval_inference_chart, eval_teacher_forcing
from audio2map.features.cond import build_cond_vec, compute_chart_meta
from audio2map.generate.overlap import DecodeConfig, GenerationRangeConfig
from audio2map.model.model import load_checkpoint

logger = logging.getLogger(__name__)

_DEFAULT_DECODE = DecodeConfig()


def _pick_paths(limit: int | None, seed: int) -> list[Path]:
    paths = list_eligible_osu_paths()
    if limit is not None and limit < len(paths):
        rng = random.Random(seed)
        paths = rng.sample(paths, limit)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Audio2Map evaluation")
    p.add_argument("--config", type=str, default=None, help="YAML config (CLI flags win)")
    p.add_argument(
        "--mode",
        choices=("teacher", "infer"),
        required=True,
        help="teacher=masked token acc; infer=note F1 with the same DecodeConfig as audio2map-infer",
    )
    p.add_argument("--limit", type=int, default=100, help="teacher: max charts to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--build-grid", action="store_true", help="build audio grid on the fly (slow)")
    p.add_argument("--osu", type=str, default=None, help="single chart (required for infer)")
    p.add_argument(
        "--grid-dir",
        type=str,
        default=None,
        help="audio_grid cache dir (must match training; default: processed/audio_grid)",
    )
    p.add_argument("--start-bar", type=int, default=None, help="fixed window start (teacher)")
    p.add_argument(
        "--range-mode",
        choices=("audio_full", "reference_chart"),
        default="audio_full",
        help="infer: audio_full=whole audio; reference_chart=clip to original chart bars",
    )
    p.add_argument("--post-margin-bars", type=int, default=GenerationRangeConfig().post_margin_bars)
    p.add_argument("--temperature", type=float, default=_DEFAULT_DECODE.temperature)
    p.add_argument("--top-p", type=float, default=_DEFAULT_DECODE.top_p)
    p.add_argument("--top-k", type=int, default=_DEFAULT_DECODE.top_k)
    p.add_argument("--greedy", action="store_true", help="infer: temperature=0")
    args = parse_args_with_config(p)

    setup_logging()
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model = load_checkpoint(Path(args.checkpoint), device)

    if args.mode == "infer":
        if not args.osu:
            raise SystemExit("infer mode requires --osu")
        path = Path(args.osu)
        cond = build_cond_vec(compute_chart_meta(path))
        grid_dir = Path(args.grid_dir) if args.grid_dir else None
        stats = eval_inference_chart(
            model,
            path,
            device=device,
            cond_vec=cond,
            build_grid_if_missing=args.build_grid,
            range_mode=args.range_mode,
            post_margin_bars=args.post_margin_bars,
            grid_dir=grid_dir,
            decode=DecodeConfig(
                temperature=0.0 if args.greedy else args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            ),
        )
        print(json.dumps({"charts": 1, "notes": stats.to_dict()}, indent=2))
        return

    if args.osu:
        paths = [Path(args.osu)]
    else:
        paths = _pick_paths(args.limit, args.seed)
    if not paths:
        raise SystemExit("no charts to evaluate")

    logger.info("evaluating %d charts mode=%s", len(paths), args.mode)
    agg = eval_teacher_forcing(
        model,
        paths,
        device=device,
        seed=args.seed,
        build_grid_if_missing=args.build_grid,
        fixed_start_bar=args.start_bar,
        grid_dir=Path(args.grid_dir) if args.grid_dir else None,
    )
    if agg.charts == 0:
        raise SystemExit("could not build any eval windows")
    print(json.dumps(agg.to_dict(), indent=2))


if __name__ == "__main__":
    main()
