"""Whole-chart v1 eval; the existing audio2map-eval command remains v0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio2map.cli.common import parse_args_with_config, setup_logging
from audio2map.eval.v1.conditions import DIVERSITY_SEEDS, SCENARIOS
from audio2map.eval.v1.motifs import DEFAULT_MOTIF_K
from audio2map.eval.v1.run import MODES, EvalConfig, evaluate
from audio2map.eval.v1.rescore import RescoreConfig, rescore
from audio2map.eval.v1.probes import ProbeConfig, run_probes
from audio2map.eval.v1.comparison import compare_runs
from audio2map.train.v1.loop import SPLIT_FILE
from audio2map.utils.paths import raw_dir, splits_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Whole-chart v1 evaluation")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=(*MODES, "rescore", "probes", "compare"), default="all")
    parser.add_argument("--source-run", type=Path, help="completed run to rescore without sampling")
    parser.add_argument("--compare-run", type=Path, help="right-hand run for paired comparison")
    parser.add_argument("--refresh-teacher", action="store_true", help="refresh teacher readouts")
    parser.add_argument("--html", action="store_true", help="write SVG/HTML figures (eval extra)")
    parser.add_argument("--probe-charts", type=int, default=ProbeConfig.charts)
    parser.add_argument("--probe-teacher-charts", type=int, default=ProbeConfig.teacher_charts)
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=list(ProbeConfig.seeds))
    parser.add_argument("--probe-protocol", type=Path, help="frozen offset/control requests JSON")
    parser.add_argument("--probe-reuse-run", type=Path, action="append", default=[],
                        help="reuse compatible completed probe cases")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DIVERSITY_SEEDS))
    parser.add_argument("--seed", type=int, default=0, help="frozen pool, donors, partial masks")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("bf16", "fp32"), default="bf16",
        help="teacher precision; production CUDA generation requires bf16",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--osu", nargs="+", type=Path, help="explicit charts belonging to --split")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--motif-vocab", type=Path)
    parser.add_argument("--motif-k", type=int, default=DEFAULT_MOTIF_K)
    parser.add_argument("--resume", action="store_true", help="reuse completed records of this run")
    return parser


def main() -> None:
    args = parse_args_with_config(build_parser())
    setup_logging()
    if args.mode == "compare":
        if args.source_run is None or args.compare_run is None or args.out is None:
            raise SystemExit("compare requires --source-run, --compare-run and --out")
        compare_runs(args.source_run, args.compare_run, args.out)
        print(json.dumps({"output": str(args.out)}, indent=2))
        return
    if args.mode == "probes":
        if args.source_run is None or args.out is None:
            raise SystemExit("probes requires --source-run and --out")
        result = run_probes(ProbeConfig(
            source=args.source_run, out=args.out, charts=args.probe_charts,
            teacher_charts=args.probe_teacher_charts, seed=args.seed,
            seeds=tuple(args.probe_seeds), device=args.device, precision=args.precision,
            resume=args.resume, html=args.html,
            protocol=args.probe_protocol, reuse_runs=tuple(args.probe_reuse_run),
        ))
        print(json.dumps({"output": str(args.out), "elapsed_s": result["elapsed_s"]}, indent=2))
        return
    if args.mode == "rescore":
        if args.source_run is None or args.out is None:
            raise SystemExit("rescore requires --source-run and --out")
        result = rescore(RescoreConfig(
            source=args.source_run, out=args.out, motif_vocab=args.motif_vocab,
            refresh_teacher=args.refresh_teacher, device=args.device,
            precision=args.precision, html=args.html,
        ))
        print(json.dumps({"output": str(args.out), "elapsed_s": result["elapsed_s"]}, indent=2))
        return
    if args.split is None or args.out is None:
        raise SystemExit("--split and --out must be supplied explicitly")
    split = args.split_manifest or splits_dir() / SPLIT_FILE
    if not split.is_absolute() and not split.is_file():
        split = splits_dir() / split
    cfg = EvalConfig(
        checkpoint=args.checkpoint,
        split_manifest=split,
        split=args.split,
        root=args.raw_root or raw_dir(),
        out=args.out,
        mode=args.mode,
        scenarios=tuple(args.scenarios),
        seeds=tuple(args.seeds),
        seed=args.seed,
        temperature=args.temperature,
        device=args.device,
        precision=args.precision,
        limit=args.limit,
        paths=tuple(args.osu or ()),
        workers=args.workers,
        motif_vocab=args.motif_vocab,
        motif_k=args.motif_k,
        resume=args.resume,
        html=args.html,
    )
    result = evaluate(cfg)
    print(json.dumps({"output": str(cfg.out), "elapsed_s": result.get("elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
