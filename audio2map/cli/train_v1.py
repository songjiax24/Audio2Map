"""``audio2map-train-v1``: whole-chart V1ChartModel training. Not the v0 entry."""

from __future__ import annotations

import json
from pathlib import Path

from audio2map.cli.common import parse_args_with_config, setup_logging
from audio2map.train.v1.batches import FRAME_BUDGET, WINDOW_FRAC
from audio2map.train.v1.loop import SPLIT_FILE, V1TrainConfig, train_v1
from audio2map.utils.paths import splits_dir


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Train V1ChartModel on whole charts")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="max train charts")
    parser.add_argument("--val-limit", type=int, default=None, dest="val_limit")
    parser.add_argument("--max-steps", type=int, default=200_000, dest="max_steps")
    parser.add_argument("--batch-size", type=int, default=2, dest="batch_size")
    parser.add_argument(
        "--frame-budget",
        type=int,
        default=FRAME_BUDGET,
        dest="frame_budget",
        help="pack while n * max(rate-1 frames) stays at or below this",
    )
    parser.add_argument(
        "--window-frac",
        type=float,
        default=WINDOW_FRAC,
        dest="window_frac",
        help="forward window width as a fraction of the shortest chart in it",
    )
    parser.add_argument("--num-workers", type=int, default=16, dest="num_workers")
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        dest="grad_clip",
        help="clip the token-mean CE gradient norm to this value",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="grad_checkpoint",
        help="recompute each encoder and decoder layer on the backward pass",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    parser.add_argument("--warmup-steps", type=int, default=2000, dest="warmup_steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--log-every", type=int, default=50, dest="log_every")
    parser.add_argument("--save-every", type=int, default=2000, dest="save_every")
    parser.add_argument("--val-every", type=int, default=2000, dest="val_every")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="step_N.pt to continue. max-steps is the total step, not an extra count",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        dest="split_manifest",
        help=f"default: splits/{SPLIT_FILE} under AUDIO2MAP_DATA_ROOT",
    )
    return parser


def main() -> None:
    args = parse_args_with_config(build_parser())
    setup_logging()
    if args.split_manifest:
        given = Path(args.split_manifest)
        split = given if given.is_absolute() else splits_dir() / given
    else:
        split = splits_dir() / SPLIT_FILE
    cfg = V1TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        log_every=args.log_every,
        save_every=args.save_every,
        val_every=args.val_every,
        val_limit=args.val_limit,
        limit=args.limit,
        out=Path(args.out) if args.out else None,
        split_manifest=split,
        resume=Path(args.resume) if args.resume else None,
        frame_budget=args.frame_budget,
        window_frac=args.window_frac,
        num_workers=args.num_workers,
        grad_checkpoint=args.grad_checkpoint,
        grad_clip=args.grad_clip,
    )
    summary = train_v1(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
