"""``audio2map-train``: formal multi-chart training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio2map.cli.common import parse_args_with_config, setup_logging
from audio2map.features.audio.tick_features import AUDIO_FEATURE_DIM
from audio2map.model.config import MAX_DECODER_LEN, WINDOW_BARS
from audio2map.train.loop import TrainConfig, train

_CONFIG_KEY_MAP = {"n_heads": "heads"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train AudioChartModel (formal multi-chart)")
    p.add_argument("--config", type=str, default=None, help="YAML config (e.g. configs/train/train_multi.yaml)")
    p.add_argument("--limit", type=int, default=None, help="max charts (debug subset)")
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--encoder-layers", type=int, default=4)
    p.add_argument("--decoder-layers", type=int, default=6)
    p.add_argument("--layers", type=int, default=None, help="set both encoder/decoder layers (overrides split)")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--audio-dim", type=int, default=AUDIO_FEATURE_DIM)
    p.add_argument("--max-decoder-len", type=int, default=MAX_DECODER_LEN)
    p.add_argument("--samples-per-chart", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--out", type=str, default=None, help="checkpoint directory")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--meta-manifest",
        type=str,
        default=None,
        help="chart_meta/manifest.jsonl (default: DATA/chart_meta/manifest.jsonl if present)",
    )
    p.add_argument(
        "--precision",
        choices=("bf16", "fp32"),
        default="bf16",
        help="training precision (bf16 uses autocast on CUDA)",
    )
    p.add_argument(
        "--lazy",
        action="store_true",
        help="skip upfront preload; load each chart from disk on demand (slow/step, instant start)",
    )
    p.add_argument(
        "--grid-dir",
        type=str,
        default=None,
        help="audio_grid cache directory (default: processed/audio_grid)",
    )
    p.add_argument(
        "--allow-missing-grid",
        action="store_true",
        help="include charts without valid precomputed audio_grid (not recommended)",
    )
    return p


def main() -> None:
    args = parse_args_with_config(
        build_parser(),
        key_map=_CONFIG_KEY_MAP,
        locked={"window_bars": WINDOW_BARS},
    )

    enc_layers = args.layers if args.layers is not None else args.encoder_layers
    dec_layers = args.layers if args.layers is not None else args.decoder_layers

    setup_logging()

    cfg = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        dropout=args.dropout,
        d_model=args.d_model,
        encoder_layers=enc_layers,
        decoder_layers=dec_layers,
        heads=args.heads,
        audio_dim=args.audio_dim,
        max_decoder_len=args.max_decoder_len,
        samples_per_chart=args.samples_per_chart,
        device=args.device,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        out=Path(args.out) if args.out else None,
        num_workers=args.num_workers,
        meta_manifest=Path(args.meta_manifest) if args.meta_manifest else None,
        precision=args.precision,
        lazy=args.lazy,
        grid_dir=Path(args.grid_dir) if args.grid_dir else None,
        allow_missing_grid=args.allow_missing_grid,
        limit=args.limit,
    )
    try:
        summary = train(cfg)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
