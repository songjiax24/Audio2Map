#!/usr/bin/env python3
"""Train AudioChartModel on eligible v2 charts (formal multi-chart training)."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audio2map.audio.tick_features import V2_FEATURE_DIM
from audio2map.data.chart_bundle import filter_paths_with_grid
from audio2map.data.eligible_charts import list_eligible_osu_paths
from audio2map.osu.grid_config import WINDOW_BARS
from audio2map.training.config import MAX_DECODER_LEN
from audio2map.training.dataset import Audio2MapV2Dataset, DatasetConfig
from audio2map.training.model import build_model
from audio2map.utils.paths import audio_grid_dir, chart_meta_dir, processed_v2_dir


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def _lr_for_step(step: int, *, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def main() -> None:
    p = argparse.ArgumentParser(description="Train v2 AudioChartModel (formal multi-chart)")
    p.add_argument("--limit", type=int, default=None, help="max charts (debug subset)")
    p.add_argument("--steps", type=int, default=None, help="optimizer steps")
    p.add_argument("--max-steps", type=int, default=200_000, help="alias for --steps")
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
    p.add_argument("--n-heads", type=int, default=None, help="alias for --heads")
    p.add_argument("--audio-dim", type=int, default=V2_FEATURE_DIM)
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
        "--no-amp",
        action="store_true",
        help="disable bfloat16 autocast on CUDA",
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
        help="audio_grid cache directory (default: processed_v2/audio_grid)",
    )
    p.add_argument(
        "--allow-missing-grid",
        action="store_true",
        help="include charts without valid precomputed audio_grid (not recommended)",
    )
    args = p.parse_args()

    if args.n_heads is not None:
        args.heads = args.n_heads
    max_steps = args.steps if args.steps is not None else args.max_steps
    use_amp = args.precision == "bf16" and not args.no_amp

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt_dir = Path(args.out) if args.out else processed_v2_dir() / "checkpoints" / "formal_enc_dec_v3"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    require_grid = not args.allow_missing_grid
    grid_dir = Path(args.grid_dir) if args.grid_dir else audio_grid_dir()
    manifest_path = Path(args.meta_manifest) if args.meta_manifest else chart_meta_dir() / "manifest.jsonl"
    eligible = list_eligible_osu_paths()
    paths = filter_paths_with_grid(eligible, grid_dir=grid_dir) if require_grid else eligible
    logging.info(
        "charts: %d trainable / %d eligible (require_grid=%s, grid_dir=%s)",
        len(paths),
        len(eligible),
        require_grid,
        grid_dir,
    )
    if require_grid and len(paths) == 0:
        raise SystemExit("no charts with valid audio_grid — run precompute_audio_grid.py first")
    if manifest_path.is_file():
        logging.info("using chart meta manifest: %s", manifest_path)
    else:
        logging.warning("chart meta manifest missing: %s (preload will compute meta on the fly)", manifest_path)
    if args.limit:
        paths = paths[: args.limit]

    ds = Audio2MapV2Dataset(
        paths,
        cfg=DatasetConfig(
            samples_per_chart=args.samples_per_chart,
            build_grid_if_missing=not require_grid,
            require_grid=require_grid,
            lazy=args.lazy,
            meta_manifest_path=manifest_path if manifest_path.is_file() else None,
        ),
        seed=args.seed,
        grid_dir=grid_dir,
    )
    logging.info(
        "dataset: %d charts, %d samples/epoch, window_bars=%d, lazy=%s",
        len(ds.bundles) if not ds.lazy else len(ds.osu_paths),
        len(ds),
        WINDOW_BARS,
        ds.lazy,
    )

    enc_layers = args.layers if args.layers is not None else args.encoder_layers
    dec_layers = args.layers if args.layers is not None else args.decoder_layers

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=Audio2MapV2Dataset.collate_fn,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(
        d_model=args.d_model,
        n_heads=args.heads,
        encoder_layers=enc_layers,
        decoder_layers=dec_layers,
        dropout=args.dropout,
        audio_dim=args.audio_dim,
        max_decoder_len=args.max_decoder_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda" and not torch.cuda.is_bf16_supported())
    logging.info(
        "model: enc_dec d=%d enc=%d dec=%d heads=%d audio_dim=%d params=%.2fM amp=%s accum=%d",
        args.d_model,
        enc_layers,
        dec_layers,
        args.heads,
        args.audio_dim,
        n_params / 1e6,
        use_amp and device.type == "cuda",
        args.grad_accum_steps,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    micro = 0
    stats: dict[str, float] = {}
    accum_loss = 0.0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    while step < max_steps:
        for batch in loader:
            _set_lr(opt, _lr_for_step(step, base_lr=args.lr, warmup_steps=args.warmup_steps))
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp and device.type == "cuda"):
                loss, stats = model.training_step(batch)
            loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += float(loss.item())
            micro += 1

            if micro % args.grad_accum_steps != 0:
                continue

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            stats = dict(stats)
            stats["loss"] = accum_loss
            accum_loss = 0.0
            micro = 0

            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                logging.info(
                    "step %d lr=%.2e loss=%.4f acc=%.3f audio_ticks=%.0f (%.1fs, %.1f step/s)",
                    step,
                    opt.param_groups[0]["lr"],
                    stats["loss"],
                    stats["token_acc"],
                    stats.get("audio_ticks", 0),
                    elapsed,
                    step / max(elapsed, 1e-6),
                )
            if step % args.save_every == 0 or step == max_steps:
                ckpt_path = ckpt_dir / f"step_{step}.pt"
                model.save_checkpoint(
                    ckpt_path,
                    extra={
                        "step": step,
                        "stats": stats,
                        "lr": opt.param_groups[0]["lr"],
                        "num_charts": len(ds.bundles) if not ds.lazy else len(ds.osu_paths),
                        "d_model": args.d_model,
                        "encoder_layers": enc_layers,
                        "decoder_layers": dec_layers,
                        "architecture": "enc_dec",
                        "grid_dir": str(grid_dir),
                        "meta_manifest": str(manifest_path) if manifest_path.is_file() else None,
                    },
                )
                logging.info("saved %s", ckpt_path)
            if step >= max_steps:
                break

    summary = {
        "steps": step,
        "charts": len(ds.bundles) if not ds.lazy else len(ds.osu_paths),
        "d_model": args.d_model,
        "encoder_layers": enc_layers,
        "decoder_layers": dec_layers,
        "architecture": "enc_dec",
        **stats,
        "checkpoint_dir": str(ckpt_dir),
        "grid_dir": str(grid_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
