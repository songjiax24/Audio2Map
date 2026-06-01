#!/usr/bin/env python3
"""Train AudioChartModel on eligible v2 charts (formal multi-chart training)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audio2map.data.chart_bundle import filter_paths_with_grid
from audio2map.data.eligible_charts import list_eligible_osu_paths
from audio2map.training.dataset import Audio2MapV2Dataset, DatasetConfig
from audio2map.training.model import build_model
from audio2map.utils.paths import default_train_checkpoint_dir, get_data_root


def main() -> None:
    p = argparse.ArgumentParser(description="Train v2 AudioChartModel (formal multi-chart)")
    p.add_argument("--limit", type=int, default=None, help="max charts (debug subset)")
    p.add_argument("--steps", type=int, default=60_000, help="optimizer steps (default: formal run)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--encoder-layers", type=int, default=4)
    p.add_argument("--decoder-layers", type=int, default=6)
    p.add_argument("--layers", type=int, default=None, help="set both encoder/decoder layers (overrides split)")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--samples-per-chart", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--out", type=str, default=None, help="checkpoint directory")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="disable bfloat16 autocast on CUDA",
    )
    p.add_argument(
        "--architecture",
        choices=("enc_dec", "prefix_lm"),
        default="enc_dec",
        help="enc_dec=formal encoder-decoder (default); prefix_lm=legacy ablation",
    )
    p.add_argument(
        "--window-bars-choices",
        type=str,
        default="8,12,16",
        help="comma-separated window sizes, e.g. 8,12,16 (formal target)",
    )
    p.add_argument(
        "--lazy",
        action="store_true",
        help="skip upfront preload; load each chart from disk on demand (slow/step, instant start)",
    )
    p.add_argument(
        "--allow-missing-grid",
        action="store_true",
        help="include charts without valid precomputed audio_grid (not recommended)",
    )
    p.add_argument(
        "--audio-pooling",
        choices=("tick", "bar"),
        default="tick",
        help="prefix_lm ablation only; enc_dec is always tick-level",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt_dir = Path(args.out) if args.out else default_train_checkpoint_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    require_grid = not args.allow_missing_grid
    eligible = list_eligible_osu_paths()
    paths = filter_paths_with_grid(eligible) if require_grid else eligible
    logging.info(
        "charts: %d trainable / %d eligible (require_grid=%s)",
        len(paths),
        len(eligible),
        require_grid,
    )
    if require_grid and len(paths) == 0:
        raise SystemExit("no charts with valid audio_grid — run audit_audio_grid.py first")
    if args.limit:
        paths = paths[: args.limit]

    window_bars_choices = tuple(int(x.strip()) for x in args.window_bars_choices.split(",") if x.strip())

    ds = Audio2MapV2Dataset(
        paths,
        cfg=DatasetConfig(
            window_bars_choices=window_bars_choices,
            samples_per_chart=args.samples_per_chart,
            build_grid_if_missing=not require_grid,
            require_grid=require_grid,
            lazy=args.lazy,
        ),
        seed=args.seed,
    )
    logging.info(
        "dataset: %d charts, %d samples/epoch, architecture=%s, window_bars=%s, lazy=%s",
        len(ds.bundles) if not ds.lazy else len(ds.osu_paths),
        len(ds),
        args.architecture,
        window_bars_choices,
        ds.lazy,
    )

    if args.architecture != "enc_dec":
        raise SystemExit("formal training requires --architecture enc_dec")

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
        architecture=args.architecture,
        d_model=args.d_model,
        n_heads=args.heads,
        encoder_layers=enc_layers,
        decoder_layers=dec_layers,
        audio_pooling=args.audio_pooling,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and torch.cuda.is_bf16_supported() is False)
    logging.info(
        "model: enc_dec d=%d enc=%d dec=%d heads=%d params=%.2fM amp=%s",
        args.d_model,
        enc_layers,
        dec_layers,
        args.heads,
        n_params / 1e6,
        use_amp,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    stats: dict[str, float] = {}
    t0 = time.time()
    while step < args.steps:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, stats = model.training_step(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                logging.info(
                    "step %d loss=%.4f acc=%.3f audio_ticks=%.0f (%.1fs, %.1f step/s)",
                    step,
                    stats["loss"],
                    stats["token_acc"],
                    stats.get("audio_ticks", 0),
                    elapsed,
                    step / max(elapsed, 1e-6),
                )
            if step % args.save_every == 0 or step == args.steps:
                ckpt_path = ckpt_dir / f"step_{step}.pt"
                model.save_checkpoint(
                    ckpt_path,
                    extra={
                        "step": step,
                        "stats": stats,
                        "num_charts": len(ds.bundles) if not ds.lazy else len(ds.osu_paths),
                        "d_model": args.d_model,
                        "encoder_layers": enc_layers,
                        "decoder_layers": dec_layers,
                        "architecture": args.architecture,
                        "audio_pooling": args.audio_pooling if args.architecture == "prefix_lm" else "tick",
                    },
                )
                logging.info("saved %s", ckpt_path)
            if step >= args.steps:
                break

    summary = {
        "steps": step,
        "charts": len(ds.bundles) if not ds.lazy else len(ds.osu_paths),
        "d_model": args.d_model,
        "encoder_layers": enc_layers,
        "decoder_layers": dec_layers,
        "architecture": args.architecture,
        "audio_pooling": args.audio_pooling if args.architecture == "prefix_lm" else "tick",
        **stats,
        "checkpoint_dir": str(ckpt_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
