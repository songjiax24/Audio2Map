#!/usr/bin/env python3
"""Train AudioChartModel on eligible v2 charts."""

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
from audio2map.training.model import AudioChartModel
from audio2map.utils.paths import get_data_root


def main() -> None:
    p = argparse.ArgumentParser(description="Train v2 AudioChartModel")
    p.add_argument("--limit", type=int, default=None, help="max eligible charts to try")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--samples-per-chart", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--out", type=str, default=None, help="checkpoint dir")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--require-grid",
        action="store_true",
        help="only train charts with precomputed audio_grid",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt_dir = Path(args.out) if args.out else get_data_root() / "processed_v2" / "checkpoints" / "trial_multi"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    paths = list_eligible_osu_paths()
    if args.require_grid:
        paths = filter_paths_with_grid(paths)
        logging.info("charts with audio_grid: %d", len(paths))
    if args.limit:
        paths = paths[: args.limit]
    logging.info("training on %d charts (require_grid=%s)", len(paths), args.require_grid)

    ds = Audio2MapV2Dataset(
        paths,
        cfg=DatasetConfig(
            samples_per_chart=args.samples_per_chart,
            build_grid_if_missing=not args.require_grid,
            require_grid=args.require_grid,
        ),
        seed=args.seed,
    )
    logging.info("dataset: %d charts, %d samples/epoch", len(ds.bundles), len(ds))

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=Audio2MapV2Dataset.collate_fn,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    model = AudioChartModel(
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    stats: dict[str, float] = {}
    t0 = time.time()
    while step < args.steps:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss, stats = model.training_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                logging.info(
                    "step %d loss=%.4f acc=%.3f (%.1fs, %.1f step/s)",
                    step,
                    stats["loss"],
                    stats["token_acc"],
                    elapsed,
                    step / max(elapsed, 1e-6),
                )
            if step % args.save_every == 0 or step == args.steps:
                ckpt_path = ckpt_dir / f"step_{step}.pt"
                model.save_checkpoint(
                    ckpt_path,
                    extra={"step": step, "stats": stats, "num_charts": len(ds.bundles)},
                )
                logging.info("saved %s", ckpt_path)
            if step >= args.steps:
                break

    summary = {
        "steps": step,
        "charts": len(ds.bundles),
        **stats,
        "checkpoint_dir": str(ckpt_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
