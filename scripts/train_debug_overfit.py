#!/usr/bin/env python3
"""Debug overfit: memorize a single (or few) v2 chart windows."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audio2map.training.dataset import Audio2MapV2Dataset, DatasetConfig
from audio2map.training.model import build_model


def main() -> None:
    p = argparse.ArgumentParser(description="Debug overfit one v2 chart window")
    p.add_argument("--osu", type=str, required=True, help="path to one eligible .osu")
    p.add_argument("--start-bar", type=int, default=None, help="fixed window start bar")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", type=str, default=None, help="save final checkpoint .pt")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    osu_path = Path(args.osu)
    if not osu_path.is_file():
        raise SystemExit(f"not found: {osu_path}")

    ds = Audio2MapV2Dataset(
        [osu_path],
        cfg=DatasetConfig(samples_per_chart=max(1, args.batch_size), build_grid_if_missing=True),
        seed=args.seed,
        fixed_start_bar=args.start_bar,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=Audio2MapV2Dataset.collate_fn,
    )

    model = build_model(
        d_model=args.d_model,
        n_heads=args.heads,
        layers=args.layers,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    logging.info("preloading one training batch (CPU) …")
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    logging.info(
        "batch on %s: tokens=%d audio=%s",
        device,
        batch["token_ids"].shape[1],
        tuple(batch["audio"].shape),
    )

    step = 0
    stats: dict[str, float] = {}
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        loss, metrics = model.training_step(batch)
        loss.backward()
        opt.step()
        stats = metrics
        step += 1
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            logging.info("step %d loss=%.4f acc=%.3f", step, stats["loss"], stats["token_acc"])

    print(json.dumps({"steps": step, **stats}, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(out_path, extra={"steps": step, "stats": stats, "osu": str(osu_path)})
        logging.info("saved checkpoint %s", out_path)
    if stats.get("token_acc", 0) >= 0.99:
        raise SystemExit(0)
    raise SystemExit(0 if stats.get("loss", 1) < 0.05 else 1)


if __name__ == "__main__":
    main()
