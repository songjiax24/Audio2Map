"""Training loop for AudioChartModel (formal multi-chart training)."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audio2map.dataset.bundle import filter_paths_with_grid
from audio2map.dataset.filter import list_eligible_osu_paths
from audio2map.features.audio.tick_features import AUDIO_FEATURE_DIM
from audio2map.model.config import MAX_DECODER_LEN, WINDOW_BARS
from audio2map.model.model import build_model
from audio2map.train.data import AudioChartDataset, pad_batch
from audio2map.train.step import training_step
from audio2map.utils.paths import audio_grid_dir, chart_meta_dir, formal_checkpoint_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    max_steps: int = 200_000
    batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    dropout: float = 0.1
    d_model: int = 512
    encoder_layers: int = 4
    decoder_layers: int = 6
    heads: int = 8
    audio_dim: int = AUDIO_FEATURE_DIM
    max_decoder_len: int = MAX_DECODER_LEN
    samples_per_chart: int = 4
    device: str = "cuda"
    seed: int = 0
    log_every: int = 50
    save_every: int = 2000
    out: Path | None = None
    num_workers: int = 8
    meta_manifest: Path | None = None
    precision: str = "bf16"  # "bf16" | "fp32"
    lazy: bool = False
    grid_dir: Path | None = None
    allow_missing_grid: bool = False
    limit: int | None = None


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def _lr_for_step(step: int, *, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def train(cfg: TrainConfig) -> dict:
    """Run formal training; returns the run summary dict."""
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(
        cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    )
    use_amp = (
        cfg.precision == "bf16"
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    if cfg.precision == "bf16" and not use_amp:
        logger.warning("bf16 requested but unavailable; using fp32")
    ckpt_dir = Path(cfg.out) if cfg.out else formal_checkpoint_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    require_grid = not cfg.allow_missing_grid
    grid_dir = Path(cfg.grid_dir) if cfg.grid_dir else audio_grid_dir()
    manifest_path = (
        Path(cfg.meta_manifest)
        if cfg.meta_manifest
        else chart_meta_dir() / "manifest.jsonl"
    )
    eligible = list_eligible_osu_paths()
    paths = (
        filter_paths_with_grid(eligible, grid_dir=grid_dir, min_window_bars=WINDOW_BARS)
        if require_grid
        else eligible
    )
    logger.info(
        "charts: %d trainable / %d eligible (require_grid=%s, min_window_bars=%d, grid_dir=%s)",
        len(paths),
        len(eligible),
        require_grid,
        WINDOW_BARS,
        grid_dir,
    )
    if require_grid and len(paths) == 0:
        raise RuntimeError("no charts with valid audio_grid — run precompute first")
    if manifest_path.is_file():
        logger.info("using chart meta manifest: %s", manifest_path)
    else:
        logger.warning(
            "chart meta manifest missing: %s (preload will compute meta on the fly)",
            manifest_path,
        )
    if cfg.limit:
        paths = paths[: cfg.limit]

    ds = AudioChartDataset(
        paths,
        samples_per_chart=cfg.samples_per_chart,
        build_grid_if_missing=not require_grid,
        lazy=cfg.lazy,
        meta_manifest_path=manifest_path if manifest_path.is_file() else None,
        seed=cfg.seed,
        grid_dir=grid_dir,
    )
    logger.info(
        "dataset: %d charts, %d samples/epoch, window_bars=%d, lazy=%s",
        ds.n_charts,
        len(ds),
        WINDOW_BARS,
        ds.lazy,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=pad_batch,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    if len(loader) == 0:
        raise RuntimeError(
            f"no training batches (samples={len(ds)}, batch_size={cfg.batch_size}, drop_last=True)"
        )

    model = build_model(
        d_model=cfg.d_model,
        n_heads=cfg.heads,
        encoder_layers=cfg.encoder_layers,
        decoder_layers=cfg.decoder_layers,
        dropout=cfg.dropout,
        audio_dim=cfg.audio_dim,
        max_decoder_len=cfg.max_decoder_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "model: enc_dec d=%d enc=%d dec=%d heads=%d audio_dim=%d params=%.2fM amp=%s accum=%d",
        cfg.d_model,
        cfg.encoder_layers,
        cfg.decoder_layers,
        cfg.heads,
        cfg.audio_dim,
        n_params / 1e6,
        use_amp,
        cfg.grad_accum_steps,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    step = 0
    micro = 0
    stats: dict[str, float] = {}
    accum_loss = 0.0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    while step < cfg.max_steps:
        for batch in loader:
            _set_lr(opt, _lr_for_step(step, base_lr=cfg.lr, warmup_steps=cfg.warmup_steps))
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, stats = training_step(model, batch)
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            accum_loss += float(loss.item())
            micro += 1

            if micro % cfg.grad_accum_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            stats = dict(stats)
            stats["loss"] = accum_loss
            accum_loss = 0.0
            micro = 0

            if step % cfg.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                logger.info(
                    "step %d lr=%.2e loss=%.4f acc=%.3f audio_ticks=%.0f (%.1fs, %.1f step/s)",
                    step,
                    opt.param_groups[0]["lr"],
                    stats["loss"],
                    stats["token_acc"],
                    stats.get("audio_ticks", 0),
                    elapsed,
                    step / max(elapsed, 1e-6),
                )
            if step % cfg.save_every == 0 or step == cfg.max_steps:
                ckpt_path = ckpt_dir / f"step_{step}.pt"
                model.save_checkpoint(
                    ckpt_path,
                    extra={
                        "step": step,
                        "stats": stats,
                        "lr": opt.param_groups[0]["lr"],
                        "num_charts": ds.n_charts,
                        "grid_dir": str(grid_dir),
                        "meta_manifest": str(manifest_path)
                        if manifest_path.is_file()
                        else None,
                    },
                )
                logger.info("saved %s", ckpt_path)
            if step >= cfg.max_steps:
                break

    summary = {
        "steps": step,
        "charts": ds.n_charts,
        "d_model": cfg.d_model,
        "encoder_layers": cfg.encoder_layers,
        "decoder_layers": cfg.decoder_layers,
        **stats,
        "checkpoint_dir": str(ckpt_dir),
        "grid_dir": str(grid_dir),
    }
    return summary
