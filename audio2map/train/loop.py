"""Training loop for AudioChartModel (formal multi-chart training)."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audio2map.dataset.bundle import filter_v0_paths
from audio2map.dataset.split import (
    DEFAULT_RATIOS,
    GROUP_BY,
    SPLIT_SCHEME,
    SplitError,
    build_split_manifest,
    default_split_manifest_path,
    list_raw_osu_paths,
    load_split_manifest,
    save_split_manifest,
    split_group_chart_counts,
    split_paths,
    manifest_sha256,
)
from audio2map.features.audio.tick_features import AUDIO_FEATURE_DIM
from audio2map.metrics.nll import token_nll_stats
from audio2map.metrics.token import SplitTokenAccuracy, split_token_accuracy
from audio2map.metrics.validity import logit_validity
from audio2map.model.config import MAX_DECODER_LEN, WINDOW_BARS
from audio2map.model.model import build_model
from audio2map.tokens import invert_vocab
from audio2map.train.data import AudioChartDataset, pad_batch
from audio2map.train.step import scale_grads_by_valid_tokens, training_step
from audio2map.utils.paths import (
    audio_grid_dir,
    chart_meta_dir,
    formal_checkpoint_dir,
    raw_dir,
)

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
    val_every: int = 2000
    val_samples_per_chart: int = 1
    out: Path | None = None
    num_workers: int = 8
    meta_manifest: Path | None = None
    split_manifest: Path | None = None
    init_split: bool = False
    allow_set_id_mismatch: bool = False
    split_seed: int = 0
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
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


def _v0_filter(paths: list[Path], *, require_grid: bool, grid_dir: Path) -> list[Path]:
    return filter_v0_paths(
        paths, require_grid=require_grid, grid_dir=grid_dir, min_window_bars=WINDOW_BARS
    )


def _make_loader(
    ds: AudioChartDataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pad_batch,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool,
    window_bars: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    n_valid = 0.0
    n_correct = 0.0
    type_acc = SplitTokenAccuracy()
    legal_top1 = 0.0
    legal_prob = 0.0
    legal_n = 0.0
    id_to_token = invert_vocab()
    try:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(
                    batch["audio"],
                    batch["cond_vec"],
                    batch["token_ids"],
                    attn_mask=batch.get("attn_mask"),
                    audio_mask=batch.get("audio_mask"),
                )
                targets = batch["token_ids"][:, 1:]
                mask = batch["loss_mask"][:, 1:]
                batch_loss_sum, batch_valid, batch_correct = token_nll_stats(
                    logits, targets, mask
                )
            loss_sum += float(batch_loss_sum.item())
            n_valid += float(batch_valid.item())
            n_correct += float(batch_correct.item())
            pred = logits.argmax(dim=-1)
            for i in range(pred.shape[0]):
                type_acc.merge(
                    split_token_accuracy(
                        pred[i], targets[i], mask[i], id_to_token=id_to_token
                    )
                )
            validity = logit_validity(
                logits,
                batch["token_ids"],
                batch["loss_mask"],
                window_bars=window_bars,
                id_to_token=id_to_token,
            )
            legal_top1 += validity.top1_count
            legal_prob += validity.prob_sum
            legal_n += validity.n_valid
    finally:
        model.train(was_training)

    types = type_acc.to_dict()
    return {
        "val_loss": loss_sum / n_valid if n_valid else 0.0,
        "val_token_acc": n_correct / n_valid if n_valid else 0.0,
        "val_BAR_acc": types["bar_acc"],
        "val_POS_acc": types["pos_acc"],
        "val_ROW_acc": types["row_exact_acc"],
        "val_EOS_acc": types["eos_acc"],
        "val_TAP_recall": types["tap_recall"],
        "val_hold_start_recall": types["hold_start_recall"],
        "val_hold_end_recall": types["hold_end_recall"],
        "val_legal_top1": legal_top1 / legal_n if legal_n else 0.0,
        "val_legal_probability": legal_prob / legal_n if legal_n else 0.0,
    }


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
    raw_root = raw_dir()
    meta_path = (
        Path(cfg.meta_manifest)
        if cfg.meta_manifest
        else chart_meta_dir() / "manifest.jsonl"
    )
    split_path = (
        Path(cfg.split_manifest) if cfg.split_manifest else default_split_manifest_path()
    )
    ratios = {
        "train": cfg.train_ratio,
        "val": cfg.val_ratio,
        "test": cfg.test_ratio,
    }
    try:
        if cfg.init_split:
            split_manifest = build_split_manifest(
                raw_root=raw_root,
                seed=cfg.split_seed,
                ratios=ratios,
                allow_set_id_mismatch=cfg.allow_set_id_mismatch,
            )
            save_split_manifest(split_manifest, split_path)
            logger.info("wrote split manifest %s (%d groups)", split_path, len(split_manifest.groups))
        else:
            split_manifest = load_split_manifest(
                split_path,
                expected_scheme=SPLIT_SCHEME,
                expected_group_by=GROUP_BY,
                expected_seed=cfg.split_seed,
                expected_ratios=ratios or DEFAULT_RATIOS,
            )
            logger.info("using split manifest %s (%d groups)", split_path, len(split_manifest.groups))
        raw_paths = list_raw_osu_paths(raw_root)
        by_split = split_paths(raw_paths, split_manifest, raw_root=raw_root)
    except SplitError as exc:
        raise RuntimeError(str(exc)) from exc

    split_hash = manifest_sha256(split_manifest)
    counts = split_group_chart_counts(split_manifest)
    logger.info(
        "split groups/charts train=%d/%d val=%d/%d test=%d/%d sha256=%s",
        counts["train"][0],
        counts["train"][1],
        counts["val"][0],
        counts["val"][1],
        counts["test"][0],
        counts["test"][1],
        split_hash[:12],
    )

    train_paths = _v0_filter(by_split["train"], require_grid=require_grid, grid_dir=grid_dir)
    val_paths = _v0_filter(by_split["val"], require_grid=require_grid, grid_dir=grid_dir)
    test_paths = _v0_filter(by_split["test"], require_grid=require_grid, grid_dir=grid_dir)
    logger.info(
        "v0-eligible charts train=%d val=%d test=%d (require_grid=%s)",
        len(train_paths),
        len(val_paths),
        len(test_paths),
        require_grid,
    )
    if require_grid and len(train_paths) == 0:
        raise RuntimeError("no train charts with valid audio_grid — run precompute first")
    if meta_path.is_file():
        logger.info("using chart meta manifest: %s", meta_path)
    else:
        logger.warning(
            "chart meta manifest missing: %s (preload will compute meta on the fly)",
            meta_path,
        )
    if cfg.limit:
        train_paths = train_paths[: cfg.limit]

    ds = AudioChartDataset(
        train_paths,
        samples_per_chart=cfg.samples_per_chart,
        build_grid_if_missing=not require_grid,
        lazy=cfg.lazy,
        meta_manifest_path=meta_path if meta_path.is_file() else None,
        seed=cfg.seed,
        grid_dir=grid_dir,
    )
    logger.info(
        "train dataset: %d charts, %d samples/epoch, window_bars=%d, lazy=%s",
        ds.n_charts,
        len(ds),
        WINDOW_BARS,
        ds.lazy,
    )

    pin_memory = device.type == "cuda"
    loader = _make_loader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    if len(loader) == 0:
        raise RuntimeError(
            f"no training batches (samples={len(ds)}, batch_size={cfg.batch_size}, drop_last=True)"
        )

    val_loader: DataLoader | None = None
    if val_paths:
        val_ds = AudioChartDataset(
            val_paths,
            samples_per_chart=cfg.val_samples_per_chart,
            build_grid_if_missing=not require_grid,
            lazy=cfg.lazy,
            meta_manifest_path=meta_path if meta_path.is_file() else None,
            seed=cfg.seed,
            grid_dir=grid_dir,
        )
        val_loader = _make_loader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )
        logger.info(
            "val dataset: %d charts, %d samples (fixed windows, samples_per_chart=%d)",
            val_ds.n_charts,
            len(val_ds),
            cfg.val_samples_per_chart,
        )
    else:
        logger.warning("no v0-eligible val charts; skipping validation")

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
    val_stats: dict[str, float] = {}
    accum_loss_sum = 0.0
    accum_valid = 0.0
    accum_correct = 0.0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    while step < cfg.max_steps:
        for batch in loader:
            _set_lr(opt, _lr_for_step(step, base_lr=cfg.lr, warmup_steps=cfg.warmup_steps))
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss_sum, batch_stats = training_step(model, batch)
            loss_sum.backward()
            accum_loss_sum += batch_stats["loss_sum"]
            accum_valid += batch_stats["valid_tokens"]
            accum_correct += batch_stats["correct_tokens"]
            micro += 1

            if micro % cfg.grad_accum_steps != 0:
                continue

            scale_grads_by_valid_tokens(model, accum_valid)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            stats = {
                "loss": accum_loss_sum / accum_valid if accum_valid else 0.0,
                "token_acc": accum_correct / accum_valid if accum_valid else 0.0,
                "audio_ticks": batch_stats.get("audio_ticks", 0.0),
            }
            accum_loss_sum = 0.0
            accum_valid = 0.0
            accum_correct = 0.0
            micro = 0

            if step % cfg.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                logger.info(
                    "step %d lr=%.2e train_loss=%.4f train_token_acc=%.3f "
                    "audio_ticks=%.0f (%.1fs, %.1f step/s)",
                    step,
                    opt.param_groups[0]["lr"],
                    stats["loss"],
                    stats["token_acc"],
                    stats.get("audio_ticks", 0),
                    elapsed,
                    step / max(elapsed, 1e-6),
                )
            if val_loader is not None and (step % cfg.val_every == 0 or step == 1):
                val_stats = _run_validation(
                    model,
                    val_loader,
                    device=device,
                    use_amp=use_amp,
                    window_bars=WINDOW_BARS,
                )
                logger.info(
                    "step %d val_loss=%.4f val_token_acc=%.3f "
                    "BAR=%.3f POS=%.3f ROW=%.3f EOS=%.3f "
                    "TAP=%.3f hold_start=%.3f hold_end=%.3f "
                    "legal_top1=%.3f legal_prob=%.3f",
                    step,
                    val_stats["val_loss"],
                    val_stats["val_token_acc"],
                    val_stats["val_BAR_acc"],
                    val_stats["val_POS_acc"],
                    val_stats["val_ROW_acc"],
                    val_stats["val_EOS_acc"],
                    val_stats["val_TAP_recall"],
                    val_stats["val_hold_start_recall"],
                    val_stats["val_hold_end_recall"],
                    val_stats["val_legal_top1"],
                    val_stats["val_legal_probability"],
                )
                stats.update(val_stats)
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
                        "meta_manifest": str(meta_path) if meta_path.is_file() else None,
                        "split_manifest": str(split_path),
                        "split_manifest_sha256": split_hash,
                        "split_scheme": split_manifest.split_scheme,
                    },
                )
                logger.info("saved %s", ckpt_path)
            if step >= cfg.max_steps:
                break

    summary = {
        "steps": step,
        "charts": ds.n_charts,
        "val_charts": len(val_paths),
        "test_charts": len(test_paths),
        "d_model": cfg.d_model,
        "encoder_layers": cfg.encoder_layers,
        "decoder_layers": cfg.decoder_layers,
        **stats,
        "checkpoint_dir": str(ckpt_dir),
        "grid_dir": str(grid_dir),
        "split_manifest": str(split_path),
        "split_manifest_sha256": split_hash,
    }
    return summary
