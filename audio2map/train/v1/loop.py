"""Whole-chart teacher forcing for ``V1ChartModel``. Does not call the v0 loop."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from audio2map.dataset.split import load_split_manifest
from audio2map.features.cond.v1.mask import DROP_ALL_P, KEEP_ALL_P, frozen_partial_mask
from audio2map.features.cond.v1.store import ensure_v1_chart_meta
from audio2map.dataset.v1.collate import collate_v1
from audio2map.dataset.v1.gate import filter_v1_charts
from audio2map.model.v1.batch import V1Batch
from audio2map.model.v1.loss import (
    accuracy_counts,
    ce_by_type,
    legal_tally,
    nonpad_count,
    plain_ce,
)
from audio2map.model.v1.model import V1ChartModel
from audio2map.model.v1.weights import load_v1_optimizer, load_v1_state_dict
from audio2map.features.audio.v1.store import ensure_rate1_audio
from audio2map.train.v1.token_cache import ensure_rate1_tokens
from audio2map.train.v1.batches import FRAME_BUDGET, WINDOW_FRAC, epoch_batches
from audio2map.train.v1.data import V1ChartDataset
from audio2map.train.v1.lengths import (
    cache_path_for,
    length_summary,
    load_rate1_lengths,
)
from audio2map.train.v1.prepare import ChartBuilder, InFlight
from audio2map.utils.paths import raw_dir, splits_dir, v1_checkpoint_dir

logger = logging.getLogger(__name__)

SPLIT_FILE = "beatmapset_v1_drop2509435.json"
KEEP_CHECKPOINTS = 3
VAL_PATIENCE = 3


@dataclass(frozen=True, slots=True)
class V1TrainConfig:
    """Defaults match the locked augmentation and Flex training path."""

    max_steps: int = 200_000
    batch_size: int = 2
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    seed: int = 0
    device: str = "cuda"
    precision: str = "bf16"
    log_every: int = 50
    save_every: int = 2000
    val_every: int = 2000
    val_limit: int | None = None
    limit: int | None = None
    out: Path | None = None
    split_manifest: Path | None = None
    resume: Path | None = None
    frame_budget: int = FRAME_BUDGET
    window_frac: float = WINDOW_FRAC
    num_workers: int = 16
    grad_checkpoint: bool = True
    grad_clip: float = 1.0


def adamw_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Decay matrices only. Bias and norm parameters get weight decay 0."""
    decay: list[torch.nn.Parameter] = []
    plain: list[torch.nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            plain.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def lr_at_step(step: int, *, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup, then the base learning rate. Same shape as the v0 loop."""
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def _use_expandable_segments() -> None:
    """Ask the CUDA allocator to grow a block instead of leaving unusable holes.

    The step that died had about 20.5 GiB allocated and 9.9 GiB reserved but
    not allocatable. This must run before the caching allocator starts.
    """
    key = "PYTORCH_CUDA_ALLOC_CONF"
    current = os.environ.get(key, "")
    if "expandable_segments" in current:
        return
    extra = "expandable_segments:True"
    os.environ[key] = f"{current},{extra}" if current else extra


def train_v1(cfg: V1TrainConfig) -> dict[str, float | int | str]:
    """Run whole-chart steps. Validation and inference-style reads use rate 1."""
    _use_expandable_segments()
    if cfg.precision not in ("bf16", "fp32"):
        raise ValueError(f"precision must be bf16 or fp32, got {cfg.precision}")
    if cfg.grad_clip <= 0.0:
        raise ValueError(f"grad_clip must be positive, got {cfg.grad_clip}")
    root = raw_dir()
    split_path = cfg.split_manifest or (splits_dir() / SPLIT_FILE)
    manifest = load_split_manifest(split_path)
    train_charts = [chart for chart in manifest.charts if chart.split == "train"]
    val_charts = [chart for chart in manifest.charts if chart.split == "val"]
    if not train_charts:
        raise RuntimeError(f"no train charts in {split_path}")
    if cfg.limit is not None:
        train_charts = train_charts[: cfg.limit]
    if not train_charts:
        raise RuntimeError("no train charts after limit")
    if cfg.val_limit is not None:
        val_charts = val_charts[: cfg.val_limit]
    # Length and rate-1 condition caches fork. Do this before CUDA is initialized.
    length_rows = load_rate1_lengths(train_charts, root, cache_path_for(split_path))
    train_charts, length_rows = filter_v1_charts(
        train_charts,
        root,
        lengths=length_rows,
        workers=cfg.num_workers,
    )
    val_charts, _ = filter_v1_charts(val_charts, root, workers=cfg.num_workers)
    if not train_charts or length_rows is None:
        raise RuntimeError("no train charts after the v1 gate")
    ensure_rate1_tokens(train_charts, root, length_rows, workers=cfg.num_workers)
    # 16 workers were slower for decode and mel on this disk. Keep the train pool at 16.
    ensure_rate1_audio(train_charts, root, length_rows, workers=8)
    ensure_v1_chart_meta(
        [chart.relpath for chart in train_charts + val_charts],
        root,
        workers=cfg.num_workers,
    )
    lengths = length_summary(length_rows)
    _log_budget(lengths, cfg.frame_budget, cfg.window_frac, cfg.num_workers)

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if cfg.device.startswith("cuda"):
        logger.info("cuda allocator %s", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
    device = torch.device(cfg.device)
    use_amp = cfg.precision == "bf16" and device.type == "cuda"
    if cfg.precision == "bf16" and device.type != "cuda":
        logger.warning("bf16 autocast is CUDA-only; this run is fp32")
    if cfg.resume is None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    train_ds = V1ChartDataset(
        train_charts,
        root=root,
        augment=True,
        cond_mask="train",
        seed=cfg.seed,
    )
    val_ds = V1ChartDataset(
        val_charts,
        root=root,
        augment=False,
        cond_mask="none",
        seed=cfg.seed + 1,
        record_legal=True,
    )
    model = V1ChartModel().to(device)
    model.train()
    opt = torch.optim.AdamW(adamw_groups(model, cfg.weight_decay), lr=cfg.lr)
    ckpt_dir = Path(cfg.out) if cfg.out else v1_checkpoint_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step, epoch, order_index = _maybe_resume(
        cfg,
        model,
        opt,
        train_ds,
        device=device,
        n_train=len(train_ds),
        split_path=split_path,
    )
    model.set_grad_checkpoint(cfg.grad_checkpoint)
    logger.info(
        "ca_train=%s grad_checkpoint=%s device=%s precision=%s step=%s epoch=%s",
        model.ca_train,
        model.grad_checkpoint,
        device,
        cfg.precision,
        step,
        epoch,
    )

    rate1_frames = [
        row.n_frames if row.error is None else 0 for row in length_rows
    ]
    builder = ChartBuilder(train_ds, cfg.num_workers)
    val_cache: list[V1Batch] | None = None
    last_loss = 0.0
    last_acc = 0.0
    last_peak = 0.0
    last_batch = 0
    last_frames = 0
    last_tokens = 0
    last_step_s = 0.0
    last_gnorm = 0.0
    last_val: dict[str, float] = {}
    started = time.perf_counter()
    try:
        step, epoch, order_index, last, rng_state, stop_reason = _run_steps(
            cfg,
            model,
            opt,
            train_ds,
            val_ds,
            builder,
            rate1_frames,
            device,
            use_amp=use_amp,
            step=step,
            epoch=epoch,
            order_index=order_index,
            ckpt_dir=ckpt_dir,
            split_path=split_path,
            val_cache_box=[val_cache],
        )
        last_loss = last["loss"]
        last_acc = last["acc"]
        last_peak = last["peak_gib"]
        last_batch = last["batch"]
        last_frames = last["frames"]
        last_tokens = last["tokens"]
        last_step_s = last["step_s"]
        last_gnorm = last["gnorm"]
        last_val = {key: float(last[key]) for key in last if str(key).startswith("val_")}
    finally:
        builder.close()

    ckpt_path = ckpt_dir / f"step_{step}.pt"
    if step > 0:
        _save(
            model, opt, train_ds, step, epoch, order_index, ckpt_path,
            seed=cfg.seed,
            split_path=split_path,
            rng_state=rng_state,
        )
        _retain_checkpoints(ckpt_dir, _best_step(ckpt_dir))
    best = _load_best(ckpt_dir / "best.json")
    summary: dict[str, float | int | str] = {
        "steps": step,
        "loss": last_loss,
        "acc": last_acc,
        "gnorm": last_gnorm,
        "ca_train": model.ca_train,
        "grad_checkpoint": model.grad_checkpoint,
        "peak_gib": last_peak,
        "step_s": last_step_s,
        "batch": last_batch,
        "frames": last_frames,
        "tokens": last_tokens,
        **last_val,
        "frame_budget": cfg.frame_budget,
        "checkpoint": str(ckpt_path) if step > 0 else "",
        "stop": stop_reason,
        "seconds": time.perf_counter() - started,
    }
    if best is not None:
        summary["best_step"] = best[0]
        summary["best_mix_token"] = best[1]
    return summary


def _token_count(batch: V1Batch) -> int:
    """``token_ids`` length, including BOS. This is what collate pads to."""
    return int(batch.token_ids.shape[1])


def _run_steps(
    cfg: V1TrainConfig,
    model: V1ChartModel,
    opt: torch.optim.Optimizer,
    train_ds: V1ChartDataset,
    val_ds: V1ChartDataset,
    builder: ChartBuilder,
    rate1_frames: list[int],
    device: torch.device,
    *,
    use_amp: bool,
    step: int,
    epoch: int,
    order_index: int,
    ckpt_dir: Path,
    split_path: Path,
    val_cache_box: list,
) -> tuple[int, int, int, dict[str, float | int], dict, str]:
    """Train until ``max_steps`` or the val patience. OOM is not caught."""
    last = {
        "loss": 0.0,
        "acc": 0.0,
        "gnorm": 0.0,
        "peak_gib": 0.0,
        "step_s": 0.0,
        "batch": 0,
        "frames": 0,
        "tokens": 0,
    }
    last.update(_blank_val_fields())
    rng_state = train_ds.rng_state()
    best = _load_best(ckpt_dir / "best.json")
    stall = 0
    stop_reason = "max_steps"
    skips = 0
    made: int | None = None
    queue: list[tuple[list[int], int]] | None = None
    cursor = 0
    inflight: InFlight | None = None
    while step < cfg.max_steps:
        if order_index >= len(train_ds):
            if made is not None and step == made:
                raise RuntimeError("epoch produced no training steps")
            epoch += 1
            order_index = 0
            queue = None
            inflight = None
            skips = 0
            made = None
            continue
        if queue is None:
            batches = epoch_batches(
                rate1_frames,
                seed=cfg.seed,
                epoch=epoch,
                budget=cfg.frame_budget,
                frac=cfg.window_frac,
            )
            queue = _queue_from(batches, order_index)
            cursor = 0
            inflight = None
            made = step
            if not queue:
                order_index = len(train_ds)
                continue
        if inflight is None:
            if cursor >= len(queue):
                order_index = len(train_ds)
                continue
            indices, end_pos = queue[cursor]
            cursor += 1
            inflight = builder.submit(indices, end_pos)
        current = inflight
        if cursor < len(queue):
            nxt_indices, nxt_end = queue[cursor]
            cursor += 1
            inflight = builder.submit(nxt_indices, nxt_end)
        else:
            inflight = None
        examples = builder.wait(current)
        usable = [example for example in examples if example is not None]
        skipped = len(examples) - len(usable)
        if skipped:
            skips += skipped
            if skips > 32:
                raise RuntimeError(f"skipped {skips} charts in one epoch")
        order_index = current.end_pos
        rng_state = current.rng_after
        if not usable:
            continue
        batch = collate_v1(usable)
        _set_lr(
            opt,
            lr_at_step(step, base_lr=cfg.lr, warmup_steps=cfg.warmup_steps),
        )
        step_started = time.perf_counter()
        loss, acc, gnorm, peak = _step(
            model,
            opt,
            batch,
            device,
            use_amp=use_amp,
            grad_clip=cfg.grad_clip,
        )
        step_s = time.perf_counter() - step_started
        last["loss"] = loss
        last["acc"] = acc
        last["gnorm"] = gnorm
        last["peak_gib"] = peak
        last["batch"] = int(batch.token_ids.shape[0])
        last["frames"] = int(batch.n_frames.max()) if batch.n_frames is not None else 0
        last["tokens"] = int(batch.token_ids.shape[1])
        last["step_s"] = step_s
        step += 1
        if step == 1 or step % cfg.log_every == 0:
            _log_step(
                model,
                step,
                loss,
                acc,
                gnorm,
                opt.param_groups[0]["lr"],
                batch,
                peak,
                step_s,
            )
        if step % cfg.val_every == 0 and len(val_ds) > 0:
            if val_cache_box[0] is None:
                val_cache_box[0] = _build_val_cache(
                    val_ds, cfg.num_workers, seed=cfg.seed,
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            cache = val_cache_box[0]
            branches = (
                ("full", None),
                ("partial", cache.partial),
                ("none", cache.none),
            )
            readouts: dict[str, _ValReadout] = {}
            for branch, masks in branches:
                readout = _validate(
                    model, cache.charts, device, use_amp=use_amp, masks=masks,
                )
                readouts[branch] = readout
                _store_val(last, branch, readout)
                _log_val(step, branch, readout)
            mixed = _mix_readout(
                readouts["full"], readouts["partial"], readouts["none"],
            )
            last["val_mix_chart"] = mixed.loss_chart
            last["val_mix_token"] = mixed.loss_token
            last["val_mix_acc_chart"] = mixed.acc_chart
            last["val_mix_acc_token"] = mixed.acc_token
            logger.info(
                "val step %s mix loss_chart %.4f loss_token %.4f "
                "acc_chart %.3f acc_token %.3f",
                step,
                mixed.loss_chart,
                mixed.loss_token,
                mixed.acc_chart,
                mixed.acc_token,
            )
            model.train()
            if step % cfg.save_every != 0:
                logger.warning(
                    "val step %s was not saved; it does not count toward stopping",
                    step,
                )
        if step % cfg.save_every == 0:
            _save(
                model, opt, train_ds, step, epoch, order_index,
                ckpt_dir / f"step_{step}.pt",
                seed=cfg.seed,
                split_path=split_path,
                rng_state=rng_state,
            )
            if step % cfg.val_every == 0 and len(val_ds) > 0:
                best, stall, stop_now = _note_val(
                    best, stall, step, float(last["val_mix_token"]),
                )
                if best is not None:
                    _write_best(ckpt_dir / "best.json", best[0], best[1])
                if stop_now:
                    stop_reason = "patience"
                    logger.info(
                        "stop at step %s: mix loss_token did not improve "
                        "for %s vals (best step %s %.4f)",
                        step,
                        VAL_PATIENCE,
                        best[0] if best is not None else 0,
                        best[1] if best is not None else 0.0,
                    )
            _retain_checkpoints(ckpt_dir, None if best is None else best[0])
            if stop_reason == "patience":
                break
    return step, epoch, order_index, last, rng_state, stop_reason


def _queue_from(
    batches: list[list[int]],
    order_index: int,
) -> list[tuple[list[int], int]]:
    """Batches that still contain charts at or after ``order_index``."""
    pos = 0
    queued: list[tuple[list[int], int]] = []
    for batch in batches:
        start = pos
        pos += len(batch)
        if pos <= order_index:
            continue
        kept = batch[order_index - start :] if start < order_index else batch
        queued.append((kept, start + len(batch)))
    return queued


@dataclass(frozen=True, slots=True)
class _ValCache:
    """Full-condition features, plus the two masks that reuse them."""

    charts: list[V1Batch]
    partial: list[torch.Tensor]
    none: list[torch.Tensor]


def _build_val_cache(dataset: V1ChartDataset, workers: int, *, seed: int) -> _ValCache:
    """Rate-1 features for every validation chart. Kept on CPU and reused.

    ``charts`` keep ``cond_m = m_real``. Partial masks are frozen from ``seed``
    and the chart path. The empty masks are the unconditional branch.
    """
    logger.info("building val cache charts=%s workers=%s", len(dataset), workers)
    builder = ChartBuilder(dataset, workers)
    try:
        flight = builder.submit(list(range(len(dataset))), len(dataset))
        built = builder.wait(flight)
    finally:
        builder.close()
    charts: list[V1Batch] = []
    partial: list[torch.Tensor] = []
    dropped: list[torch.Tensor] = []
    skipped = 0
    for index, example in enumerate(built):
        if example is None:
            skipped += 1
            continue
        real = example.cond_m.detach().reshape(-1).cpu().numpy()
        drawn = frozen_partial_mask(
            real, seed=seed, key=dataset.charts[index].relpath,
        )
        mask = torch.from_numpy(np.ascontiguousarray(drawn))
        charts.append(example)
        partial.append(mask.reshape(example.cond_m.shape))
        dropped.append(torch.zeros_like(example.cond_m))
    if skipped:
        logger.warning("val cache skipped %s charts", skipped)
        if skipped > 32:
            raise RuntimeError(f"skipped {skipped} validation charts")
    if not charts:
        raise RuntimeError("validation produced no charts")
    logger.info("val cache ready charts=%s", len(charts))
    return _ValCache(charts=charts, partial=partial, none=dropped)


def _log_budget(
    lengths: dict[str, float | int],
    frame_budget: int,
    window_frac: float,
    workers: int,
) -> None:
    logger.info(
        "rate1 n=%s failed=%s frames p50 %.0f p90 %.0f max %.0f "
        "tokens p50 %.0f p90 %.0f max %.0f",
        lengths["n"],
        lengths["failed"],
        lengths["frame_p50"],
        lengths["frame_p90"],
        lengths["frame_max"],
        lengths["token_p50"],
        lengths["token_p90"],
        lengths["token_max"],
    )
    logger.info(
        "pack rate1 windows +%.0f%% n*max_frames<=%s workers=%s",
        100.0 * window_frac,
        frame_budget,
        workers,
    )


def _log_step(
    model: V1ChartModel,
    step: int,
    loss: float,
    acc: float,
    gnorm: float,
    lr: float,
    batch: V1Batch,
    peak: float,
    step_s: float,
) -> None:
    cross = model.last_cross
    impl = cross.impl if cross is not None else ""
    compiled = bool(cross.compiled) if cross is not None else False
    frame_span = _span(_per_example_frames(batch))
    token_span = _span(_per_example_tokens(batch))
    logger.info(
        "step %s loss %.4f acc %.3f gnorm %.2f lr %.3e batch %s frames %s tokens %s "
        "ca_train=%s grad_checkpoint=%s impl=%s compiled=%s peak_gib %.2f step_s %.2f",
        step,
        loss,
        acc,
        gnorm,
        lr,
        int(batch.token_ids.shape[0]),
        frame_span,
        token_span,
        model.ca_train,
        model.grad_checkpoint,
        impl,
        compiled,
        peak,
        step_s,
    )


def _per_example_frames(batch: V1Batch) -> list[int]:
    if batch.n_frames is None:
        return []
    return [int(value) for value in batch.n_frames.tolist()]


def _per_example_tokens(batch: V1Batch) -> list[int]:
    if batch.token_mask is None:
        return [_token_count(batch)]
    # token_mask covers token_ids[:, 1:], so the sequence length is one more.
    return [int(value) + 1 for value in batch.token_mask.sum(dim=1).tolist()]


def _span(values: list[int]) -> str:
    if not values:
        return "0"
    lo, hi = min(values), max(values)
    if lo == hi:
        return str(lo)
    return f"{lo}-{hi}"


def _step(
    model: V1ChartModel,
    opt: torch.optim.Optimizer,
    batch: V1Batch,
    device: torch.device,
    *,
    use_amp: bool,
    grad_clip: float,
) -> tuple[float, float, float, float]:
    batch = _to_device(batch, device)
    opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        logits = model(batch)
        targets = batch.token_ids[:, 1:]
        loss = plain_ce(logits, targets)
    if not torch.isfinite(loss.detach()):
        raise RuntimeError("non-finite loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not torch.isfinite(grad_norm):
        raise RuntimeError("non-finite gradient")
    opt.step()
    correct, total = accuracy_counts(logits, targets)["all"]
    acc = correct / total if total else 0.0
    peak = 0.0
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024**3)
    return float(loss.detach().item()), acc, float(grad_norm), peak


@dataclass(frozen=True, slots=True)
class _ValReadout:
    loss_chart: float
    loss_token: float
    acc_chart: float
    acc_token: float
    pos: float
    row: float
    beat: float
    eos: float
    legal_top1_chart: float = 0.0
    legal_top1_token: float = 0.0
    legal_prob_chart: float = 0.0
    legal_prob_token: float = 0.0
    legal_top1_pos: float = 0.0
    legal_top1_row: float = 0.0
    legal_top1_beat: float = 0.0
    legal_top1_eos: float = 0.0
    legal_prob_pos: float = 0.0
    legal_prob_row: float = 0.0
    legal_prob_beat: float = 0.0
    legal_prob_eos: float = 0.0
    loss_pos: float = 0.0
    loss_row: float = 0.0
    loss_beat: float = 0.0
    loss_eos: float = 0.0


@torch.no_grad()
def _validate(
    model: V1ChartModel,
    charts: list[V1Batch],
    device: torch.device,
    *,
    use_amp: bool,
    masks: list[torch.Tensor] | None = None,
) -> _ValReadout:
    """Per-chart means, then the token-weighted loss and accuracy.

    ``masks`` replaces ``cond_m`` and leaves the cached audio and tokens.
    """
    model.eval()
    chart_total = 0.0
    token_total = 0.0
    token_count = 0
    seen = 0
    chart_acc = 0.0
    chart_acc_n = 0
    correct = {name: 0 for name in ("all", "pos", "row", "beat", "eos")}
    counted = {name: 0 for name in correct}
    legal_hits = 0
    legal_mass = 0.0
    legal_count = 0
    legal_chart_top1 = 0.0
    legal_chart_prob = 0.0
    legal_charts = 0
    token_types = ("pos", "row", "beat", "eos")
    legal_type_hits = {name: 0 for name in token_types}
    legal_type_mass = {name: 0.0 for name in token_types}
    legal_type_count = {name: 0 for name in token_types}
    loss_type_sum = {name: 0.0 for name in token_types}
    loss_type_count = {name: 0 for name in token_types}
    for index, example in enumerate(charts):
        if masks is not None:
            example = replace(example, cond_m=masks[index])
        batch = _to_device(example, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(batch)
            targets = batch.token_ids[:, 1:]
            loss = plain_ce(logits, targets)
        count = nonpad_count(targets)
        value = float(loss.detach().item())
        chart_total += value
        token_total += value * count
        token_count += count
        seen += 1
        counts = accuracy_counts(logits, targets)
        hits, total = counts["all"]
        if total:
            chart_acc += hits / total
            chart_acc_n += 1
        for name, (name_hits, name_total) in counts.items():
            correct[name] += name_hits
            counted[name] += name_total
        for name, (type_sum, type_n) in ce_by_type(logits, targets).items():
            loss_type_sum[name] += type_sum
            loss_type_count[name] += type_n
        if example.legal_ids is None:
            raise RuntimeError("validation chart has no legal-id rows")
        tally = legal_tally(logits, targets, example.legal_ids)
        legal_hits += tally.hits
        legal_mass += tally.mass
        legal_count += tally.count
        if tally.count:
            legal_chart_top1 += tally.hits / tally.count
            legal_chart_prob += tally.mass / tally.count
            legal_charts += 1
        for name, (type_hits, type_mass, type_n) in tally.by_type.items():
            legal_type_hits[name] += type_hits
            legal_type_mass[name] += type_mass
            legal_type_count[name] += type_n
    if seen == 0 or token_count == 0:
        raise RuntimeError("validation produced no charts")

    def ratio(name: str) -> float:
        total = counted[name]
        return correct[name] / total if total else 0.0

    def loss_type(name: str) -> float:
        total = loss_type_count[name]
        return loss_type_sum[name] / total if total else 0.0

    def legal_type(name: str, *, mass: bool) -> float:
        total = legal_type_count[name]
        if not total:
            return 0.0
        value = legal_type_mass[name] if mass else legal_type_hits[name]
        return value / total

    return _ValReadout(
        loss_chart=chart_total / seen,
        loss_token=token_total / token_count,
        acc_chart=chart_acc / chart_acc_n if chart_acc_n else 0.0,
        acc_token=ratio("all"),
        pos=ratio("pos"),
        row=ratio("row"),
        beat=ratio("beat"),
        eos=ratio("eos"),
        legal_top1_chart=(
            legal_chart_top1 / legal_charts if legal_charts else 0.0
        ),
        legal_top1_token=legal_hits / legal_count if legal_count else 0.0,
        legal_prob_chart=(
            legal_chart_prob / legal_charts if legal_charts else 0.0
        ),
        legal_prob_token=legal_mass / legal_count if legal_count else 0.0,
        legal_top1_pos=legal_type("pos", mass=False),
        legal_top1_row=legal_type("row", mass=False),
        legal_top1_beat=legal_type("beat", mass=False),
        legal_top1_eos=legal_type("eos", mass=False),
        legal_prob_pos=legal_type("pos", mass=True),
        legal_prob_row=legal_type("row", mass=True),
        legal_prob_beat=legal_type("beat", mass=True),
        legal_prob_eos=legal_type("eos", mass=True),
        loss_pos=loss_type("pos"),
        loss_row=loss_type("row"),
        loss_beat=loss_type("beat"),
        loss_eos=loss_type("eos"),
    )


_VAL_BRANCHES = ("full", "partial", "none")
_VAL_FIELDS = (
    "chart",
    "token",
    "acc_chart",
    "acc_token",
    "pos",
    "row",
    "beat",
    "eos",
    "legal_top1_chart",
    "legal_top1_token",
    "legal_prob_chart",
    "legal_prob_token",
    "legal_top1_pos",
    "legal_top1_row",
    "legal_top1_beat",
    "legal_top1_eos",
    "legal_prob_pos",
    "legal_prob_row",
    "legal_prob_beat",
    "legal_prob_eos",
    "loss_pos",
    "loss_row",
    "loss_beat",
    "loss_eos",
)


def _blank_val_fields() -> dict[str, float]:
    fields = {
        f"val_{branch}_{name}": 0.0
        for branch in _VAL_BRANCHES
        for name in _VAL_FIELDS
    }
    fields["val_mix_chart"] = 0.0
    fields["val_mix_token"] = 0.0
    fields["val_mix_acc_chart"] = 0.0
    fields["val_mix_acc_token"] = 0.0
    return fields


def _store_val(last: dict[str, float | int], branch: str, readout: _ValReadout) -> None:
    values = {
        "chart": readout.loss_chart,
        "token": readout.loss_token,
        "acc_chart": readout.acc_chart,
        "acc_token": readout.acc_token,
        "pos": readout.pos,
        "row": readout.row,
        "beat": readout.beat,
        "eos": readout.eos,
        "legal_top1_chart": readout.legal_top1_chart,
        "legal_top1_token": readout.legal_top1_token,
        "legal_prob_chart": readout.legal_prob_chart,
        "legal_prob_token": readout.legal_prob_token,
        "legal_top1_pos": readout.legal_top1_pos,
        "legal_top1_row": readout.legal_top1_row,
        "legal_top1_beat": readout.legal_top1_beat,
        "legal_top1_eos": readout.legal_top1_eos,
        "legal_prob_pos": readout.legal_prob_pos,
        "legal_prob_row": readout.legal_prob_row,
        "legal_prob_beat": readout.legal_prob_beat,
        "legal_prob_eos": readout.legal_prob_eos,
        "loss_pos": readout.loss_pos,
        "loss_row": readout.loss_row,
        "loss_beat": readout.loss_beat,
        "loss_eos": readout.loss_eos,
    }
    for name, value in values.items():
        last[f"val_{branch}_{name}"] = value


def _log_val(step: int, branch: str, readout: _ValReadout) -> None:
    logger.info(
        "val step %s %s loss_chart %.4f loss_token %.4f "
        "loss_POS %.4f loss_ROW %.4f loss_BEAT %.4f loss_EOS %.4f "
        "acc_chart %.3f acc_token %.3f POS %.3f ROW %.3f BEAT %.3f EOS %.3f "
        "legal_top1_chart %.3f legal_top1 %.3f "
        "legal_prob_chart %.3f legal_prob %.3f "
        "legal_top1_POS %.3f legal_top1_ROW %.3f "
        "legal_top1_BEAT %.3f legal_top1_EOS %.3f "
        "legal_prob_POS %.3f legal_prob_ROW %.3f "
        "legal_prob_BEAT %.3f legal_prob_EOS %.3f",
        step,
        branch,
        readout.loss_chart,
        readout.loss_token,
        readout.loss_pos,
        readout.loss_row,
        readout.loss_beat,
        readout.loss_eos,
        readout.acc_chart,
        readout.acc_token,
        readout.pos,
        readout.row,
        readout.beat,
        readout.eos,
        readout.legal_top1_chart,
        readout.legal_top1_token,
        readout.legal_prob_chart,
        readout.legal_prob_token,
        readout.legal_top1_pos,
        readout.legal_top1_row,
        readout.legal_top1_beat,
        readout.legal_top1_eos,
        readout.legal_prob_pos,
        readout.legal_prob_row,
        readout.legal_prob_beat,
        readout.legal_prob_eos,
    )


def _mix_readout(
    full: _ValReadout,
    partial: _ValReadout,
    dropped: _ValReadout,
) -> _ValReadout:
    """Training mixture: keep-all, Bernoulli partial, drop-all.

    Type accuracies and type losses stay on each branch. The mixture carries
    the two overall losses and the two overall accuracies only.
    """
    partial_p = 1.0 - KEEP_ALL_P - DROP_ALL_P

    def mix(full_value: float, partial_value: float, dropped_value: float) -> float:
        return (
            KEEP_ALL_P * full_value
            + partial_p * partial_value
            + DROP_ALL_P * dropped_value
        )

    return _ValReadout(
        loss_chart=mix(full.loss_chart, partial.loss_chart, dropped.loss_chart),
        loss_token=mix(full.loss_token, partial.loss_token, dropped.loss_token),
        acc_chart=mix(full.acc_chart, partial.acc_chart, dropped.acc_chart),
        acc_token=mix(full.acc_token, partial.acc_token, dropped.acc_token),
        pos=0.0,
        row=0.0,
        beat=0.0,
        eos=0.0,
    )


def _to_device(batch: V1Batch, device: torch.device) -> V1Batch:
    def move(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value.to(device, non_blocking=True)

    return V1Batch(
        frame_lo=move(batch.frame_lo),
        cond_x=move(batch.cond_x),
        cond_m=move(batch.cond_m),
        token_ids=move(batch.token_ids),
        cursor_time_ms=move(batch.cursor_time_ms),
        phi=move(batch.phi),
        beat_b=move(batch.beat_b),
        log_bpm=move(batch.log_bpm),
        rel_u=move(batch.rel_u),
        coverage_start_ms=move(batch.coverage_start_ms),
        coverage_end_ms=move(batch.coverage_end_ms),
        redline_offset_ms=move(batch.redline_offset_ms),
        redline_beat_ms=move(batch.redline_beat_ms),
        audio_mask=move(batch.audio_mask),
        token_mask=move(batch.token_mask),
        mel=move(batch.mel),
        n_frames=move(batch.n_frames),
        legal_ids=batch.legal_ids,
    )


_CKPT_KEYS = frozenset(
    {
        "step",
        "epoch",
        "order_index",
        "model",
        "optimizer",
        "ca_train",
        "torch_rng",
        "cuda_rng",
        "train_rng",
        "seed",
        "n_train",
        "split_manifest",
    }
)


def _note_val(
    best: tuple[int, float] | None,
    stall: int,
    step: int,
    score: float,
) -> tuple[tuple[int, float], int, bool]:
    """Strict improvement of ``mix`` loss_token resets the stall."""
    if best is None or score < best[1]:
        return (step, score), 0, False
    stall += 1
    return best, stall, stall >= VAL_PATIENCE


def _best_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "best.json"


def _load_best(path: Path) -> tuple[int, float] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data["step"]), float(data["mix_loss_token"])


def _best_step(ckpt_dir: Path) -> int | None:
    best = _load_best(_best_path(ckpt_dir))
    if best is None:
        return None
    return best[0]


def _write_best(path: Path, step: int, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    temporary.write_text(
        json.dumps({"step": step, "mix_loss_token": score}),
        encoding="utf-8",
    )
    temporary.replace(path)


def _retain_checkpoints(ckpt_dir: Path, best_step: int | None) -> None:
    """Keep the newest ``KEEP_CHECKPOINTS`` files plus the best step."""
    found: list[tuple[int, Path]] = []
    for path in ckpt_dir.glob("step_*.pt"):
        suffix = path.stem.removeprefix("step_")
        if not suffix.isdecimal():
            continue
        found.append((int(suffix), path))
    found.sort()
    keep = {step for step, _path in found[-KEEP_CHECKPOINTS:]}
    if best_step is not None:
        keep.add(best_step)
    for step, path in found:
        if step not in keep:
            path.unlink()


def _save(
    model: V1ChartModel,
    opt: torch.optim.Optimizer,
    dataset: V1ChartDataset,
    step: int,
    epoch: int,
    order_index: int,
    path: Path,
    *,
    seed: int,
    split_path: Path,
    rng_state: dict | None = None,
) -> None:
    """Atomic write, same replace pattern as the v0 checkpoint helper.

    This loop does not use a GradScaler. bf16 autocast has no loss scale, so
    the file has no ``scaler`` field.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "epoch": epoch,
        "order_index": order_index,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "ca_train": model.ca_train,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
        "train_rng": dataset.rng_state() if rng_state is None else rng_state,
        "seed": seed,
        "n_train": len(dataset),
        "split_manifest": str(split_path.resolve()),
    }
    temporary = path.with_name(f"{path.name}.part")
    torch.save(payload, temporary)
    temporary.replace(path)
    logger.info("saved %s", path)


def load_train_checkpoint(
    path: Path,
    model: V1ChartModel,
    opt: torch.optim.Optimizer,
    dataset: V1ChartDataset,
    *,
    device: torch.device,
    seed: int,
    n_train: int,
    split_path: Path,
) -> tuple[int, int, int]:
    """Restore a v1 ``step_N.pt``. Returns completed step, epoch, and order index."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"checkpoint is not a dict: {path}")
    missing = [key for key in ("step", "model", "optimizer") if key not in ckpt]
    if missing:
        raise RuntimeError(f"checkpoint {path} is missing {missing}")
    extra = sorted(set(ckpt) - _CKPT_KEYS)
    if extra:
        logger.warning("checkpoint has keys this loop does not restore: %s", extra)
    if "scaler" in ckpt:
        logger.warning("checkpoint scaler is unused; v1 bf16 training has no GradScaler")

    restart_order = _check_resume_identity(
        ckpt, path, seed=seed, n_train=n_train, split_path=split_path
    )
    legacy = load_v1_state_dict(model, ckpt["model"])
    load_v1_optimizer(model, opt, ckpt["optimizer"], legacy=legacy)
    if "ca_train" in ckpt:
        model.set_ca_train(str(ckpt["ca_train"]))
    else:
        logger.warning("checkpoint has no ca_train; keeping %s", model.ca_train)
    step = int(ckpt["step"])
    epoch, order_index = _restore_cursor(
        ckpt, dataset, device, restart_order=restart_order
    )
    logger.info(
        "resumed %s at step %s epoch %s order_index %s",
        path,
        step,
        epoch,
        order_index,
    )
    return step, epoch, order_index


def _maybe_resume(
    cfg: V1TrainConfig,
    model: V1ChartModel,
    opt: torch.optim.Optimizer,
    dataset: V1ChartDataset,
    *,
    device: torch.device,
    n_train: int,
    split_path: Path,
) -> tuple[int, int, int]:
    if cfg.resume is None:
        return 0, 0, 0
    return load_train_checkpoint(
        cfg.resume,
        model,
        opt,
        dataset,
        device=device,
        seed=cfg.seed,
        n_train=n_train,
        split_path=split_path,
    )


def _check_resume_identity(
    ckpt: dict,
    path: Path,
    *,
    seed: int,
    n_train: int,
    split_path: Path,
) -> bool:
    """True when the saved chart cursor belongs to a different chart list."""
    if "seed" in ckpt and int(ckpt["seed"]) != seed:
        raise RuntimeError(
            f"checkpoint seed {ckpt['seed']} != run seed {seed} ({path})"
        )
    saved_split = ckpt.get("split_manifest")
    if saved_split is not None and Path(saved_split).resolve() != split_path.resolve():
        raise RuntimeError(
            f"checkpoint split {saved_split} != this run {split_path} ({path})"
        )
    if "seed" not in ckpt or "n_train" not in ckpt:
        logger.warning("checkpoint has no seed or n_train; not checking the data order")
    if "n_train" in ckpt and int(ckpt["n_train"]) != n_train:
        logger.warning(
            "checkpoint n_train %s != this run %s; data order starts again at epoch 0",
            ckpt["n_train"],
            n_train,
        )
        return True
    return False


def _restore_cursor(
    ckpt: dict,
    dataset: V1ChartDataset,
    device: torch.device,
    *,
    restart_order: bool = False,
) -> tuple[int, int]:
    have_cursor = all(key in ckpt for key in ("epoch", "order_index", "train_rng"))
    if restart_order:
        epoch = 0
        order_index = 0
    elif have_cursor:
        dataset.set_rng_state(ckpt["train_rng"])
        epoch = int(ckpt["epoch"])
        order_index = int(ckpt["order_index"])
    else:
        logger.warning(
            "checkpoint has no epoch cursor and train RNG; data order starts again at epoch 0"
        )
        epoch = 0
        order_index = 0
    if "torch_rng" in ckpt:
        torch.set_rng_state(ckpt["torch_rng"].cpu())
    else:
        logger.warning("checkpoint has no torch RNG")
    if device.type == "cuda":
        if ckpt.get("cuda_rng") is not None:
            # map_location moves every tensor, including these byte states.
            torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng"]])
        else:
            logger.warning("checkpoint has no CUDA RNG")
    return epoch, order_index
