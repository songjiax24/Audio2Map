"""Checkpoint retention, val patience, and AdamW decay groups."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from audio2map.dataset.split import ChartSnapshot
from audio2map.train.v1.data import V1ChartDataset
from audio2map.train.v1.loop import (
    KEEP_CHECKPOINTS,
    VAL_PATIENCE,
    _note_val,
    _retain_checkpoints,
    adamw_groups,
)


def test_val_plan_defers_the_condition(tmp_path: Path) -> None:
    dataset = V1ChartDataset(
        [ChartSnapshot("missing.osu", "g", "val")],
        root=tmp_path,
        augment=False,
        cond_mask="none",
        seed=0,
    )
    planned = dataset.plan(0)
    assert planned.meta is None
    assert planned.rate == 1.0
    assert planned.mirror is False
    assert planned.cond_mask is None


def test_retain_keeps_last_three_and_the_best(tmp_path: Path) -> None:
    for step in (2000, 4000, 6000, 8000):
        (tmp_path / f"step_{step}.pt").write_bytes(b"x")
    _retain_checkpoints(tmp_path, best_step=2000)
    kept = sorted(path.name for path in tmp_path.glob("step_*.pt"))
    assert kept == [
        "step_2000.pt",
        "step_4000.pt",
        "step_6000.pt",
        "step_8000.pt",
    ]
    assert KEEP_CHECKPOINTS == 3


def test_retain_drops_an_old_file_when_best_is_recent(tmp_path: Path) -> None:
    for step in (2000, 4000, 6000, 8000):
        (tmp_path / f"step_{step}.pt").write_bytes(b"x")
    _retain_checkpoints(tmp_path, best_step=6000)
    kept = sorted(path.name for path in tmp_path.glob("step_*.pt"))
    assert kept == ["step_4000.pt", "step_6000.pt", "step_8000.pt"]


def test_patience_stops_after_a_run_of_misses() -> None:
    best, stall, stop = _note_val(None, 0, 2000, 2.0)
    assert (best, stall, stop) == ((2000, 2.0), 0, False)
    best, stall, stop = _note_val(best, stall, 4000, 1.5)
    assert best == (4000, 1.5)
    assert stop is False
    for index in range(1, VAL_PATIENCE):
        best, stall, stop = _note_val(best, stall, 4000 + 2000 * index, 1.6)
        assert best == (4000, 1.5)
        assert stall == index
        assert stop is False
    best, stall, stop = _note_val(best, stall, 4000 + 2000 * VAL_PATIENCE, 1.6)
    assert best == (4000, 1.5)
    assert stall == VAL_PATIENCE
    assert stop is True


def test_adamw_decays_matrices_only() -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 4, bias=True)
            self.norm = nn.LayerNorm(4)

    groups = adamw_groups(Net(), 0.01)
    decay = groups[0]["params"]
    plain = groups[1]["params"]
    assert groups[0]["weight_decay"] == 0.01
    assert groups[1]["weight_decay"] == 0.0
    assert decay and all(param.ndim >= 2 for param in decay)
    assert plain and all(param.ndim < 2 for param in plain)
    assert sum(param.numel() for param in decay + plain) == sum(
        param.numel() for param in Net().parameters()
    )
