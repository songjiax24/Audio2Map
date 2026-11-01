"""v1 checkpoint keeps weights, Adam state, step, and the train RNG."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from audio2map.train.v1.data import V1ChartDataset
from audio2map.train.v1.loop import _save, load_train_checkpoint
from audio2map.model.v1.model import V1ChartModel


def _stepped_pair(lr: float = 1e-3) -> tuple[V1ChartModel, torch.optim.Optimizer]:
    model = V1ChartModel()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.token_emb.weight.sum().backward()
    opt.step()
    return model, opt


def _dataset(root: Path, seed: int) -> V1ChartDataset:
    return V1ChartDataset([], root=root, augment=True, cond_mask="train", seed=seed)


def test_resume_restores_weights_optimizer_and_rng(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model, opt = _stepped_pair()
    dataset = _dataset(tmp_path, seed=3)
    float(dataset._rng.random())
    torch.rand(1)
    path = tmp_path / "step_1.pt"
    split = tmp_path / "split.json"
    _save(
        model, opt, dataset, 1, 2, 4, path,
        seed=3,
        split_path=split,
    )
    next_torch = torch.rand(4).clone()
    next_data = float(dataset._rng.random())
    assert not path.with_name("step_1.pt.part").exists()

    torch.manual_seed(99)
    restored, restored_opt = _stepped_pair(lr=1e-4)
    restored_ds = _dataset(tmp_path, seed=0)
    step, epoch, order_index = load_train_checkpoint(
        path,
        restored,
        restored_opt,
        restored_ds,
        device=torch.device("cpu"),
        seed=3,
        n_train=0,
        split_path=split,
    )
    assert (step, epoch, order_index) == (1, 2, 4)
    assert torch.equal(restored.token_emb.weight, model.token_emb.weight)
    saved_adam = next(iter(opt.state_dict()["state"].values()))
    loaded_adam = next(iter(restored_opt.state_dict()["state"].values()))
    assert torch.equal(loaded_adam["step"], saved_adam["step"])
    assert torch.equal(loaded_adam["exp_avg"], saved_adam["exp_avg"])
    assert torch.equal(torch.rand(4), next_torch)
    assert float(restored_ds._rng.random()) == next_data
    assert restored.ca_train == "flex"


def test_resume_restarts_order_when_the_train_count_changes(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model, opt = _stepped_pair()
    dataset = _dataset(tmp_path, seed=1)
    path = tmp_path / "step_2.pt"
    split = tmp_path / "split.json"
    _save(model, opt, dataset, 2, 4, 9, path, seed=1, split_path=split)
    other, other_opt = _stepped_pair()
    with caplog.at_level(logging.WARNING):
        step, epoch, order_index = load_train_checkpoint(
            path,
            other,
            other_opt,
            _dataset(tmp_path, seed=1),
            device=torch.device("cpu"),
            seed=1,
            n_train=3,
            split_path=split,
        )
    assert (step, epoch, order_index) == (2, 0, 0)
    assert torch.equal(other.token_emb.weight, model.token_emb.weight)
    assert "n_train" in caplog.text
    assert "epoch 0" in caplog.text


def test_old_checkpoint_keeps_weights_and_warns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model, opt = _stepped_pair()
    path = tmp_path / "step_4.pt"
    torch.save(
        {
            "step": 4,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "ca_train": "flex",
        },
        path,
    )
    restored, restored_opt = _stepped_pair()
    with caplog.at_level(logging.WARNING):
        step, epoch, order_index = load_train_checkpoint(
            path,
            restored,
            restored_opt,
            _dataset(tmp_path, seed=0),
            device=torch.device("cpu"),
            seed=0,
            n_train=0,
            split_path=tmp_path / "missing.json",
        )
    assert (step, epoch, order_index) == (4, 0, 0)
    assert torch.equal(restored.token_emb.weight, model.token_emb.weight)
    assert "data order starts again" in caplog.text
