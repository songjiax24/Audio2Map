"""Checkpoint retention, val patience, and AdamW decay groups."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from audio2map.dataset.split import ChartSnapshot
from audio2map.dataset.v1.augment import AugmentChoice
from audio2map.features.cond.chart import ChartMeta
from audio2map.features.cond.v1.rate import CLOCK_SCALAR_FIELDS
from audio2map.train.v1.data import V1ChartDataset
from audio2map.train.v1.prepare import PlannedChart, build_planned
from audio2map.train.v1.loop import (
    KEEP_CHECKPOINTS,
    VAL_PATIENCE,
    _note_val,
    _retain_checkpoints,
    adamw_groups,
)


def _ready_meta() -> ChartMeta:
    values = {name: 1.0 for name in CLOCK_SCALAR_FIELDS}
    return ChartMeta(osu_path="a.osu", **values)


def test_plan_defers_a_complete_rate_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "audio2map.train.v1.data.sample_augment",
        lambda rng, enabled: AugmentChoice(mirror=False, rate=1.25),
    )

    def forbidden(*_args: object, **_kwargs: object) -> ChartMeta:
        raise AssertionError("parent rated a complete meta")

    monkeypatch.setattr("audio2map.train.v1.data.scalars_at_rate", forbidden)
    dataset = V1ChartDataset(
        [ChartSnapshot("a.osu", "g", "train")],
        root=tmp_path,
        augment=True,
        cond_mask="train",
        seed=0,
    )
    dataset._meta["a.osu"] = _ready_meta()
    dataset._meta_loaded = True
    planned = dataset.plan(0)
    assert planned.rate == 1.25
    assert planned.rate_scalars is True
    assert planned.meta is not None
    assert planned.meta["official_sr"] == 1.0
    assert planned.cond_mask is not None


def test_plan_rates_on_the_parent_when_a_scalar_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "audio2map.train.v1.data.sample_augment",
        lambda rng, enabled: AugmentChoice(mirror=False, rate=1.25),
    )
    missing = replace(_ready_meta(), official_sr=None)

    def fill(path: Path, meta: ChartMeta, rate: float) -> ChartMeta:
        assert rate == 1.25
        assert path.name == "a.osu"
        return replace(meta, official_sr=4.0)

    monkeypatch.setattr("audio2map.train.v1.data.scalars_at_rate", fill)
    dataset = V1ChartDataset(
        [ChartSnapshot("a.osu", "g", "train")],
        root=tmp_path,
        augment=True,
        cond_mask="train",
        seed=0,
    )
    dataset._meta["a.osu"] = missing
    dataset._meta_loaded = True
    planned = dataset.plan(0)
    assert planned.rate_scalars is False
    assert planned.meta is not None
    assert planned.meta["official_sr"] == 4.0


def test_worker_applies_deferred_scalars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_load(path: Path) -> object:
        return type(
            "Loaded",
            (),
            {
                "notes": (),
                "redlines": (),
                "pcm": None,
                "t_audio_ms": 1000.0,
                "audio_key": None,
            },
        )()

    def fake_clock(
        notes: object,
        redlines: object,
        pcm: object,
        t_audio: float,
        choice: AugmentChoice,
        relpath: str | None = None,
    ) -> object:
        return type(
            "Clocked",
            (),
            {"notes": (), "redlines": (), "pcm": None, "encoded": object()},
        )()

    def fake_build(source: object, **kwargs: object) -> str:
        seen["meta"] = kwargs["meta"]
        return "batch"

    def fake_rate(path: Path, meta: ChartMeta, rate: float) -> ChartMeta:
        seen["rate"] = rate
        return replace(meta, official_sr=9.0)

    monkeypatch.setattr("audio2map.train.v1.prepare.load_chart_audio", fake_load)
    monkeypatch.setattr("audio2map.train.v1.prepare.clock_chart", fake_clock)
    monkeypatch.setattr("audio2map.train.v1.prepare.build_v1_example", fake_build)
    monkeypatch.setattr("audio2map.features.cond.v1.rate.scalars_at_rate", fake_rate)
    planned = PlannedChart(
        relpath="a.osu",
        mirror=False,
        rate=1.25,
        meta=_ready_meta().to_dict(),
        cond_mask=[0.0] * 18,
        rate_scalars=True,
    )
    assert build_planned(str(tmp_path), planned) == "batch"
    assert seen["rate"] == 1.25
    meta = seen["meta"]
    assert isinstance(meta, ChartMeta)
    assert meta.official_sr == 9.0


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
