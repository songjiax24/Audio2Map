"""CLI argparse / config helpers (no model or dataset)."""

from __future__ import annotations

import argparse

import pytest

from audio2map.cli.common import parse_args_with_config
from audio2map.cli.infer import _check_args, build_parser
from audio2map.generate.overlap import DecodeConfig, GenerationRangeConfig, OverlapConfig


def test_config_yaml_loses_to_cli(tmp_path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("limit: 3\nseed: 1\n")
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    ns = parse_args_with_config(p, ["--config", str(cfg), "--limit", "9"])
    assert ns.limit == 9
    assert ns.seed == 1


def test_locked_config_key_must_match(tmp_path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("window_bars: 32\nlimit: 2\n")
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--limit", type=int, default=100)
    with pytest.raises(SystemExit, match="window_bars"):
        parse_args_with_config(p, ["--config", str(cfg)], locked={"window_bars": 16})


def test_locked_config_key_matching_is_silent(tmp_path, capsys) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("window_bars: 16\nlimit: 2\n")
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--limit", type=int, default=100)
    ns = parse_args_with_config(p, ["--config", str(cfg)], locked={"window_bars": 16})
    assert ns.limit == 2
    assert "window_bars" not in capsys.readouterr().err
    assert not hasattr(ns, "window_bars")


def test_unknown_config_keys_are_ignored(tmp_path, capsys) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("window_bars: 16\nlimit: 2\n")
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--limit", type=int, default=100)
    ns = parse_args_with_config(p, ["--config", str(cfg)])
    assert ns.limit == 2
    err = capsys.readouterr().err
    assert "window_bars" in err


def test_infer_overlap_is_locked_to_overlap_config() -> None:
    dests = {a.dest for a in build_parser()._actions}
    assert "window_bars" not in dests
    assert "context_bars" not in dests
    assert "keep_bars" not in dests
    assert "future_bars" not in dests
    assert "max_seq_len" not in dests
    overlap = OverlapConfig()
    decode = DecodeConfig()
    ns = build_parser().parse_args(["--osu", "a.osu", "--checkpoint", "c.pt"])
    assert ns.temperature == decode.temperature
    assert ns.top_p == decode.top_p
    assert ns.top_k == decode.top_k
    assert ns.post_margin_bars == GenerationRangeConfig().post_margin_bars
    assert overlap.window_bars == 16
    assert overlap.context_bars + overlap.keep_bars + overlap.future_bars == overlap.window_bars


def test_infer_config_yaml_locks_overlap(tmp_path) -> None:
    from audio2map.cli.infer import _LOCKED_OVERLAP

    cfg = tmp_path / "v2.yaml"
    cfg.write_text("window_bars: 16\ncontext_bars: 8\nkeep_bars: 4\nfuture_bars: 4\nmax_seq_len: 2048\n")
    ns = parse_args_with_config(
        build_parser(),
        ["--config", str(cfg), "--osu", "a.osu", "--checkpoint", "c.pt"],
        locked=_LOCKED_OVERLAP,
    )
    assert ns.osu == "a.osu"

    bad = tmp_path / "bad.yaml"
    bad.write_text("window_bars: 32\n")
    with pytest.raises(SystemExit, match="window_bars"):
        parse_args_with_config(
            build_parser(),
            ["--config", str(bad), "--osu", "a.osu", "--checkpoint", "c.pt"],
            locked=_LOCKED_OVERLAP,
        )


def test_start_bar_requires_single_window() -> None:
    ns = build_parser().parse_args(
        ["--osu", "a.osu", "--checkpoint", "c.pt", "--start-bar", "0"]
    )
    with pytest.raises(SystemExit, match="single-window"):
        _check_args(ns)
    ns.single_window = True
    _check_args(ns)
