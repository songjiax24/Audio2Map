"""Shared CLI helpers: ``--config`` YAML loading and logging setup."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any


def load_config_file(path: Path) -> dict[str, Any]:
    """Load a YAML config mapping (empty file → ``{}``)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
    *,
    key_map: dict[str, str] | None = None,
    locked: dict[str, Any] | None = None,
) -> argparse.Namespace:
    """Parse args where ``--config`` YAML supplies defaults and CLI flags win.

    ``key_map`` renames YAML keys to argparse dests (e.g. ``n_heads`` →
    ``heads``). Keys in ``locked`` are not CLI options: they must match the
    given value if present, otherwise they are ignored with a warning.
    """
    base, _ = parser.parse_known_args(argv)
    config_path = getattr(base, "config", None)
    if config_path:
        data = load_config_file(Path(config_path))
        if key_map:
            data = {key_map.get(k, k): v for k, v in data.items()}
        locked = locked or {}
        for key, expected in locked.items():
            if key in data and data[key] != expected:
                raise SystemExit(
                    f"config {key}={data[key]!r} does not match {expected!r} (not configurable)"
                )
        known = {a.dest for a in parser._actions}
        unknown = sorted(k for k in data if k not in known and k not in locked)
        if unknown:
            print(f"warning: ignoring unknown config keys {unknown}", file=sys.stderr)
        parser.set_defaults(**{k: v for k, v in data.items() if k in known})
    return parser.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
