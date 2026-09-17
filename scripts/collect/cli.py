"""Sayobot mania 4K collection — tools script entry (not a core-package CLI).

Run::

    python -m scripts.collect.cli --target 1000
    python -m scripts.collect.cli --config configs/data/collector.yaml --status
"""

from __future__ import annotations

import argparse

from audio2map.cli.common import parse_args_with_config
from scripts.collect.sayobot import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download osu! mania 4K beatmaps (Sayobot)")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config supplying defaults (e.g. configs/data/collector.yaml)",
    )
    p.add_argument("--page-size", type=int, default=50, help="List API page size (L)")
    p.add_argument("--list-delay", type=float, default=1.0, help="Delay after each list page (seconds)")
    p.add_argument(
        "--target",
        type=int,
        default=None,
        help="max unseen list entries this run (known sids do not count); resumes from saved offset",
    )
    p.add_argument("--max-pages", type=int, default=None, help="Stop after this many list pages")
    p.add_argument("--status", action="store_true", help="Print state.json stats and exit")
    return p


def main() -> None:
    args = parse_args_with_config(build_parser())
    run(
        target=args.target,
        page_size=args.page_size,
        list_delay=args.list_delay,
        max_pages=args.max_pages,
        status_only=args.status,
    )


if __name__ == "__main__":
    main()
