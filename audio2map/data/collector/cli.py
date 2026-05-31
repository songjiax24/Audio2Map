"""CLI for Sayobot mania 4K collection."""

from __future__ import annotations

import argparse

from audio2map.data.collector.sayobot import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download osu! mania 4K beatmaps (Sayobot)")
    p.add_argument("--page-size", type=int, default=50, help="List API page size (L)")
    p.add_argument("--list-delay", type=float, default=1.0, help="Delay after each list page (seconds)")
    p.add_argument(
        "--download-delay",
        type=float,
        default=2.0,
        help="Minimum delay after each download; scales with file size (cap 30s)",
    )
    p.add_argument(
        "--target",
        type=int,
        default=None,
        help="Process N list entries this run (includes skips, not only successes)",
    )
    p.add_argument("--max-pages", type=int, default=None, help="Stop after this many list pages")
    p.add_argument("--status", action="store_true", help="Print state.json stats and exit")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(
        target=args.target,
        page_size=args.page_size,
        list_delay=args.list_delay,
        download_delay=args.download_delay,
        max_pages=args.max_pages,
        status_only=args.status,
    )


if __name__ == "__main__":
    main()
