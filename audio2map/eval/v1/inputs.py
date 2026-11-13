"""Select a declared split and freeze the v1 evaluation pool and donor map."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from audio2map.dataset.donor import (
    assert_exact_donor_universe,
    create_donor_map,
    require_donor_map,
    save_donor_map,
)
from audio2map.dataset.split import ChartSnapshot, load_split_manifest
from audio2map.dataset.v1.gate import filter_v1_charts
from audio2map.features.cond.v1.store import ensure_v1_chart_meta, load_v1_chart_meta


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches_digest(path: Path, digest: str | None) -> bool:
    return digest is not None and path.is_file() and file_digest(path) == digest


def verify_training_corpus(corpus: dict, split_manifest: Path, root: Path) -> None:
    """Check that a fitted vocabulary belongs to this dataset and split."""
    if corpus.get("split") != "train" or corpus.get("split_sha256") != file_digest(split_manifest):
        raise ValueError("motif training corpus differs from the current split manifest")
    if not corpus.get("raw_root") or Path(corpus["raw_root"]).resolve() != root.resolve():
        raise ValueError("motif training corpus differs from the current raw root")
    train = {c.relpath for c in load_split_manifest(split_manifest).charts if c.split == "train"}
    charts = corpus.get("charts")
    if not charts or not set(charts) <= train:
        raise ValueError(
            "motif training corpus is empty or contains charts outside the train split"
        )


def select_charts(
    manifest_path: Path,
    root: Path,
    split: str,
    *,
    limit: int | None,
    seed: int,
    paths: tuple[Path, ...] = (),
) -> list[ChartSnapshot]:
    manifest = load_split_manifest(manifest_path)
    charts = sorted(
        (chart for chart in manifest.charts if chart.split == split),
        key=lambda chart: chart.relpath,
    )
    if paths:
        wanted = set()
        for path in paths:
            full = path if path.is_absolute() else root / path
            wanted.add(full.resolve().relative_to(root.resolve()).as_posix())
        available = {chart.relpath for chart in charts}
        if not wanted <= available:
            raise ValueError(
                f"requested charts are not in split {split}: {sorted(wanted - available)}"
            )
        charts = [chart for chart in charts if chart.relpath in wanted]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        charts = sorted(
            random.Random(seed).sample(charts, min(limit, len(charts))),
            key=lambda chart: chart.relpath,
        )
    if not charts:
        raise ValueError(f"no charts in split {split}")
    for chart in charts:
        if not (root / chart.relpath).resolve().is_relative_to(root.resolve()):
            raise ValueError(f"chart escapes raw root: {chart.relpath}")
    return charts


def prepare_pool(
    charts: list[ChartSnapshot], root: Path, out: Path, *, seed: int, workers: int
) -> tuple[list[ChartSnapshot], dict[str, str], dict]:
    rejected: dict[str, str] = {}
    kept, _ = filter_v1_charts(
        charts, root, workers=workers, rejections=rejected, gate_dir=out / "gate"
    )
    if not kept:
        raise ValueError("no usable v1 charts in the requested evaluation pool")
    targets = [chart.relpath for chart in kept]
    pool = {
        "requested": [chart.relpath for chart in charts],
        "accepted": targets,
        "rejected": rejected,
        "reject_counts": dict(Counter(rejected.values())),
    }
    freeze_json(out / "pool.json", pool)
    donor_path = out / "donors.json"
    if donor_path.exists():
        donors = require_donor_map(donor_path, targets, seed=seed)
    else:
        donors = create_donor_map(targets, seed=seed)
        save_donor_map(donors, donor_path)
    assert_exact_donor_universe(donors, targets)
    if not set(donors.pairs.values()) <= set(targets):
        raise ValueError("v1 donor map contains a donor outside this evaluation pool")
    pool["self_draws"] = sum(target == donor for target, donor in donors.pairs.items())
    return kept, donors.pairs, pool


def freeze_json(path: Path, data: dict) -> None:
    """A run may be repeated, but its saved inputs cannot silently change."""
    encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(encoded):
            raise ValueError(f"frozen inputs differ: {path}; choose another output directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def prepare_metadata(charts: list[ChartSnapshot], root: Path, out: Path, *, workers: int) -> dict:
    """Reuse the parallel metadata builder, scoped to this frozen evaluation run."""
    path = out / "metadata.jsonl"
    ensure_v1_chart_meta([chart.relpath for chart in charts], root, workers=workers, path=path)
    return load_v1_chart_meta(path)
