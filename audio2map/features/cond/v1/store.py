"""Rate-1 v1 ChartMeta on disk. v0 ``chart_meta/manifest.jsonl`` is a different file."""

from __future__ import annotations

import fcntl
import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from audio2map.features.cond.chart import ChartMeta, ChartMetaVersionError, compute_chart_meta
from audio2map.utils.paths import v1_chart_meta_path

logger = logging.getLogger(__name__)

_CACHE: dict[str, ChartMeta] | None = None


def load_v1_chart_meta(path: Path | None = None) -> dict[str, ChartMeta]:
    """Return ``relpath -> ChartMeta``. A missing file is an empty table."""
    manifest = path or v1_chart_meta_path()
    if not manifest.is_file():
        return {}
    rows: dict[str, ChartMeta] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                relpath = str(record["relpath"])
                rows[relpath] = ChartMeta.from_dict(record)
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ChartMetaVersionError,
            ) as exc:
                logger.warning("skip v1 chart meta line %s: %s", line_no, exc)
    return rows


def append_v1_chart_meta_records(
    records: list[dict],
    path: Path | None = None,
) -> None:
    """Append records whose ``relpath`` is not already in the file."""
    if not records:
        return
    manifest = path or v1_chart_meta_path()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest.with_name(manifest.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            have = set(load_v1_chart_meta(manifest))
            with manifest.open("a", encoding="utf-8") as handle:
                for record in records:
                    relpath = str(record["relpath"])
                    if relpath in have:
                        continue
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    have.add(relpath)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def measure_v1_chart_meta(job: tuple[str, str]) -> dict:
    """Compute one rate-1 ``ChartMeta``. ``job`` is ``(raw root, relpath)``."""
    root, relpath = job
    osu_path = Path(root) / relpath
    try:
        meta = compute_chart_meta(osu_path)
    except Exception as exc:
        meta = ChartMeta.failed(osu_path, f"{type(exc).__name__}: {exc}")
    record = meta.to_dict()
    record["relpath"] = relpath
    return record


def ensure_v1_chart_meta(
    relpaths: list[str],
    root: Path,
    *,
    workers: int,
    path: Path | None = None,
) -> None:
    """Fill any missing rate-1 rows. Call this before CUDA is initialized."""
    manifest = path or v1_chart_meta_path()
    stored = load_v1_chart_meta(manifest)
    missing = list(dict.fromkeys(rel for rel in relpaths if rel not in stored))
    if not missing:
        logger.info("v1 chart meta hit %s", len(relpaths))
        return
    logger.info("v1 chart meta missing %s/%s", len(missing), len(relpaths))
    jobs = [(str(root), relpath) for relpath in missing]
    context = mp.get_context("spawn")
    batch: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=context) as pool:
        mapped = pool.map(measure_v1_chart_meta, jobs, chunksize=8)
        for index, record in enumerate(mapped, start=1):
            batch.append(record)
            if len(batch) >= 200 or index == len(jobs):
                append_v1_chart_meta_records(batch, manifest)
                batch.clear()
                logger.info("v1 chart meta %s/%s", index, len(jobs))
    clear_v1_chart_meta_cache()


def clear_v1_chart_meta_cache() -> None:
    """Drop the worker-local table. Tests use this between manifests."""
    global _CACHE
    _CACHE = None


def rate1_meta_for(root: Path, relpath: str, path: Path | None = None) -> ChartMeta:
    """Rate-1 metadata for one chart. Reads the disk table, then computes."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load_v1_chart_meta(path)
    cached = _CACHE.get(relpath)
    if cached is not None:
        return cached
    osu_path = root / relpath
    meta = compute_chart_meta(osu_path)
    record = meta.to_dict()
    record["relpath"] = relpath
    append_v1_chart_meta_records([record], path)
    _CACHE[relpath] = meta
    if meta.error:
        logger.warning("chart meta %s: %s", osu_path.name, meta.error)
    return meta
