"""Rate-1 frame and token counts, cached so startup does not rescan.

Packing uses these frame counts. Rate is drawn after the batch is chosen,
so the forward is this long or shorter. Object-end time is not stored here.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio2map.dataset.split import ChartSnapshot
from audio2map.dataset.v1.shape import chart_shape
from audio2map.features.audio.v1.decode import decode_audio
from audio2map.osu.parser import chart_audio_path, parse_beatmap
from audio2map.osu.v1.timing import parse_redlines_file
from audio2map.utils.paths import v1_length_dir

logger = logging.getLogger(__name__)

LENGTH_VERSION = 1
_KIND = "v1-rate1-shape"


@dataclass(frozen=True, slots=True)
class ChartLength:
    relpath: str
    n_frames: int
    n_tokens: int
    t_audio_ms: float
    error: str | None = None


def cache_path_for(split_path: Path) -> Path:
    return v1_length_dir() / f"{Path(split_path).stem}.jsonl"


def load_rate1_lengths(
    charts: list[ChartSnapshot],
    root: Path,
    cache_path: Path,
    *,
    workers: int = 32,
) -> list[ChartLength]:
    """Return one row per chart. Missing rows are decoded once and appended."""
    stored = _read_cache(cache_path)
    missing = [chart.relpath for chart in charts if chart.relpath not in stored]
    if missing:
        logger.info(
            "length cache missing %s/%s; measuring rate-1 shapes",
            len(missing),
            len(charts),
        )
        _measure_into(stored, root, missing, cache_path, workers=workers)
        logger.info("wrote %s", cache_path)
    else:
        logger.info("length cache hit %s (%s)", len(charts), cache_path)
    rows = []
    for chart in charts:
        row = stored.get(chart.relpath)
        if row is None:
            rows.append(ChartLength(chart.relpath, 0, 0, 0.0, error="missing"))
        else:
            rows.append(row)
    failed = sum(row.error is not None for row in rows)
    if failed:
        logger.warning("rate-1 shape failed for %s charts", failed)
    return rows


def length_percentile(rows: list[ChartLength], percentile: float) -> tuple[float, float]:
    """``(frame p, token p)`` over charts that measured successfully."""
    frames = [row.n_frames for row in rows if row.error is None and row.n_frames > 0]
    tokens = [row.n_tokens for row in rows if row.error is None and row.n_tokens > 0]
    if not frames or not tokens:
        raise RuntimeError("no rate-1 frame/token lengths to bucket on")
    frame_value = float(np.percentile(np.asarray(frames, dtype=np.float64), percentile))
    token_value = float(np.percentile(np.asarray(tokens, dtype=np.float64), percentile))
    return frame_value, token_value


def length_summary(rows: list[ChartLength]) -> dict[str, float | int]:
    frames = np.asarray(
        [row.n_frames for row in rows if row.error is None and row.n_frames > 0],
        dtype=np.float64,
    )
    tokens = np.asarray(
        [row.n_tokens for row in rows if row.error is None and row.n_tokens > 0],
        dtype=np.float64,
    )
    if frames.size == 0 or tokens.size == 0:
        raise RuntimeError("no rate-1 frame/token lengths to bucket on")
    return {
        "n": int(frames.size),
        "failed": sum(row.error is not None for row in rows),
        "frame_p50": float(np.percentile(frames, 50)),
        "frame_p90": float(np.percentile(frames, 90)),
        "frame_max": float(frames.max()),
        "token_p50": float(np.percentile(tokens, 50)),
        "token_p90": float(np.percentile(tokens, 90)),
        "token_max": float(tokens.max()),
    }


def measure_chart(job: tuple[str, str]) -> dict:
    """Decode one chart at rate 1. ``job`` is ``(raw root, relpath)``."""
    root, relpath = job
    path = Path(root) / relpath
    try:
        decoded = decode_audio(chart_audio_path(path))
        beatmap = parse_beatmap(path)
        redlines = parse_redlines_file(path).redlines
        n_frames, n_tokens = chart_shape(list(beatmap.notes), redlines, decoded.t_audio_ms)
        if n_frames <= 0 or n_tokens <= 0:
            raise RuntimeError(f"empty shape frames={n_frames} tokens={n_tokens}")
        return {
            "relpath": relpath,
            "n_frames": n_frames,
            "n_tokens": n_tokens,
            "t_audio_ms": decoded.t_audio_ms,
        }
    except Exception as exc:
        return {"relpath": relpath, "error": f"{type(exc).__name__}: {exc}"[:300]}


def _measure_into(
    stored: dict[str, ChartLength],
    root: Path,
    relpaths: list[str],
    cache_path: Path,
    *,
    workers: int,
) -> None:
    jobs = [(str(root), relpath) for relpath in relpaths]
    # Spawn so workers do not inherit a torch process. They only decode and encode.
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=context) as pool:
        for index, payload in enumerate(pool.map(measure_chart, jobs, chunksize=8), start=1):
            stored[payload["relpath"]] = _row_from_payload(payload)
            if index % 1000 == 0 or index == len(jobs):
                _write_cache(cache_path, stored)
                logger.info("length measure %s/%s", index, len(jobs))


def _row_from_payload(payload: dict) -> ChartLength:
    error = payload.get("error")
    if error:
        return ChartLength(payload["relpath"], 0, 0, 0.0, error=str(error))
    return ChartLength(
        payload["relpath"],
        int(payload["n_frames"]),
        int(payload["n_tokens"]),
        float(payload["t_audio_ms"]),
    )


def _read_cache(path: Path) -> dict[str, ChartLength]:
    if not path.is_file():
        return {}
    rows: dict[str, ChartLength] = {}
    with path.open(encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            return {}
        header = json.loads(first)
        if header.get("kind") != _KIND or header.get("version") != LENGTH_VERSION:
            logger.warning("ignoring length cache %s (version mismatch)", path)
            return {}
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows[payload["relpath"]] = _row_from_payload(payload)
    return rows


def _write_cache(path: Path, rows: dict[str, ChartLength]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": _KIND, "version": LENGTH_VERSION}) + "\n")
        for relpath in sorted(rows):
            row = rows[relpath]
            payload: dict[str, str | int | float] = {"relpath": row.relpath}
            if row.error:
                payload["error"] = row.error
            else:
                payload["n_frames"] = row.n_frames
                payload["n_tokens"] = row.n_tokens
                payload["t_audio_ms"] = row.t_audio_ms
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    temporary.replace(path)
