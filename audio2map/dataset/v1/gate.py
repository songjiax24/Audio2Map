"""Drop charts before a v1 dataset is built.

The split file stays unchanged. Variable BPM and meter are not reasons.
Training and inference share :func:`timeline_spliced` for damaged audio.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from audio2map.dataset.split import ChartSnapshot
from audio2map.features.audio.v1.decode import (
    decode_audio,
    ffmpeg_decode_log,
    timeline_spliced,
)
from audio2map.osu.parser import (
    InvalidHitObjectError,
    InvalidTimingPointError,
    chart_audio_path,
    is_mania_4k,
    parse_beatmap,
)
from audio2map.osu.v1.timing import parse_redlines_file
from audio2map.tokens.v1.codec import encode_chart
from audio2map.utils.errors import ChartRejected
from audio2map.utils.paths import v1_gate_dir

logger = logging.getLogger(__name__)

GATE_VERSION = 1
_CHART_KIND = "v1-chart-gate"
_TIMELINE_KIND = "v1-timeline-gate"

NOT_4K = "not_4k"
INVALID_HIT_OBJECTS = "invalid_hit_objects"
INVALID_TIMING_POINTS = "invalid_timing_points"
NO_NOTES = "no_notes"
MISSING_AUDIO = "missing_audio"
SPLICED_TIMELINE = "spliced_timeline"
UNREADABLE = "unreadable"
DECODE_FAILED = "decode_failed"
# ffmpeg log scans do not use the mel pool. 16 of those were slower.
_TIMELINE_WORKERS = 8


@dataclass(frozen=True, slots=True)
class _ChartHit:
    osu_bytes: int
    audio_bytes: int
    reason: str | None


@dataclass(frozen=True, slots=True)
class _TimelineHit:
    source_bytes: int
    spliced: bool
    t_audio_ms: float | None


def osu_reject_reason(path: Path) -> str | None:
    """Parse failures, empty charts, and red-line rejects. No audio read."""
    try:
        if not is_mania_4k(path):
            return NOT_4K
        beatmap = parse_beatmap(path)
    except InvalidHitObjectError:
        return INVALID_HIT_OBJECTS
    except InvalidTimingPointError:
        return INVALID_TIMING_POINTS
    except ValueError:
        return NOT_4K
    except OSError:
        return UNREADABLE
    if len(beatmap.notes) == 0:
        return NO_NOTES
    try:
        parse_redlines_file(path)
    except ChartRejected as exc:
        return exc.reason
    return None


def structure_reject_reason(path: Path, t_audio_ms: float) -> str | None:
    """``encode_chart`` rejects. ``None`` means the chart can be trained."""
    try:
        beatmap = parse_beatmap(path)
        redlines = parse_redlines_file(path).redlines
        encode_chart(list(beatmap.notes), redlines, t_audio_ms)
    except ChartRejected as exc:
        return exc.reason
    return None


def chart_reject_reason(
    path: Path,
    *,
    t_audio_ms: float | None,
    ffmpeg_log: str | None,
    structure_known_ok: bool,
) -> str | None:
    """One chart. A supplied log is not decoded again."""
    reason = osu_reject_reason(path)
    if reason is not None:
        return reason
    if ffmpeg_log is not None and timeline_spliced(ffmpeg_log):
        return SPLICED_TIMELINE
    if structure_known_ok:
        return None
    if t_audio_ms is None:
        raise ValueError(f"t_audio_ms is required to encode {path}")
    return structure_reject_reason(path, t_audio_ms)


def filter_v1_charts(
    charts: list[ChartSnapshot],
    root: Path,
    *,
    lengths: list | None = None,
    workers: int = 16,
    gate_dir: Path | None = None,
) -> tuple[list[ChartSnapshot], list | None]:
    """Keep charts that pass §1.2 and §1.4. ``lengths`` stays aligned when given."""
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    if lengths is not None and len(lengths) != len(charts):
        raise ValueError(f"lengths ({len(lengths)}) must match charts ({len(charts)})")
    directory = gate_dir or v1_gate_dir()
    reasons = _decide(charts, root, lengths, workers=workers, gate_dir=directory)
    kept_charts: list[ChartSnapshot] = []
    kept_rows: list | None = [] if lengths is not None else None
    counts: Counter[str] = Counter()
    for index, chart in enumerate(charts):
        reason = reasons[chart.relpath]
        if reason is None:
            kept_charts.append(chart)
            if kept_rows is not None and lengths is not None:
                kept_rows.append(lengths[index])
            continue
        counts[reason] += 1
        logger.warning("v1 gate drop %s: %s", chart.relpath, reason)
    summary = " ".join(f"{name}={counts[name]}" for name in sorted(counts))
    logger.info(
        "v1 gate kept %s dropped %s%s",
        len(kept_charts),
        sum(counts.values()),
        f" {summary}" if summary else "",
    )
    return kept_charts, kept_rows


def _decide(
    charts: list[ChartSnapshot],
    root: Path,
    lengths: list | None,
    *,
    workers: int,
    gate_dir: Path,
) -> dict[str, str | None]:
    rows = {}
    if lengths is not None:
        rows = {chart.relpath: row for chart, row in zip(charts, lengths, strict=True)}
    cached = _load_charts(gate_dir)
    result: dict[str, str | None] = {}
    fresh: list[ChartSnapshot] = []
    sizes: dict[str, tuple[int, int]] = {}
    for chart in charts:
        osu_bytes, audio_bytes = _sizes(root, chart.relpath)
        sizes[chart.relpath] = (osu_bytes, audio_bytes)
        hit = cached.get(chart.relpath)
        row = rows.get(chart.relpath)
        stale_keep = hit is not None and hit.reason is None and row is not None and row.error
        if (
            hit is not None
            and hit.osu_bytes == osu_bytes
            and hit.audio_bytes == audio_bytes
            and not stale_keep
        ):
            result[chart.relpath] = hit.reason
            continue
        fresh.append(chart)
    if fresh:
        decided = _decide_fresh(fresh, root, rows, workers=workers, gate_dir=gate_dir)
        result.update(decided)
        _store_charts(gate_dir, cached, sizes, decided)
    return result


def _decide_fresh(
    charts: list[ChartSnapshot],
    root: Path,
    rows: dict,
    *,
    workers: int,
    gate_dir: Path,
) -> dict[str, str | None]:
    osu_reasons = _map_jobs(
        [(str(root), chart.relpath) for chart in charts],
        _osu_job,
        workers=workers,
        label="v1 gate parse",
    )
    result: dict[str, str | None] = {}
    pending: list[ChartSnapshot] = []
    for chart in charts:
        reason = osu_reasons[chart.relpath]
        row = rows.get(chart.relpath)
        if reason is None and row is not None and row.error:
            reason = _reason_from_length_error(str(row.error))
        if reason is not None:
            result[chart.relpath] = reason
            continue
        pending.append(chart)
    groups: dict[str, list[str]] = {}
    known_t: dict[str, float] = {}
    for chart in pending:
        key = _audio_key(root, chart.relpath)
        if key is None:
            result[chart.relpath] = MISSING_AUDIO
            continue
        groups.setdefault(key, []).append(chart.relpath)
        row = rows.get(chart.relpath)
        if row is not None and row.error is None and float(row.t_audio_ms) > 0.0:
            known_t[key] = float(row.t_audio_ms)
    timeline = _timelines(root, groups, known_t, workers=workers, gate_dir=gate_dir)
    encode_jobs: list[tuple[str, str, float]] = []
    for key, relpaths in groups.items():
        hit = timeline.get(key)
        if hit is None:
            for relpath in relpaths:
                result[relpath] = DECODE_FAILED
            continue
        for relpath in relpaths:
            if hit.spliced:
                result[relpath] = SPLICED_TIMELINE
                continue
            row = rows.get(relpath)
            known = (
                row is not None
                and row.error is None
                and int(row.n_frames) > 0
                and int(row.n_tokens) > 0
            )
            if known:
                result[relpath] = None
                continue
            duration = hit.t_audio_ms
            if duration is None and row is not None and row.error is None:
                duration = float(row.t_audio_ms)
            if duration is None:
                result[relpath] = DECODE_FAILED
                continue
            encode_jobs.append((str(root), relpath, float(duration)))
    if encode_jobs:
        encoded = _map_jobs(encode_jobs, _encode_job, workers=workers, label="v1 gate encode")
        result.update(encoded)
    return result


def _timelines(
    root: Path,
    groups: dict[str, list[str]],
    known_t: dict[str, float],
    *,
    workers: int,
    gate_dir: Path,
) -> dict[str, _TimelineHit | None]:
    """``None`` in the returned map means the read failed and was not stored."""
    stored = _load_timeline(gate_dir)
    found: dict[str, _TimelineHit | None] = {}
    jobs: list[tuple[str, str, bool]] = []
    for key in groups:
        path = root / key
        try:
            source_bytes = path.stat().st_size
        except OSError:
            found[key] = None
            continue
        need_duration = key not in known_t
        hit = stored.get(key)
        duration_ok = (not need_duration) or (hit is not None and hit.t_audio_ms is not None)
        if hit is not None and hit.source_bytes == source_bytes and duration_ok:
            found[key] = hit
            continue
        jobs.append((str(root), key, need_duration))
    if not jobs:
        return found
    width = min(workers, _TIMELINE_WORKERS)
    logger.info("v1 gate timeline missing %s/%s", len(jobs), len(groups))
    updates: dict[str, _TimelineHit] = {}
    for index, (key, payload) in enumerate(
        _iter_jobs(jobs, _timeline_job, workers=width, label="v1 gate timeline"),
        start=1,
    ):
        source_bytes, spliced, t_audio, error = payload
        if error or source_bytes <= 0:
            found[key] = None
        else:
            record = _TimelineHit(source_bytes, spliced, t_audio)
            found[key] = record
            updates[key] = record
        if updates and (index % 100 == 0 or index == len(jobs)):
            _store_timeline(gate_dir, stored, updates)
            stored.update(updates)
    return found


def _map_jobs(jobs: list, function, *, workers: int, label: str) -> dict:
    found = {}
    for key, payload in _iter_jobs(jobs, function, workers=workers, label=label):
        found[key] = payload
    return found


def _iter_jobs(jobs: list, function, *, workers: int, label: str):
    if not jobs:
        return
    if workers == 1:
        total = len(jobs)
        for index, job in enumerate(jobs, start=1):
            payload = function(job)
            yield payload[0], payload[1]
            if index % 100 == 0 or index == total:
                logger.info("%s %s/%s", label, index, total)
        return
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(function, job) for job in jobs]
        total = len(futures)
        for index, future in enumerate(as_completed(futures), start=1):
            payload = future.result()
            yield payload[0], payload[1]
            if index % 100 == 0 or index == total:
                logger.info("%s %s/%s", label, index, total)


def _osu_job(job: tuple[str, str]) -> tuple[str, str | None]:
    os.environ["OMP_NUM_THREADS"] = "1"
    root, relpath = job
    return relpath, osu_reject_reason(Path(root) / relpath)


def _encode_job(job: tuple[str, str, float]) -> tuple[str, str | None]:
    os.environ["OMP_NUM_THREADS"] = "1"
    root, relpath, t_audio_ms = job
    return relpath, structure_reject_reason(Path(root) / relpath, t_audio_ms)


def _timeline_job(
    job: tuple[str, str, bool],
) -> tuple[str, tuple[int, bool, float | None, str | None]]:
    os.environ["OMP_NUM_THREADS"] = "1"
    root, key, need_duration = job
    path = Path(root) / key
    try:
        source_bytes = path.stat().st_size
        if need_duration:
            decoded = decode_audio(path)
            return key, (source_bytes, decoded.timeline_spliced, decoded.t_audio_ms, None)
        spliced = timeline_spliced(ffmpeg_decode_log(path))
        return key, (source_bytes, spliced, None, None)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"[:300]
        return key, (0, False, None, detail)


def _reason_from_length_error(error: str) -> str:
    if error.startswith("ChartRejected:"):
        parts = [part.strip() for part in error.split(":")]
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    if error.startswith("InvalidHitObjectError"):
        return INVALID_HIT_OBJECTS
    if error.startswith("InvalidTimingPointError"):
        return INVALID_TIMING_POINTS
    if error.startswith("FileNotFoundError"):
        return MISSING_AUDIO
    if "empty shape" in error:
        return "empty_shape"
    return "rejected"


def _sizes(root: Path, relpath: str) -> tuple[int, int]:
    path = root / relpath
    try:
        osu_bytes = path.stat().st_size
    except OSError:
        osu_bytes = -1
    return osu_bytes, _audio_size(root, relpath)


def _audio_size(root: Path, relpath: str) -> int:
    try:
        audio = chart_audio_path(root / relpath)
        return audio.stat().st_size
    except (OSError, ValueError, FileNotFoundError):
        return -1


def _audio_key(root: Path, relpath: str) -> str | None:
    try:
        audio = chart_audio_path(root / relpath)
        return str(audio.resolve().relative_to(root.resolve()))
    except (OSError, ValueError, FileNotFoundError):
        return None


def _load_charts(directory: Path) -> dict[str, _ChartHit]:
    path = directory / "charts.jsonl"
    rows: dict[str, _ChartHit] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        header = json.loads(handle.readline() or "{}")
        if header.get("kind") != _CHART_KIND or header.get("version") != GATE_VERSION:
            logger.warning("ignoring chart gate cache %s", path)
            return {}
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            relpath = str(payload["relpath"])
            reason = payload.get("reason")
            rows[relpath] = _ChartHit(
                int(payload["osu_bytes"]),
                int(payload["audio_bytes"]),
                None if reason in (None, "") else str(reason),
            )
    return rows


def _store_charts(
    directory: Path,
    previous: dict[str, _ChartHit],
    sizes: dict[str, tuple[int, int]],
    decided: dict[str, str | None],
) -> None:
    merged = dict(previous)
    for relpath, reason in decided.items():
        osu_bytes, audio_bytes = sizes[relpath]
        merged[relpath] = _ChartHit(osu_bytes, audio_bytes, reason)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "charts.jsonl"
    temporary = path.with_name("charts.jsonl.part")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": _CHART_KIND, "version": GATE_VERSION}) + "\n")
        for relpath in sorted(merged):
            hit = merged[relpath]
            payload = {
                "relpath": relpath,
                "osu_bytes": hit.osu_bytes,
                "audio_bytes": hit.audio_bytes,
                "reason": hit.reason,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load_timeline(directory: Path) -> dict[str, _TimelineHit]:
    path = directory / "timeline.jsonl"
    rows: dict[str, _TimelineHit] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        header = json.loads(handle.readline() or "{}")
        if header.get("kind") != _TIMELINE_KIND or header.get("version") != GATE_VERSION:
            logger.warning("ignoring timeline gate cache %s", path)
            return {}
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            t_audio = payload.get("t_audio_ms")
            rows[str(payload["audio"])] = _TimelineHit(
                int(payload["source_bytes"]),
                bool(payload["spliced"]),
                None if t_audio is None else float(t_audio),
            )
    return rows


def _store_timeline(
    directory: Path,
    previous: dict[str, _TimelineHit],
    updates: dict[str, _TimelineHit],
) -> None:
    merged = dict(previous)
    merged.update(updates)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "timeline.jsonl"
    temporary = path.with_name("timeline.jsonl.part")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": _TIMELINE_KIND, "version": GATE_VERSION}) + "\n")
        for key in sorted(merged):
            hit = merged[key]
            payload = {
                "audio": key,
                "source_bytes": hit.source_bytes,
                "spliced": hit.spliced,
                "t_audio_ms": hit.t_audio_ms,
            }
            handle.write(json.dumps(payload) + "\n")
    temporary.replace(path)
