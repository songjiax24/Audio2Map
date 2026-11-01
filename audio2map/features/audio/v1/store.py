"""Rate-1 PCM and mel for training.

PCM is one float32 buffer per audio file. Mel is the normalized span of one
coverage window, so charts that share a file and a window share that mel.
A clock rate other than 1 stretches the PCM and computes a new mel.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio2map.features.audio.v1.constants import N_MELS
from audio2map.utils.paths import v1_audio_dir

logger = logging.getLogger(__name__)

AUDIO_CACHE_VERSION = 1
_KIND = "v1-rate1-audio"
_AUDIO_ABS_TOL_MS = 1e-3


@dataclass(frozen=True, slots=True)
class MelPiece:
    frame_lo: int
    frame_hi: int
    file: str


@dataclass(frozen=True, slots=True)
class AudioRecord:
    audio: str
    t_audio_ms: float
    source_bytes: int
    n_pcm: int
    pcm_file: str
    mels: tuple[MelPiece, ...]
    error: str | None = None


class AudioCache:
    """Manifest of per-file PCM and per-window mel."""

    def __init__(self, directory: Path, records: dict[str, AudioRecord]) -> None:
        self.directory = directory
        self.records = records

    def pcm(self, audio: str, source_bytes: int) -> tuple[np.ndarray, float] | None:
        record = self.records.get(audio)
        if record is None or record.error is not None or record.source_bytes != source_bytes:
            return None
        loaded = _load_f32(self.directory / record.pcm_file, record.n_pcm)
        if loaded is None:
            return None
        return loaded, record.t_audio_ms

    def mel(
        self,
        audio: str,
        t_audio_ms: float,
        frame_lo: int,
        frame_hi: int,
    ) -> np.ndarray | None:
        record = self.records.get(audio)
        if record is None or record.error is not None:
            return None
        if not math.isclose(record.t_audio_ms, t_audio_ms, rel_tol=0.0, abs_tol=_AUDIO_ABS_TOL_MS):
            return None
        piece = None
        for item in record.mels:
            if item.frame_lo == frame_lo and item.frame_hi == frame_hi:
                piece = item
                break
        if piece is None or frame_hi <= frame_lo:
            return None
        n_cols = 2 * (frame_hi - frame_lo) + 3
        loaded = _load_f32(self.directory / piece.file, N_MELS * n_cols)
        if loaded is None:
            return None
        return np.ascontiguousarray(loaded.reshape(N_MELS, n_cols))


_LOADED: AudioCache | None = None


def cached_rate1_pcm(audio: str, source_bytes: int) -> tuple[np.ndarray, float] | None:
    """PCM for this source size, or None when the file is absent or stale."""
    return _cache().pcm(audio, source_bytes)


def cached_rate1_mel(
    audio: str,
    t_audio_ms: float,
    frame_lo: int,
    frame_hi: int,
) -> np.ndarray | None:
    """Normalized mel for this coverage window, or None."""
    return _cache().mel(audio, t_audio_ms, frame_lo, frame_hi)


def _cache() -> AudioCache:
    global _LOADED
    if _LOADED is None:
        _LOADED = load_audio_cache(v1_audio_dir())
    return _LOADED


def load_audio_cache(directory: Path) -> AudioCache:
    """Read the manifest. A version mismatch or a missing file is an empty cache."""
    path = directory / "manifest.jsonl"
    if not path.is_file():
        return AudioCache(directory, {})
    records: dict[str, AudioRecord] = {}
    with path.open(encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line:
            return AudioCache(directory, {})
        header = json.loads(header_line)
        if header.get("kind") != _KIND or header.get("version") != AUDIO_CACHE_VERSION:
            return AudioCache(directory, {})
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            audio = str(payload["audio"])
            records[audio] = _record_from_payload(payload)
    return AudioCache(directory, records)


def ensure_rate1_audio(
    charts: list,
    root: Path,
    lengths: list,
    *,
    workers: int,
    directory: Path | None = None,
) -> AudioCache:
    """Decode any train audio whose rate-1 PCM or mel windows are missing."""
    global _LOADED
    directory = directory or v1_audio_dir()
    directory.mkdir(parents=True, exist_ok=True)
    _ensure_header(directory)
    cached = load_audio_cache(directory)
    groups: dict[str, list[str]] = {}
    for chart, row in zip(charts, lengths, strict=True):
        if row.error is not None:
            continue
        key = _audio_key(root, chart.relpath)
        if key is None:
            logger.warning("audio cache skip %s: no audio file", chart.relpath)
            continue
        groups.setdefault(key, []).append(chart.relpath)
    pending = [
        key for key in groups if not _complete(cached, directory, root / key, key)
    ]
    if not pending:
        logger.info("audio cache hit %s (%s)", len(groups), directory)
        _LOADED = cached
        return cached
    logger.info("audio cache missing %s/%s; decoding rate-1 audio", len(pending), len(groups))
    _encode_missing(root, groups, pending, directory, workers=workers)
    written = load_audio_cache(directory)
    _LOADED = written
    logger.info("wrote %s", directory)
    return written


def _audio_key(root: Path, relpath: str) -> str | None:
    from audio2map.osu.parser import chart_audio_path

    try:
        audio = chart_audio_path(root / relpath)
        return str(audio.resolve().relative_to(root.resolve()))
    except (OSError, ValueError, FileNotFoundError):
        return None


def _complete(cache: AudioCache, directory: Path, source: Path, key: str) -> bool:
    record = cache.records.get(key)
    if record is None or record.error is not None or record.n_pcm <= 0:
        return False
    try:
        if source.stat().st_size != record.source_bytes:
            return False
    except OSError:
        return False
    pcm_path = directory / record.pcm_file
    if not pcm_path.is_file() or pcm_path.stat().st_size != record.n_pcm * 4:
        return False
    for piece in record.mels:
        mel_path = directory / piece.file
        n_cols = 2 * (piece.frame_hi - piece.frame_lo) + 3
        if not mel_path.is_file() or mel_path.stat().st_size != N_MELS * n_cols * 4:
            return False
    return True


def _encode_missing(
    root: Path,
    groups: dict[str, list[str]],
    pending: list[str],
    directory: Path,
    *,
    workers: int,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    jobs = [(str(root), key, groups[key], str(directory)) for key in pending]
    context = mp.get_context("spawn")
    mel_count = 0
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=context) as pool:
        futures = [pool.submit(_build_one, job) for job in jobs]
        total = len(futures)
        for index, future in enumerate(as_completed(futures), start=1):
            payload = future.result()
            _append_record(directory, payload)
            mel_count += len(payload.get("mels") or [])
            if index % 100 == 0 or index == total:
                logger.info("audio %s/%s mels=%s", index, total, mel_count)


def _build_one(job: tuple[str, str, list[str], str]) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    root_s, audio_key, relpaths, directory_s = job
    try:
        return _build_audio(Path(root_s), audio_key, relpaths, Path(directory_s))
    except Exception as exc:
        return {"audio": audio_key, "error": f"{type(exc).__name__}: {exc}"[:300]}


def _build_audio(root: Path, audio_key: str, relpaths: list[str], directory: Path) -> dict:
    from audio2map.features.audio.v1.decode import decode_audio
    from audio2map.features.audio.v1.mel import frame_index, mel_span
    from audio2map.grid.v1.beatgrid import build_beatgrid
    from audio2map.osu.v1.timing import parse_redlines_file

    source = root / audio_key
    decoded = decode_audio(source)
    windows: dict[tuple[int, int], np.ndarray] = {}
    skipped: list[str] = []
    for relpath in relpaths:
        try:
            redlines = parse_redlines_file(root / relpath).redlines
            grid = build_beatgrid(redlines, decoded.t_audio_ms)
            frame_lo = frame_index(grid.coverage_start_ms)
            frame_hi = frame_index(grid.coverage_end_ms)
        except Exception as exc:
            skipped.append(f"{relpath}: {type(exc).__name__}"[:180])
            continue
        key = (frame_lo, frame_hi)
        if key in windows or frame_hi <= frame_lo:
            continue
        span = mel_span(decoded.y, grid.coverage_start_ms, grid.coverage_end_ms)
        windows[key] = np.ascontiguousarray(span.mel, dtype=np.float32)
    pcm_file = _pcm_rel(audio_key)
    _dump(directory / pcm_file, decoded.y)
    mels: list[dict[str, int | str]] = []
    for (frame_lo, frame_hi), mel in sorted(windows.items()):
        file = _mel_rel(audio_key, frame_lo, frame_hi)
        _dump(directory / file, mel)
        mels.append({"frame_lo": frame_lo, "frame_hi": frame_hi, "file": file})
    payload: dict[str, object] = {
        "audio": audio_key,
        "t_audio_ms": decoded.t_audio_ms,
        "source_bytes": source.stat().st_size,
        "n_pcm": int(decoded.y.shape[0]),
        "pcm": pcm_file,
        "mels": mels,
    }
    if skipped:
        payload["skipped"] = skipped[:8]
    return payload


def _pcm_rel(audio_key: str) -> str:
    return str(Path("pcm") / f"{audio_key}.f32")


def _mel_rel(audio_key: str, frame_lo: int, frame_hi: int) -> str:
    return str(Path("mel") / audio_key / f"{frame_lo}_{frame_hi}.f32")


def _dump(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    values = np.ascontiguousarray(array, dtype=np.float32)
    with temporary.open("wb") as handle:
        values.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_f32(path: Path, n: int) -> np.ndarray | None:
    if n < 0 or not path.is_file():
        return None
    if path.stat().st_size != n * 4:
        return None
    loaded = np.fromfile(path, dtype="<f4", count=n)
    if int(loaded.shape[0]) != n:
        return None
    return loaded


def _ensure_header(directory: Path) -> None:
    path = directory / "manifest.jsonl"
    if path.is_file() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": _KIND, "version": AUDIO_CACHE_VERSION}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_record(directory: Path, payload: dict) -> None:
    path = directory / "manifest.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _record_from_payload(payload: dict) -> AudioRecord:
    error = payload.get("error")
    mels = tuple(
        MelPiece(int(item["frame_lo"]), int(item["frame_hi"]), str(item["file"]))
        for item in payload.get("mels") or []
    )
    return AudioRecord(
        audio=str(payload["audio"]),
        t_audio_ms=float(payload.get("t_audio_ms", 0.0)),
        source_bytes=int(payload.get("source_bytes", 0)),
        n_pcm=int(payload.get("n_pcm", 0)),
        pcm_file=str(payload.get("pcm", "")),
        mels=mels,
        error=None if error is None else str(error),
    )
