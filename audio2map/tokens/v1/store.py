"""Rate-1 token strings for training.

The string does not depend on the clock rate. Mirror is a lane swap on each
``<ROW_>`` and is applied by the caller. One int16 id sequence is stored per
chart, next to the ``t_audio_ms`` it was labeled with.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio2map.dataset.split import ChartSnapshot
from audio2map.tokens.v1.vocab import TOKENIZER_VERSION, build_vocab, invert_vocab
from audio2map.train.v1.lengths import ChartLength
from audio2map.utils.paths import v1_token_dir

logger = logging.getLogger(__name__)

TOKEN_CACHE_VERSION = 1
_KIND = "v1-rate1-tokens"
# One sample at 22050 Hz is about 0.045 ms. JSON and a second decode stay inside this.
_AUDIO_ABS_TOL_MS = 1e-3


@dataclass(frozen=True, slots=True)
class TokenRecord:
    relpath: str
    t_audio_ms: float
    offset: int
    n: int
    error: str | None = None


class TokenCache:
    """Manifest plus the packed id file."""

    def __init__(self, directory: Path, records: dict[str, TokenRecord], ids: np.ndarray) -> None:
        self.directory = directory
        self.records = records
        self.ids = ids

    def tokens(self, relpath: str, t_audio_ms: float) -> list[str] | None:
        record = self.records.get(relpath)
        if record is None or record.error is not None or record.n <= 0:
            return None
        if not math.isclose(record.t_audio_ms, t_audio_ms, rel_tol=0.0, abs_tol=_AUDIO_ABS_TOL_MS):
            return None
        invert = invert_vocab()
        piece = self.ids[record.offset : record.offset + record.n]
        return [invert[int(token_id)] for token_id in piece]


_LOADED: TokenCache | None = None


def cached_rate1_tokens(relpath: str, t_audio_ms: float) -> list[str] | None:
    """Tokens labeled at this duration, or ``None`` when the cache has no row."""
    global _LOADED
    if _LOADED is None:
        _LOADED = load_token_cache(v1_token_dir())
    return _LOADED.tokens(relpath, t_audio_ms)


def load_token_cache(directory: Path) -> TokenCache:
    manifest = directory / "manifest.jsonl"
    blob = directory / "ids.i16"
    if not manifest.is_file() or not blob.is_file():
        return TokenCache(directory, {}, np.zeros(0, dtype=np.int16))
    records: dict[str, TokenRecord] = {}
    with manifest.open(encoding="utf-8") as handle:
        header = json.loads(handle.readline() or "{}")
        if header.get("kind") != _KIND or header.get("version") != TOKEN_CACHE_VERSION:
            logger.warning("ignoring token cache %s (version mismatch)", manifest)
            return TokenCache(directory, {}, np.zeros(0, dtype=np.int16))
        if header.get("tokenizer") != TOKENIZER_VERSION:
            logger.warning("ignoring token cache %s (tokenizer mismatch)", manifest)
            return TokenCache(directory, {}, np.zeros(0, dtype=np.int16))
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            records[str(payload["relpath"])] = TokenRecord(
                relpath=str(payload["relpath"]),
                t_audio_ms=float(payload.get("t_audio_ms", 0.0)),
                offset=int(payload.get("offset", 0)),
                n=int(payload.get("n", 0)),
                error=payload.get("error"),
            )
    ids = np.fromfile(blob, dtype="<i2")
    return TokenCache(directory, records, ids)


def ensure_rate1_tokens(
    charts: list[ChartSnapshot],
    root: Path,
    lengths: list[ChartLength],
    *,
    workers: int,
    directory: Path | None = None,
) -> TokenCache:
    """Encode any train chart whose rate-1 tokens are missing. Returns the cache."""
    global _LOADED
    directory = directory or v1_token_dir()
    cached = load_token_cache(directory)
    pending: list[tuple[str, float]] = []
    for chart, row in zip(charts, lengths, strict=True):
        if row.error is not None:
            continue
        have = cached.records.get(chart.relpath)
        if (
            have is not None
            and have.error is None
            and have.n == row.n_tokens
            and math.isclose(
                have.t_audio_ms,
                row.t_audio_ms,
                rel_tol=0.0,
                abs_tol=_AUDIO_ABS_TOL_MS,
            )
        ):
            continue
        pending.append((chart.relpath, row.t_audio_ms))
    if not pending:
        logger.info("token cache hit %s (%s)", len(charts), directory)
        _LOADED = cached
        return cached
    logger.info("token cache missing %s/%s; encoding rate-1 tokens", len(pending), len(charts))
    fresh = _encode_missing(root, pending, workers=workers)
    merged = dict(cached.records)
    arrays: dict[str, np.ndarray] = {}
    for relpath, record in cached.records.items():
        if record.error is None and record.n > 0:
            arrays[relpath] = cached.ids[record.offset : record.offset + record.n].copy()
    for relpath, t_audio_ms, payload in fresh:
        if isinstance(payload, str):
            merged[relpath] = TokenRecord(relpath, t_audio_ms, 0, 0, error=payload)
            arrays.pop(relpath, None)
        else:
            arrays[relpath] = payload
            merged[relpath] = TokenRecord(relpath, t_audio_ms, 0, int(payload.shape[0]))
    written = _write_token_cache(directory, merged, arrays)
    _LOADED = written
    logger.info("wrote %s", directory)
    return written


def _encode_missing(
    root: Path,
    pending: list[tuple[str, float]],
    *,
    workers: int,
) -> list[tuple[str, float, np.ndarray | str]]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    jobs = [(str(root), relpath, t_audio_ms) for relpath, t_audio_ms in pending]
    context = mp.get_context("spawn")
    found: list[tuple[str, float, np.ndarray | str]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=context) as pool:
        for index, payload in enumerate(pool.map(_encode_ids, jobs, chunksize=4), start=1):
            found.append(payload)
            if index % 1000 == 0 or index == len(jobs):
                logger.info("token encode %s/%s", index, len(jobs))
    return found


def _encode_ids(job: tuple[str, str, float]) -> tuple[str, float, np.ndarray | str]:
    os.environ["OMP_NUM_THREADS"] = "1"
    root, relpath, t_audio_ms = job
    try:
        from audio2map.osu.parser import parse_beatmap
        from audio2map.osu.v1.timing import parse_redlines_file
        from audio2map.tokens.v1.codec import encode_chart

        path = Path(root) / relpath
        beatmap = parse_beatmap(path)
        redlines = parse_redlines_file(path).redlines
        encoded = encode_chart(list(beatmap.notes), redlines, t_audio_ms)
        vocab = build_vocab()
        ids = np.array([vocab[token] for token in encoded.tokens], dtype=np.int16)
        return relpath, t_audio_ms, ids
    except Exception as exc:
        return relpath, t_audio_ms, f"{type(exc).__name__}: {exc}"[:300]


def _write_token_cache(
    directory: Path,
    records: dict[str, TokenRecord],
    arrays: dict[str, np.ndarray],
) -> TokenCache:
    directory.mkdir(parents=True, exist_ok=True)
    pieces: list[np.ndarray] = []
    packed: dict[str, TokenRecord] = {}
    offset = 0
    for relpath in sorted(records):
        record = records[relpath]
        array = arrays.get(relpath)
        if record.error is not None or array is None:
            packed[relpath] = TokenRecord(
                relpath, record.t_audio_ms, 0, 0, error=record.error or "missing"
            )
            continue
        piece = np.asarray(array, dtype=np.int16)
        pieces.append(piece)
        packed[relpath] = TokenRecord(relpath, record.t_audio_ms, offset, int(piece.shape[0]))
        offset += int(piece.shape[0])
    ids = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)
    blob = directory / "ids.i16"
    temporary = directory / "ids.i16.part"
    np.asarray(ids, dtype="<i2").tofile(temporary)
    temporary.replace(blob)
    manifest = directory / "manifest.jsonl"
    manifest_part = directory / "manifest.jsonl.part"
    with manifest_part.open("w", encoding="utf-8") as handle:
        header = {
            "kind": _KIND,
            "version": TOKEN_CACHE_VERSION,
            "tokenizer": TOKENIZER_VERSION,
        }
        handle.write(json.dumps(header) + "\n")
        for relpath in sorted(packed):
            record = packed[relpath]
            payload: dict[str, str | int | float] = {
                "relpath": record.relpath,
                "t_audio_ms": record.t_audio_ms,
                "offset": record.offset,
                "n": record.n,
            }
            if record.error:
                payload["error"] = record.error
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    manifest_part.replace(manifest)
    return TokenCache(directory, packed, ids)
