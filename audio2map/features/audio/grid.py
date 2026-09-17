"""Tick-grid audio cache: compute, store, slice.

Aligns log-mel frames onto the canonical BPM tick axis and caches
``(num_ticks, 128)`` grids for training and inference windows.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from audio2map.features.audio.loader import audio_duration_ms, load_mono_audio
from audio2map.features.audio.tick_features import (
    AUDIO_FEATURE_SPEC_VERSION,
    AUDIO_FEATURE_DIM,
    AUDIO_LOG_MEL_FLOOR_DB,
    AUDIO_SAMPLE_RATE,
    compute_tick_grid_features,
)
from audio2map.grid import (
    CanonicalTiming,
    TICKS_PER_BAR,
    audio_bar_range_from_ticks,
    audio_tick_range_from_duration,
)
from audio2map.osu.parser import chart_audio_path, parse_beatmap

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioGridMeta:
    """Sidecar for one cached tick grid.

    ``npy`` rows are ``[tick_min, tick_max)``. STFT recipe is ``feature_spec_version``
    (see ``tick_features``); n_fft / hop / sr / dim are not stored.
    """

    tick_min: int
    tick_max: int
    offset_ms: int
    canonical_bpm: float
    audio_hash: str
    feature_spec_version: int = AUDIO_FEATURE_SPEC_VERSION

    @property
    def num_ticks(self) -> int:
        return self.tick_max - self.tick_min

    @property
    def audio_bar_range(self) -> tuple[int, int]:
        return audio_bar_range_from_ticks(self.tick_min, self.tick_max)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AudioGridMeta:
        if "feature_spec_version" not in data:
            raise ValueError("audio grid meta missing feature_spec_version")
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in fields})


def audio_content_hash(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()[:16]


def _cache_stem(audio_hash: str, *, canonical_bpm: float, offset_ms: float) -> str:
    return f"{audio_hash}_bpm{canonical_bpm:.3f}_offset{offset_ms:.1f}"


def grid_cache_paths(out_dir: Path, stem: str) -> tuple[Path, Path]:
    return out_dir / f"{stem}.npy", out_dir / f"{stem}.json"


def expected_grid_shape(meta: AudioGridMeta) -> tuple[int, int]:
    return meta.num_ticks, AUDIO_FEATURE_DIM


def is_valid_audio_grid_cache(npy_path: Path, json_path: Path) -> bool:
    """Return True only when both artifacts exist and ``npy`` matches ``json`` meta."""
    if not npy_path.is_file() or not json_path.is_file():
        return False
    try:
        meta = AudioGridMeta.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
        features = np.load(npy_path)
    except Exception:
        return False
    if meta.feature_spec_version != AUDIO_FEATURE_SPEC_VERSION:
        return False
    return features.shape == expected_grid_shape(meta) and features.dtype == np.float32


def remove_audio_grid_cache(npy_path: Path, json_path: Path) -> None:
    """Delete one grid cache pair (missing paths are ignored)."""
    npy_path.unlink(missing_ok=True)
    json_path.unlink(missing_ok=True)


def cleanup_stale_grid_parts(out_dir: Path) -> int:
    """Remove leftover atomic-write temp files under ``out_dir``."""
    removed = 0
    for pattern in ("*.npy.part", "*.json.part", "*.part"):
        for path in out_dir.glob(pattern):
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def compute_audio_grid(
    audio_path: Path,
    timing: CanonicalTiming,
    *,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> tuple[np.ndarray, AudioGridMeta]:
    """Compute tick-grid features for one audio + timing configuration."""
    y, sr = load_mono_audio(audio_path, sample_rate=sample_rate)
    duration_ms = audio_duration_ms(y, sr)
    tick_min, tick_max = audio_tick_range_from_duration(duration_ms, timing)
    features = compute_tick_grid_features(
        y,
        sr,
        offset_ms=float(timing.offset_ms),
        tick_ms=timing.tick_ms,
        tick_min=tick_min,
        tick_max=tick_max,
    )
    meta = AudioGridMeta(
        tick_min=tick_min,
        tick_max=tick_max,
        offset_ms=timing.offset_ms,
        canonical_bpm=timing.canonical_bpm,
        audio_hash=audio_content_hash(audio_path),
        feature_spec_version=AUDIO_FEATURE_SPEC_VERSION,
    )
    return features, meta


def save_audio_grid(
    features: np.ndarray,
    meta: AudioGridMeta,
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _cache_stem(
        meta.audio_hash,
        canonical_bpm=meta.canonical_bpm,
        offset_ms=meta.offset_ms,
    )
    npy_path, json_path = grid_cache_paths(out_dir, stem)
    tmp_npy = npy_path.with_name(f"{npy_path.stem}.npy.part")
    tmp_json = json_path.with_name(f"{json_path.stem}.json.part")
    payload = features.astype(np.float32)
    try:
        with tmp_npy.open("wb") as fh:
            np.save(fh, payload, allow_pickle=False)
        tmp_json.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
        tmp_json.replace(json_path)
        tmp_npy.replace(npy_path)
    except Exception:
        tmp_npy.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)
        raise
    if not is_valid_audio_grid_cache(npy_path, json_path):
        remove_audio_grid_cache(npy_path, json_path)
        raise RuntimeError(f"audio grid cache failed post-save validation: {npy_path.stem}")
    return npy_path, json_path


def load_audio_grid(out_dir: Path, stem: str) -> tuple[np.ndarray, AudioGridMeta]:
    npy_path, json_path = grid_cache_paths(out_dir, stem)
    if not is_valid_audio_grid_cache(npy_path, json_path):
        raise FileNotFoundError(f"audio grid cache missing or invalid: {stem}")
    features = np.load(npy_path)
    meta = AudioGridMeta.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
    return features, meta


def resolve_grid_stem(audio_path: Path, timing: CanonicalTiming) -> str:
    return _cache_stem(
        audio_content_hash(audio_path),
        canonical_bpm=timing.canonical_bpm,
        offset_ms=float(timing.offset_ms),
    )


def precompute_chart_audio_grid(
    audio_path: Path,
    timing: CanonicalTiming,
    out_dir: Path,
    *,
    skip_existing: bool = True,
) -> tuple[Path, Path]:
    """Build cache for one audio file at chart timing; skip if present."""
    stem = resolve_grid_stem(audio_path, timing)
    npy_path, json_path = grid_cache_paths(out_dir, stem)
    if skip_existing and is_valid_audio_grid_cache(npy_path, json_path):
        return npy_path, json_path

    remove_audio_grid_cache(npy_path, json_path)
    features, meta = compute_audio_grid(audio_path, timing)
    return save_audio_grid(features, meta, out_dir)


def slice_audio_window(
    grid: np.ndarray,
    meta: AudioGridMeta,
    window_start_bar: int,
    window_end_bar: int,
    *,
    padding_value: float = AUDIO_LOG_MEL_FLOOR_DB,
) -> tuple[np.ndarray, dict[str, int]]:
    """Slice ``[window_start_bar, window_end_bar)``.

    Out-of-grid ticks keep their window positions and are filled with the log-mel
    floor (silence), not dropped from ``audio_mask``.
    """
    window_start_tick = window_start_bar * TICKS_PER_BAR
    window_end_tick = window_end_bar * TICKS_PER_BAR
    length = window_end_tick - window_start_tick
    out = np.full((length, AUDIO_FEATURE_DIM), padding_value, dtype=grid.dtype)

    overlap_start = max(window_start_tick, meta.tick_min)
    overlap_end = min(window_end_tick, meta.tick_max)
    n_copy = overlap_end - overlap_start
    if n_copy > 0:
        dst = overlap_start - window_start_tick
        src = overlap_start - meta.tick_min
        out[dst : dst + n_copy] = grid[src : src + n_copy]
        pad_start = dst
        pad_end = length - dst - n_copy
    else:
        pad_start = 0
        pad_end = length

    return out, {
        "window_start_tick": window_start_tick,
        "window_end_tick": window_end_tick,
        "padded_ticks_start": pad_start,
        "padded_ticks_end": pad_end,
    }


@dataclass(frozen=True, slots=True)
class SetTimingJobs:
    """Unique ``(audio_hash, bpm, offset)`` jobs under a raw root."""

    jobs: list[tuple[Path, CanonicalTiming, str, Path]]
    enum_failed: int


def iter_set_timing_jobs(
    raw_root: Path,
    *,
    is_eligible: Callable[[Path], bool] | None = None,
) -> SetTimingJobs:
    """One job per unique ``(audio_hash, bpm, offset)`` under ``raw_root``.

    Ineligible charts are skipped. Missing audio / parse / hash errors are
    logged and counted in ``enum_failed``.
    """
    jobs: list[tuple[Path, CanonicalTiming, str, Path]] = []
    seen_stems: set[str] = set()
    enum_failed = 0

    for set_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        for osu_path in sorted(set_dir.glob("*.osu")):
            if is_eligible is not None and not is_eligible(osu_path):
                continue
            try:
                timing = CanonicalTiming.from_beatmap(parse_beatmap(osu_path))
                audio_path = chart_audio_path(osu_path)
                stem = resolve_grid_stem(audio_path, timing)
            except Exception as exc:
                logger.warning("skip grid job %s: %s", osu_path, exc)
                enum_failed += 1
                continue
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            jobs.append((set_dir, timing, stem, audio_path))
    return SetTimingJobs(jobs=jobs, enum_failed=enum_failed)
