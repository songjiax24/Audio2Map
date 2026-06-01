"""Offline tick-grid audio cache (v2)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from audio2map.audio.loader import audio_duration_ms, find_audio_file, load_mono_audio
from audio2map.audio.tick_features import V2_FEATURE_DIM, V2_SAMPLE_RATE, compute_tick_grid_features
from audio2map.osu.grid_config import TICKS_PER_BAR, TICKS_PER_BEAT
from audio2map.osu.row_tokens import CanonicalTiming
from audio2map.osu.tick_range import audio_grid_cache_stem, audio_tick_range_ms


@dataclass(frozen=True, slots=True)
class AudioGridMeta:
    tick_min: int
    tick_max: int
    offset_ms: float
    canonical_bpm: float
    ticks_per_beat: int
    feature_dim: int
    sample_rate: int
    audio_hash: str
    audio_path: str
    duration_ms: int

    @property
    def num_ticks(self) -> int:
        return self.tick_max - self.tick_min

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AudioGridMeta:
        return cls(**data)


def audio_content_hash(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()[:16]


def grid_cache_paths(out_dir: Path, stem: str) -> tuple[Path, Path]:
    return out_dir / f"{stem}.npy", out_dir / f"{stem}.json"


def expected_grid_shape(meta: AudioGridMeta | dict) -> tuple[int, int]:
    if isinstance(meta, AudioGridMeta):
        tick_min, tick_max, feature_dim = meta.tick_min, meta.tick_max, meta.feature_dim
    else:
        tick_min, tick_max, feature_dim = meta["tick_min"], meta["tick_max"], meta["feature_dim"]
    return tick_max - tick_min, feature_dim


def is_valid_audio_grid_cache(npy_path: Path, json_path: Path) -> bool:
    """Return True only when both artifacts exist and ``npy`` matches ``json`` meta."""
    if not npy_path.is_file() or not json_path.is_file():
        return False
    try:
        meta = AudioGridMeta.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
        features = np.load(npy_path)
    except Exception:
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
    sample_rate: int = V2_SAMPLE_RATE,
) -> tuple[np.ndarray, AudioGridMeta]:
    """Compute tick-grid features for one audio + timing configuration."""
    y, sr = load_mono_audio(audio_path, sample_rate=sample_rate)
    duration_ms = audio_duration_ms(y, sr)
    tick_min, tick_max = audio_tick_range_ms(
        duration_ms,
        offset_ms=float(timing.offset_ms),
        tick_ms=timing.tick_ms,
    )
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
        offset_ms=float(timing.offset_ms),
        canonical_bpm=float(timing.canonical_bpm),
        ticks_per_beat=TICKS_PER_BEAT,
        feature_dim=V2_FEATURE_DIM,
        sample_rate=sample_rate,
        audio_hash=audio_content_hash(audio_path),
        audio_path=str(audio_path),
        duration_ms=duration_ms,
    )
    return features, meta


def save_audio_grid(
    features: np.ndarray,
    meta: AudioGridMeta,
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_grid_cache_stem(
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
    return audio_grid_cache_stem(
        audio_content_hash(audio_path),
        canonical_bpm=timing.canonical_bpm,
        offset_ms=float(timing.offset_ms),
    )


def precompute_chart_audio_grid(
    set_dir: Path,
    timing: CanonicalTiming,
    out_dir: Path,
    *,
    skip_existing: bool = True,
) -> tuple[Path, Path] | None:
    """Build cache for ``set_dir`` audio at chart timing; skip if present."""
    audio_path = find_audio_file(set_dir)
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
    padding_value: float = 0.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Slice ``[window_start_bar, window_end_bar)`` with zero padding outside grid."""
    window_start_tick = window_start_bar * TICKS_PER_BAR
    window_end_tick = window_end_bar * TICKS_PER_BAR
    length = window_end_tick - window_start_tick

    start_idx = window_start_tick - meta.tick_min
    end_idx = window_end_tick - meta.tick_min

    out = np.full((length, meta.feature_dim), padding_value, dtype=grid.dtype)

    grid_start = max(0, start_idx)
    grid_end = min(grid.shape[0], end_idx)
    out_start = max(0, -start_idx)
    out_end = out_start + (grid_end - grid_start)

    if grid_end > grid_start:
        out[out_start:out_end] = grid[grid_start:grid_end]

    return out, {
        "window_start_tick": window_start_tick,
        "window_end_tick": window_end_tick,
        "padded_ticks_start": out_start,
        "padded_ticks_end": length - (out_end - out_start) - out_start,
    }
