"""log BPM, relative position, and the frame-wise 4-vector that feeds phase.

Scalars use the same ``t`` as φ and ψ. ``log BPM`` is ``ln(60000 / beatLength)``,
which is ``−ln(minutes per beat)``. This side is the one that is stored. The
affine is ``(ln(BPM) - ln(180)) / 0.5``. Extremes are not clipped.

``phase_rows`` is the batch path shared by training examples and segmented
encode. It builds the beat-start and redline tables once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from audio2map.grid.v1.beatgrid import BeatGrid
from audio2map.osu.v1.timing import Redline
from audio2map.utils.time import DT_MS

MS_PER_MINUTE = 60_000.0
LOG_BPM_CENTER = math.log(180.0)
LOG_BPM_SCALE = 0.5


def bpm_from_segments(
    offsets_ms: Sequence[float],
    beat_ms: Sequence[float],
    t_ms: float,
) -> float:
    """BPM of the redline in force at ``t_ms``.

    Segments are in time order. A time before the first offset uses the
    first redline. A time on a later offset uses that later redline.
    """
    if len(offsets_ms) == 0 or len(offsets_ms) != len(beat_ms):
        raise ValueError("redline offsets and beat lengths must be paired")
    if not math.isfinite(t_ms):
        raise ValueError(f"t_ms is not finite: {t_ms}")
    index = _segment_index(offsets_ms, t_ms)
    length = float(beat_ms[index])
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(f"beat length must be positive, got {length}")
    return MS_PER_MINUTE / length


# Identity cache. A tuple of frozen red lines does not change under our feet.
_REDLINE_COLUMNS: dict[int, tuple[tuple[Redline, ...], tuple[float, ...], tuple[float, ...]]] = {}
_REDLINE_LIMIT = 16


def redline_columns(
    redlines: Sequence[Redline],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Offset and beat-length columns. A tuple is materialized once."""
    if isinstance(redlines, tuple):
        key = id(redlines)
        hit = _REDLINE_COLUMNS.get(key)
        if hit is not None and hit[0] is redlines:
            return hit[1], hit[2]
        columns = _materialize_redlines(redlines)
        if len(_REDLINE_COLUMNS) >= _REDLINE_LIMIT:
            _REDLINE_COLUMNS.clear()
        _REDLINE_COLUMNS[key] = (redlines, columns[0], columns[1])
        return columns
    return _materialize_redlines(redlines)


def _materialize_redlines(
    redlines: Sequence[Redline],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        tuple(float(line.offset_ms) for line in redlines),
        tuple(float(line.beat_length_ms) for line in redlines),
    )


def bpm_at(redlines: Sequence[Redline], t_ms: float) -> float:
    """BPM of ``redlines`` at ``t_ms``. The lines are already rate-stretched."""
    offsets, lengths = redline_columns(redlines)
    return bpm_from_segments(offsets, lengths, t_ms)


def segment_durations(offsets_ms: Sequence[float], t_audio_ms: float) -> list[float]:
    """Milliseconds each redline is in force on ``[0, t_audio]``.

    ``offsets_ms`` is in time order. The first redline covers from 0 even
    when its offset is later. A redline that starts at or after ``t_audio``
    gets weight 0.
    """
    if t_audio_ms < 0.0 or not math.isfinite(t_audio_ms):
        raise ValueError(f"t_audio_ms must be non-negative, got {t_audio_ms}")
    if len(offsets_ms) == 0:
        raise ValueError("redline offsets are empty")
    weights: list[float] = []
    count = len(offsets_ms)
    for index, offset in enumerate(offsets_ms):
        start = 0.0 if index == 0 else float(offset)
        end = float(offsets_ms[index + 1]) if index + 1 < count else t_audio_ms
        weights.append(max(0.0, min(end, t_audio_ms) - max(start, 0.0)))
    return weights


def log_bpm_norm(bpm: float) -> float:
    """``(ln(bpm) - ln(180)) / 0.5``. Not clipped."""
    if not math.isfinite(bpm) or bpm <= 0.0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    return (math.log(bpm) - LOG_BPM_CENTER) / LOG_BPM_SCALE


def relative_u(t_ms: float, t0_ms: float, t1_ms: float) -> float:
    """``(t - t0) / (t1 - t0)``. Not clamped. The span is coverage, not pad frames."""
    if not math.isfinite(t_ms) or not math.isfinite(t0_ms) or not math.isfinite(t1_ms):
        raise ValueError("coverage times must be finite")
    span = t1_ms - t0_ms
    if span == 0.0 or not math.isfinite(span):
        raise ValueError(f"coverage span must be non-zero, got {t0_ms} .. {t1_ms}")
    return (t_ms - t0_ms) / span


def log_bpm_and_u(
    offsets_ms: np.ndarray,
    beat_ms: np.ndarray,
    t_ms: np.ndarray,
    t0_ms: float,
    t1_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """log BPM and u at each ``t_ms``. Tables are built by the caller, once."""
    if offsets_ms.shape != beat_ms.shape or offsets_ms.size == 0:
        raise ValueError("redline offsets and beat lengths must be paired")
    if not math.isfinite(t0_ms) or not math.isfinite(t1_ms):
        raise ValueError("coverage times must be finite")
    span = t1_ms - t0_ms
    if span == 0.0 or not math.isfinite(span):
        raise ValueError(f"coverage span must be non-zero, got {t0_ms} .. {t1_ms}")
    if not np.isfinite(t_ms).all():
        bad = float(t_ms[np.flatnonzero(~np.isfinite(t_ms))[0]])
        raise ValueError(f"t_ms is not finite: {bad}")
    segment = np.searchsorted(offsets_ms, t_ms, side="right") - 1
    segment = np.clip(segment, 0, beat_ms.shape[0] - 1)
    used = beat_ms[segment]
    bad_length = ~np.isfinite(used) | (used <= 0.0)
    if np.any(bad_length):
        length = float(used[np.flatnonzero(bad_length)[0]])
        raise ValueError(f"beat length must be positive, got {length}")
    log_bpm = (np.log(MS_PER_MINUTE / used) - LOG_BPM_CENTER) / LOG_BPM_SCALE
    rel_u = (t_ms - t0_ms) / span
    return log_bpm, rel_u


def phase_rows(
    grid: BeatGrid,
    redlines: Sequence[Redline],
    frame_lo: int,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """φ, beat index, log BPM, and u at ``t = (frame_lo + i) · Δt``.

    Each array is float64 with shape ``(n_frames,)``. Beat columns come from
    the grid cache. Redline columns are materialized once. One ``searchsorted``
    covers every frame: Θ(B + R + F log B). Residual φ divides by nominal
    ``L_i``. ``log BPM`` is not clipped and ``u`` is not clamped. ``φ ≥ 1``
    becomes ``nextafter(1, 0)``, matching ``phase_at``.
    """
    if n_frames < 0:
        raise ValueError(f"n_frames must be non-negative, got {n_frames}")
    if n_frames == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy()
    if not grid.beats:
        raise ValueError("beat grid has no beats")
    if len(redlines) == 0:
        raise ValueError("redline offsets and beat lengths must be paired")
    t0 = float(grid.coverage_start_ms)
    t1 = float(grid.coverage_end_ms)
    if not math.isfinite(t0) or not math.isfinite(t1):
        raise ValueError("coverage times must be finite")
    span = t1 - t0
    if span == 0.0 or not math.isfinite(span):
        raise ValueError(f"coverage span must be non-zero, got {t0} .. {t1}")

    index = np.arange(n_frames, dtype=np.int64) + np.int64(frame_lo)
    t = index.astype(np.float64) * DT_MS
    if not np.isfinite(t).all():
        bad = float(t[np.flatnonzero(~np.isfinite(t))[0]])
        raise ValueError(f"t_ms is not finite: {bad}")

    starts = np.asarray(grid._starts, dtype=np.float64)
    nominal = np.asarray(grid._nominal_ms, dtype=np.float64)
    global_b = np.asarray(grid._global_b, dtype=np.int64)
    chosen = np.searchsorted(starts, t, side="right") - 1
    early = t <= starts[0]
    late = t >= t1
    valid = ~early & ~late
    chosen = np.clip(chosen, 0, starts.shape[0] - 1)

    phi = np.zeros(n_frames, dtype=np.float64)
    beat_b = np.zeros(n_frames, dtype=np.float64)
    if np.any(valid):
        pick = chosen[valid]
        raw = (t[valid] - starts[pick]) / nominal[pick]
        raw = np.where(raw < 0.0, 0.0, raw)
        raw = np.where(raw >= 1.0, math.nextafter(1.0, 0.0), raw)
        phi[valid] = raw
        beat_b[valid] = global_b[pick]
    # phase_at checks the first beat before coverage end, so early wins a tie.
    beat_b[late] = float(global_b[-1] + 1)
    beat_b[early] = float(global_b[0])

    offset_col, length_col = redline_columns(redlines)
    log_bpm, rel_u = log_bpm_and_u(
        np.asarray(offset_col, dtype=np.float64),
        np.asarray(length_col, dtype=np.float64),
        t,
        t0,
        t1,
    )
    return phi, beat_b, log_bpm, rel_u


def _segment_index(offsets_ms: Sequence[float], t_ms: float) -> int:
    # bisect_right without importing a name the hot loop would re-bind.
    lo = 0
    hi = len(offsets_ms)
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets_ms[mid] <= t_ms:
            lo = mid + 1
        else:
            hi = mid
    return max(lo - 1, 0)
