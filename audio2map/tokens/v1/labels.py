"""V80 window labeling on a BeatGrid.

Window ``(orig - EPS, orig + 1 + EPS)``, across sections. Minimum reduced
denominator, then closer to ``orig``, then smaller numerator. An empty window
falls back to the nearest legal point; distances within ``TIE`` prefer the
smaller denominator, then the smaller numerator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from audio2map.grid.v1.beatgrid import Beat, BeatGrid
from audio2map.utils.errors import NO_LEGAL_GRID, ChartRejected
from audio2map.utils.tolerance import EPS, TIE

# Larger than every V80 denominator. A candidate still holding this has no hit.
NO_CANDIDATE = 10_000
_FALLBACK_RADIUS = 2


@dataclass(slots=True)
class _SectionIndex:
    """Beats of one red line, packed for the labeler. Not part of BeatGrid."""

    section: int
    offset_ms: float
    beat_length_ms: float
    k_base: int
    k_mask: tuple[bool, ...]
    beat_ids: tuple[int, ...]
    residual_k: int | None
    t_red_ms: float | None


def _sections_from_beats(beats: tuple[Beat, ...]) -> tuple[_SectionIndex, ...]:
    """Rebuild the labeler's section table from emitted beats.

    A red line that emitted nothing is omitted. The labeler skips an empty
    mask, so those sections do not change a label.
    """
    grouped: dict[int, list[tuple[int, Beat]]] = {}
    for beat_id, beat in enumerate(beats):
        grouped.setdefault(beat.section, []).append((beat_id, beat))
    indexes: list[_SectionIndex] = []
    for section in sorted(grouped):
        pairs = grouped[section]
        _beat_id, sample = pairs[0]
        offset = sample.section_offset_ms
        k_base = min(beat.beat_k for _, beat in pairs)
        k_max = max(beat.beat_k for _, beat in pairs)
        width = k_max - k_base + 1
        mask = [False] * width
        beat_ids = [-1] * width
        residual_k: int | None = None
        t_red: float | None = None
        for beat_id, beat in pairs:
            slot = beat.beat_k - k_base
            mask[slot] = True
            beat_ids[slot] = beat_id
            if beat.is_residual:
                residual_k = beat.beat_k
                t_red = beat.t_red_ms
        indexes.append(
            _SectionIndex(
                section=section,
                offset_ms=offset,
                beat_length_ms=sample.nominal_length_ms,
                k_base=k_base,
                k_mask=tuple(mask),
                beat_ids=tuple(beat_ids),
                residual_k=residual_k,
                t_red_ms=t_red,
            )
        )
    return tuple(indexes)


@dataclass(frozen=True, slots=True)
class LabelBatch:
    numer: np.ndarray
    denom: np.ndarray
    time_ms: np.ndarray
    section: np.ndarray
    beat_k: np.ndarray
    beat_id: np.ndarray
    global_b: np.ndarray
    from_window: np.ndarray


@dataclass
class Candidates:
    """Best cell so far, one row per endpoint."""

    denom: np.ndarray
    dist: np.ndarray
    numer: np.ndarray
    time_ms: np.ndarray
    section: np.ndarray
    beat_k: np.ndarray
    beat_id: np.ndarray

    @classmethod
    def blank(cls, count: int) -> Candidates:
        return cls(
            denom=np.full(count, NO_CANDIDATE, dtype=np.int16),
            dist=np.full(count, np.inf, dtype=np.float64),
            numer=np.full(count, NO_CANDIDATE, dtype=np.int16),
            time_ms=np.zeros(count, dtype=np.float64),
            section=np.full(count, -1, dtype=np.int32),
            beat_k=np.zeros(count, dtype=np.int32),
            beat_id=np.full(count, -1, dtype=np.int32),
        )

    def subset(self, index: np.ndarray) -> Candidates:
        return Candidates(
            denom=self.denom[index].copy(),
            dist=self.dist[index].copy(),
            numer=self.numer[index].copy(),
            time_ms=self.time_ms[index].copy(),
            section=self.section[index].copy(),
            beat_k=self.beat_k[index].copy(),
            beat_id=self.beat_id[index].copy(),
        )

    def write_back(self, index: np.ndarray, chosen: Candidates) -> None:
        self.denom[index] = chosen.denom
        self.dist[index] = chosen.dist
        self.numer[index] = chosen.numer
        self.time_ms[index] = chosen.time_ms
        self.section[index] = chosen.section
        self.beat_k[index] = chosen.beat_k
        self.beat_id[index] = chosen.beat_id

    def _keep(
        self,
        better: np.ndarray,
        *,
        numer: int,
        denom: int,
        dist: np.ndarray,
        time_ms: np.ndarray,
        section: int,
        beat_k: np.ndarray,
        beat_id: np.ndarray,
    ) -> None:
        if not np.any(better):
            return
        self.denom[better] = denom
        self.dist[better] = dist[better]
        self.numer[better] = numer
        self.time_ms[better] = time_ms[better]
        self.section[better] = section
        self.beat_k[better] = beat_k[better]
        self.beat_id[better] = beat_id[better]

    def consider_window(
        self,
        *,
        ok: np.ndarray,
        dist: np.ndarray,
        numer: int,
        denom: int,
        time_ms: np.ndarray,
        section: int,
        beat_k: np.ndarray,
        beat_id: np.ndarray,
    ) -> None:
        better = ok & (
            (denom < self.denom)
            | ((denom == self.denom) & (dist < self.dist))
            | ((denom == self.denom) & (dist == self.dist) & (numer < self.numer))
        )
        self._keep(
            better,
            numer=numer,
            denom=denom,
            dist=dist,
            time_ms=time_ms,
            section=section,
            beat_k=beat_k,
            beat_id=beat_id,
        )

    def consider_fallback(
        self,
        *,
        ok: np.ndarray,
        dist: np.ndarray,
        numer: int,
        denom: int,
        time_ms: np.ndarray,
        section: int,
        beat_k: np.ndarray,
        beat_id: np.ndarray,
    ) -> None:
        closer = dist + TIE < self.dist
        tied = dist <= self.dist + TIE
        prefer = (denom < self.denom) | ((denom == self.denom) & (numer < self.numer))
        better = ok & (closer | (tied & prefer))
        self._keep(
            better,
            numer=numer,
            denom=denom,
            dist=dist,
            time_ms=time_ms,
            section=section,
            beat_k=beat_k,
            beat_id=beat_id,
        )


def _beat_ok(
    section: _SectionIndex,
    beat_k: np.ndarray,
    time_ms: np.ndarray,
    t_audio_ms: float,
) -> np.ndarray:
    if not section.k_mask:
        return np.zeros(beat_k.shape, dtype=bool)
    slot = beat_k - section.k_base
    width = len(section.k_mask)
    in_range = (slot >= 0) & (slot < width)
    safe = np.where(in_range, slot, 0)
    mask = np.asarray(section.k_mask, dtype=bool)
    ok = in_range & mask[safe]
    ok &= (time_ms >= 0.0) & (time_ms < t_audio_ms)
    if section.residual_k is not None and section.t_red_ms is not None:
        stripped = (beat_k == section.residual_k) & (time_ms >= section.t_red_ms - EPS)
        ok &= ~stripped
    return ok


def _beat_ids(section: _SectionIndex, beat_k: np.ndarray) -> np.ndarray:
    slot = beat_k - section.k_base
    width = len(section.beat_ids)
    in_range = (slot >= 0) & (slot < width)
    safe = np.where(in_range, slot, 0)
    ids = np.asarray(section.beat_ids, dtype=np.int32)
    return np.where(in_range, ids[safe], -1)


def _search_section(
    section: _SectionIndex,
    *,
    orig: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    cells: tuple[tuple[int, int], ...],
    t_audio_ms: float,
    window: Candidates,
    fallback: Candidates,
) -> None:
    length = section.beat_length_ms
    if length <= 0.0 or not section.k_mask:
        return
    span = (1.0 + 2.0 * EPS) / length
    n_extra = int(math.ceil(span)) + 2
    t_lo = section.offset_ms + section.k_base * length
    t_hi = section.offset_ms + (section.k_base + len(section.k_mask)) * length
    pad = max(length * (n_extra + 2), length * 3.0) + 1.0
    index = np.flatnonzero((orig >= t_lo - pad) & (orig <= t_hi + pad))
    if index.size == 0:
        return
    orig_i = orig[index]
    lo_i = lo[index]
    hi_i = hi[index]
    window_i = window.subset(index)
    fallback_i = fallback.subset(index)
    for numer, denom in cells:
        fraction = numer / denom
        left = (lo_i - section.offset_ms) / length - fraction
        first = np.floor(left).astype(np.int64) + 1
        for extra in range(n_extra):
            beat_k = first + extra
            time_ms = section.offset_ms + (beat_k.astype(np.float64) + fraction) * length
            ok = (time_ms > lo_i) & (time_ms < hi_i)
            ok &= _beat_ok(section, beat_k, time_ms, t_audio_ms)
            window_i.consider_window(
                ok=ok,
                dist=np.abs(time_ms - orig_i),
                numer=numer,
                denom=denom,
                time_ms=time_ms,
                section=section.section,
                beat_k=beat_k,
                beat_id=_beat_ids(section, beat_k),
            )
        center = (orig_i - section.offset_ms) / length - fraction
        nearest = np.floor(center + 0.5).astype(np.int64)
        for extra in range(-_FALLBACK_RADIUS, _FALLBACK_RADIUS + 1):
            beat_k = nearest + extra
            time_ms = section.offset_ms + (beat_k.astype(np.float64) + fraction) * length
            ok = _beat_ok(section, beat_k, time_ms, t_audio_ms)
            fallback_i.consider_fallback(
                ok=ok,
                dist=np.abs(time_ms - orig_i),
                numer=numer,
                denom=denom,
                time_ms=time_ms,
                section=section.section,
                beat_k=beat_k,
                beat_id=_beat_ids(section, beat_k),
            )
    window.write_back(index, window_i)
    fallback.write_back(index, fallback_i)


def label_times(grid: BeatGrid, orig_ms: np.ndarray) -> LabelBatch:
    """Label each original millisecond. Reject the chart if any point has no cell."""
    orig = np.asarray(orig_ms, dtype=np.float64)
    count = int(orig.shape[0])
    window = Candidates.blank(count)
    fallback = Candidates.blank(count)
    lo = orig - EPS
    hi = orig + 1.0 + EPS
    for section in _sections_from_beats(grid.beats):
        _search_section(
            section,
            orig=orig,
            lo=lo,
            hi=hi,
            cells=grid.cells,
            t_audio_ms=grid.t_audio_ms,
            window=window,
            fallback=fallback,
        )
    from_window = window.denom < NO_CANDIDATE
    missing = ~from_window & (fallback.denom >= NO_CANDIDATE)
    if np.any(missing):
        index = int(np.flatnonzero(missing)[0])
        raise ChartRejected(NO_LEGAL_GRID, f"endpoint index {index} orig={orig[index]}")
    use_fallback = ~from_window
    numer = window.numer.copy()
    denom = window.denom.copy()
    time_ms = window.time_ms.copy()
    section = window.section.copy()
    beat_k = window.beat_k.copy()
    beat_id = window.beat_id.copy()
    numer[use_fallback] = fallback.numer[use_fallback]
    denom[use_fallback] = fallback.denom[use_fallback]
    time_ms[use_fallback] = fallback.time_ms[use_fallback]
    section[use_fallback] = fallback.section[use_fallback]
    beat_k[use_fallback] = fallback.beat_k[use_fallback]
    beat_id[use_fallback] = fallback.beat_id[use_fallback]
    global_b = np.zeros(count, dtype=np.int32)
    if count:
        valid = beat_id >= 0
        ids = beat_id[valid]
        global_b[valid] = np.fromiter(
            (grid.beats[int(i)].global_b for i in ids),
            dtype=np.int32,
            count=int(ids.size),
        )
    return LabelBatch(
        numer=numer,
        denom=denom,
        time_ms=time_ms,
        section=section,
        beat_k=beat_k,
        beat_id=beat_id,
        global_b=global_b,
        from_window=from_window,
    )
