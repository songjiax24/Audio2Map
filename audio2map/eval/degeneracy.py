"""Degeneracy / mode-collapse metrics for generated charts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from audio2map.osu.grid_config import TICKS_PER_BAR
from audio2map.osu.row_tokens import CanonicalTiming, ms_to_tick, split_tick
from audio2map.osu.schema import ManiaNote, NoteType


@dataclass(slots=True)
class DegeneracyReport:
    note_count: int
    unique_pos_count: int
    pos_histogram: dict[int, int]
    notes_per_bar_mean: float
    autocorr_lag1: float
    autocorr_lag2: float
    autocorr_lag4: float
    hold_count: int
    hold_ratio: float
    repeated_row_bigram_rate: float
    collapse_flags: list[str]

    def to_dict(self) -> dict:
        return {
            "note_count": self.note_count,
            "unique_pos_count": self.unique_pos_count,
            "pos_histogram_top10": dict(
                sorted(self.pos_histogram.items(), key=lambda x: -x[1])[:10]
            ),
            "notes_per_bar_mean": self.notes_per_bar_mean,
            "autocorr_lag1": self.autocorr_lag1,
            "autocorr_lag2": self.autocorr_lag2,
            "autocorr_lag4": self.autocorr_lag4,
            "hold_count": self.hold_count,
            "hold_ratio": self.hold_ratio,
            "repeated_row_bigram_rate": self.repeated_row_bigram_rate,
            "collapse_flags": self.collapse_flags,
        }


def _bar_autocorr(counts: list[int], lag: int) -> float:
    if lag <= 0 or len(counts) <= lag:
        return 0.0
    a = np.asarray(counts[:-lag], dtype=np.float64)
    b = np.asarray(counts[lag:], dtype=np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def analyze_chart_degeneracy(
    notes: list[ManiaNote],
    timing: CanonicalTiming,
    *,
    row_tokens: list[str] | None = None,
) -> DegeneracyReport:
    """POS histogram, bar autocorr, hold stats, collapse heuristics."""
    ticks = [ms_to_tick(n.time_ms, timing) for n in notes]
    bars = [split_tick(t)[0] for t in ticks]
    poss = [split_tick(t)[1] for t in ticks]
    pos_hist = Counter(poss)

    if bars:
        lo, hi = min(bars), max(bars)
        npc = [sum(1 for b in bars if b == bar) for bar in range(lo, hi + 1)]
    else:
        npc = []

    holds = sum(1 for n in notes if n.note_type == NoteType.HOLD)
    hold_ratio = holds / len(notes) if notes else 0.0

    bigram_repeats = 0
    bigram_total = 0
    if row_tokens:
        rows = [t for t in row_tokens if t.startswith("<ROW_") and "initial" not in t.lower()]
        for i in range(1, len(rows)):
            bigram_total += 1
            if rows[i] == rows[i - 1]:
                bigram_repeats += 1

    flags: list[str] = []
    if len(pos_hist) <= 12:
        flags.append("low_pos_diversity")
    half_beat_only = poss and all(p % (TICKS_PER_BAR // 8) == 0 for p in poss)
    if half_beat_only and len(pos_hist) <= 8:
        flags.append("half_beat_grid_only")
    if _bar_autocorr(npc, 2) > 0.65:
        flags.append("high_2bar_autocorr")
    if holds == 0 and notes:
        flags.append("zero_holds")

    return DegeneracyReport(
        note_count=len(notes),
        unique_pos_count=len(pos_hist),
        pos_histogram=dict(pos_hist),
        notes_per_bar_mean=len(notes) / len(npc) if npc else 0.0,
        autocorr_lag1=_bar_autocorr(npc, 1),
        autocorr_lag2=_bar_autocorr(npc, 2),
        autocorr_lag4=_bar_autocorr(npc, 4),
        hold_count=holds,
        hold_ratio=hold_ratio,
        repeated_row_bigram_rate=bigram_repeats / bigram_total if bigram_total else 0.0,
        collapse_flags=flags,
    )
