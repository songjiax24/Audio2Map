"""Global Top-K motif vocabulary fitting.

There is no length cap. A motif stays eligible while it is inside the current
global Top-K. The order is occurrence descending, then ``format_motif``
ascending. That order is part of the pruning rule: an extension formats to a
string strictly after its prefix, and its occurrence cannot exceed the
prefix, so it always ranks behind the prefix.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from audio2map.eval.motif import (
    MotifKey,
    MotifSequences,
    format_motif,
    mirror_hand_symbol,
    motif_key,
    motif_length,
)
from audio2map.dataset.split import (
    default_split_manifest_path,
    list_raw_osu_paths,
    load_split_manifest,
    raw_dir,
    split_paths,
)
from audio2map.grid import CanonicalTiming
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.schema import ManiaNote
from audio2map.osu.timing import summarize_beatmap_timing

TIE_BREAK = "occurrence_desc_then_format_motif_asc"

SequencePositions = tuple[list, list[int]]


@dataclass(frozen=True, slots=True)
class RankedMotif:
    key: MotifKey
    occurrence: int
    length: int

    def text(self) -> str:
        return format_motif(self.key)


def extend_motif_key(parent: MotifKey, gap: int, symbol: object) -> MotifKey:
    """One-step extension. Must equal ``motif_key(..., length + 1)``."""
    return parent + (gap, symbol)


def _rank(counts: dict[MotifKey, int]) -> list[tuple[MotifKey, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], format_motif(item[0])))


def _as_ranked(head: list[tuple[MotifKey, int]]) -> list[RankedMotif]:
    return [
        RankedMotif(key=key, occurrence=count, length=motif_length(key))
        for key, count in head
    ]


def fit_scope(
    sequences: Sequence[SequencePositions],
    *,
    k_fit: int,
    trace: list[dict] | None = None,
) -> list[RankedMotif]:
    """Rank one pooled scope.

    ``sequences`` are already pooled: four lanes in one list, or hand 12
    together with mirrored hand 34. The Top-K cutoff is recomputed only after
    a layer has scanned every sequence.
    """
    if k_fit < 1:
        raise ValueError(f"k_fit must be positive, got {k_fit}")
    counts = _count_length_one(sequences)
    if not counts:
        return []
    length = 1
    while True:
        head = _rank(counts)[:k_fit]
        frontier = {key for key, _count in head if motif_length(key) == length}
        if not frontier:
            return _as_ranked(head)
        extensions = _count_extensions(sequences, frontier, length)
        if trace is not None:
            trace.append(
                {
                    "length": length,
                    "frontier": len(frontier),
                    "extensions_observed": len(extensions),
                }
            )
        if not extensions:
            return _as_ranked(head)
        merged = {key: count for key, count in head}
        merged.update(extensions)
        new_head = _rank(merged)[:k_fit]
        entered = sum(motif_length(key) == length + 1 for key, _count in new_head)
        if trace is not None:
            trace[-1]["entered_top_k"] = entered
        if entered == 0:
            return _as_ranked(new_head)
        counts = {key: count for key, count in new_head}
        length += 1


def _count_length_one(sequences: Iterable[SequencePositions]) -> dict[MotifKey, int]:
    counts: dict[MotifKey, int] = {}
    for symbols, _ticks in sequences:
        for symbol in symbols:
            key = (symbol,)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _count_extensions(
    sequences: Sequence[SequencePositions],
    frontier: set[MotifKey],
    length: int,
) -> dict[MotifKey, int]:
    """Count only extensions that a frontier motif actually meets in a scan."""
    extensions: dict[MotifKey, int] = {}
    for symbols, ticks in sequences:
        limit = len(symbols) - length
        if limit <= 0:
            continue
        for start in range(limit):
            parent = _parent_key(symbols, ticks, start, length)
            if parent not in frontier:
                continue
            extended = extend_motif_key(
                parent,
                ticks[start + length] - ticks[start + length - 1],
                symbols[start + length],
            )
            extensions[extended] = extensions.get(extended, 0) + 1
    return extensions


def _parent_key(symbols: list, ticks: list[int], start: int, length: int) -> MotifKey:
    if length == 1:
        return (symbols[start],)
    return motif_key(symbols, ticks, start, length)


def scope_sequences(seqs: MotifSequences) -> dict[str, list[SequencePositions]]:
    """Lane, mirrored hand, and row sequences used by fitting and the evaluator."""
    lane: list[SequencePositions] = []
    for positions in seqs.lane:
        extracted = _extract(positions, mirror=False)
        if extracted is not None:
            lane.append(extracted)
    hand: list[SequencePositions] = []
    for name in ("12", "34"):
        extracted = _extract(seqs.hand[name], mirror=(name == "34"))
        if extracted is not None:
            hand.append(extracted)
    row_extracted = _extract(seqs.row, mirror=False)
    row = [row_extracted] if row_extracted is not None else []
    return {"lane": lane, "hand": hand, "row": row}


def _extract(positions: Sequence[tuple[int, object]], *, mirror: bool) -> SequencePositions | None:
    if not positions:
        return None
    ticks = [tick for tick, _symbol in positions]
    if mirror:
        symbols = [mirror_hand_symbol(symbol) for _tick, symbol in positions]
    else:
        symbols = [symbol for _tick, symbol in positions]
    return symbols, ticks


def add_chart_sequences(
    scopes: dict[str, list[SequencePositions]],
    notes: list[ManiaNote],
    timing: CanonicalTiming,
) -> None:
    extracted = scope_sequences(MotifSequences.from_notes(notes, timing))
    for name in ("lane", "hand", "row"):
        scopes[name].extend(extracted[name])


@dataclass(frozen=True, slots=True)
class MotifVocabFit:
    k_fit: int
    lane: list[RankedMotif]
    hand: list[RankedMotif]
    row: list[RankedMotif]


def fit_scopes(
    scopes: dict[str, list[SequencePositions]],
    *,
    k_fit: int,
    traces: dict[str, list[dict]] | None = None,
) -> MotifVocabFit:
    return MotifVocabFit(
        k_fit=k_fit,
        lane=fit_scope(scopes["lane"], k_fit=k_fit, trace=None if traces is None else traces.setdefault("lane", [])),
        hand=fit_scope(scopes["hand"], k_fit=k_fit, trace=None if traces is None else traces.setdefault("hand", [])),
        row=fit_scope(scopes["row"], k_fit=k_fit, trace=None if traces is None else traces.setdefault("row", [])),
    )


def evaluator_vocab(fit: MotifVocabFit) -> dict[str, list[str]]:
    """JSON object ``_motif_vocab_from_json`` already accepts. Rank order."""
    return {
        "lane": [item.text() for item in fit.lane],
        "hand": [item.text() for item in fit.hand],
        "row": [item.text() for item in fit.row],
    }


def sidecar_document(fit: MotifVocabFit, corpus: dict) -> dict:
    def rows(items: list[RankedMotif]) -> list[dict]:
        return [
            {"motif": item.text(), "occurrence": item.occurrence, "length": item.length}
            for item in items
        ]

    return {
        "k_fit": fit.k_fit,
        "tie_break": TIE_BREAK,
        "l_max": None,
        "containment": "retained",
        "pruning": "extend motifs in the current global top-k only; cutoff updates after each full layer",
        "corpus": corpus,
        "lane": rows(fit.lane),
        "hand": rows(fit.hand),
        "row": rows(fit.row),
    }


@dataclass
class FittingCorpus:
    """Train charts excluding variable BPM whose motif sequences can be built."""

    charts: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    scopes: dict[str, list[SequencePositions]] = field(
        default_factory=lambda: {"lane": [], "hand": [], "row": []}
    )

    def provenance(self) -> dict:
        return {
            "split": "train",
            "excluded": "variable_bpm, plus charts whose motif sequences fail",
            "charts": self.charts,
            "skipped": self.skipped,
        }


def load_fitting_corpus(split_path: Path | None = None) -> FittingCorpus:
    root = raw_dir()
    manifest = load_split_manifest(split_path or default_split_manifest_path())
    paths = split_paths(list_raw_osu_paths(root), manifest, raw_root=root)["train"]
    corpus = FittingCorpus()
    skipped: Counter[str] = Counter()
    for path in paths:
        try:
            beatmap = parse_beatmap(path)
        except Exception:
            skipped["parse"] += 1
            continue
        if not summarize_beatmap_timing(beatmap).constant_bpm:
            skipped["variable_bpm"] += 1
            continue
        try:
            timing = CanonicalTiming.from_beatmap(beatmap)
            add_chart_sequences(corpus.scopes, beatmap.notes, timing)
        except Exception:
            skipped["motif"] += 1
            continue
        corpus.charts += 1
    corpus.skipped = dict(skipped)
    return corpus


def write_vocab_files(
    fit: MotifVocabFit,
    corpus: dict,
    *,
    vocab_path: Path,
    sidecar_path: Path,
) -> None:
    vocab_path = Path(vocab_path)
    sidecar_path = Path(sidecar_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(
        json.dumps(evaluator_vocab(fit), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sidecar_path.write_text(
        json.dumps(sidecar_document(fit, corpus), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
