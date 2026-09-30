"""Seed-to-seed diversity on one input.

Reference matching compares a generation with the human chart. This module
compares generations with each other and does not replace that comparison.
"""

from __future__ import annotations

from collections import Counter

from audio2map.eval.motif_freq import OrderedMotifVocab, scope_frequency_distance
from audio2map.eval.note_match import reference_match
from audio2map.grid import CanonicalTiming, TickNote
from audio2map.osu.schema import ManiaNote


def note_multiset(notes: list[ManiaNote], timing: CanonicalTiming) -> Counter:
    return Counter(TickNote.from_note(note, timing) for note in notes)


def exact_duplicate_report(
    charts: list[list[ManiaNote]],
    timing: CanonicalTiming,
) -> dict:
    """Pairs whose tick-note multisets are identical, including hold end ticks."""
    sets = [note_multiset(notes, timing) for notes in charts]
    pairs = []
    exact = 0
    for i in range(len(charts)):
        for j in range(i + 1, len(charts)):
            same = sets[i] == sets[j]
            exact += int(same)
            pairs.append({"i": i, "j": j, "exact": same})
    total = len(pairs)
    return {
        "pair_count": total,
        "exact_pair_count": exact,
        "exact_pair_rate": exact / total if total else 0.0,
        "pairs": pairs,
    }


def seed_pair_report(
    charts: list[list[ManiaNote]],
    timing: CanonicalTiming,
    *,
    frequencies: list[dict[str, list[float]]] | None = None,
    observations: list[dict[str, int]] | None = None,
    vocab: OrderedMotifVocab | None = None,
) -> dict:
    """Full matching for every seed pair, plus optional motif frequency distance."""
    matching = []
    motif = []
    for i in range(len(charts)):
        for j in range(i + 1, len(charts)):
            matched = reference_match(charts[i], charts[j], timing)
            matching.append({"i": i, "j": j, "match": matched})
            if frequencies is not None and vocab is not None:
                if observations is None:
                    raise ValueError("motif frequency distance requires scope observation counts")
                motif.append(
                    {
                        "i": i,
                        "j": j,
                        "distance": {
                            scope: scope_frequency_distance(
                                frequencies[i][scope],
                                frequencies[j][scope],
                                vocab.texts[scope],
                                generated_observations=observations[i][scope],
                                human_observations=observations[j][scope],
                            )
                            for scope in vocab.texts
                        },
                    }
                )
    report = {
        "exact": exact_duplicate_report(charts, timing),
        "matching": matching,
    }
    if motif:
        report["motif"] = motif
    return report
