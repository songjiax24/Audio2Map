"""Pooled motif frequency and frequency distance.

Frequency divides by the scope's event-position count. Lengths are not a
partition, so frequencies do not sum to 1. Every loaded vocabulary entry is
kept, including motifs whose count on this corpus is 0.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from audio2map.eval.motif import (
    MotifKey,
    count_pooled_motifs,
    format_motif,
    motif_length,
    parse_motif,
)
from audio2map.grid import CanonicalTiming
from audio2map.osu.schema import ManiaNote

SCOPES = ("lane", "hand", "row")
LENGTH_BINS = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4", 4, 4),
    ("5", 5, 5),
    ("6-8", 6, 8),
    ("9-16", 9, 16),
    ("17+", 17, 10**9),
)


class MotifVocabError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OrderedMotifVocab:
    texts: dict[str, tuple[str, ...]]
    keys: dict[str, tuple[MotifKey, ...]]

    def sets(self) -> dict[str, set[MotifKey]]:
        return {scope: set(self.keys[scope]) for scope in SCOPES}


def load_ordered_motif_vocab(data: dict) -> OrderedMotifVocab:
    """Load ``{lane, hand, row}`` string lists. Order is preserved."""
    if set(data) != set(SCOPES):
        raise MotifVocabError(
            f"motif lists must contain exactly {SCOPES}, got {sorted(data)}"
        )
    texts: dict[str, tuple[str, ...]] = {}
    keys: dict[str, tuple[MotifKey, ...]] = {}
    for scope in SCOPES:
        items = data[scope]
        if not isinstance(items, list) or not items:
            raise MotifVocabError(f"{scope} motif list is empty or not a list")
        if len(items) != len(set(items)):
            raise MotifVocabError(f"{scope} motif list contains duplicate strings")
        parsed = tuple(parse_motif(text) for text in items)
        texts[scope] = tuple(items)
        keys[scope] = parsed
    return OrderedMotifVocab(texts=texts, keys=keys)


@dataclass
class MotifCorpus:
    vocab: OrderedMotifVocab
    counts: dict[str, Counter[MotifKey]] = field(
        default_factory=lambda: {scope: Counter() for scope in SCOPES}
    )
    observations: dict[str, int] = field(
        default_factory=lambda: {scope: 0 for scope in SCOPES}
    )
    charts: int = 0

    def add_chart(self, notes: list[ManiaNote], timing: CanonicalTiming) -> None:
        sparse, observed = count_pooled_motifs(notes, timing, self.vocab.sets())
        self.charts += 1
        for scope in SCOPES:
            self.counts[scope].update(sparse[scope])
            self.observations[scope] += observed[scope]

    def frequency(self, scope: str) -> list[float]:
        """``C(m) / N_scope_observations``, aligned to the frozen list.

        A scope with no event positions has an undefined frequency. The returned
        vector is then zeros of the vocabulary length, and ``frequency_defined``
        is false. ``corpus_distance`` does not score that vector.
        """
        total = self.observations[scope]
        counts = self.counts[scope]
        if total == 0:
            return [0.0 for _ in self.vocab.keys[scope]]
        return [counts[key] / total for key in self.vocab.keys[scope]]

    def avg_count(self, scope: str) -> list[float]:
        counts = self.counts[scope]
        n = self.charts
        if n == 0:
            return [0.0 for _ in self.vocab.keys[scope]]
        return [counts[key] / n for key in self.vocab.keys[scope]]

    def as_dict(self) -> dict:
        return {
            "charts": self.charts,
            "scopes": {
                scope: {
                    "observations": self.observations[scope],
                    "frequency": self.frequency(scope),
                    "avg_count": self.avg_count(scope),
                    "zero_count": sum(
                        1 for key in self.vocab.keys[scope] if self.counts[scope][key] == 0
                    ),
                    "vocabulary": len(self.vocab.keys[scope]),
                    "frequency_defined": self.observations[scope] > 0,
                }
                for scope in SCOPES
            },
        }


def _pearson(left: list[float], right: list[float]) -> float | None:
    n = len(left)
    if n < 2 or n != len(right):
        return None
    mean_left = sum(left) / n
    mean_right = sum(right) / n
    cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    scale_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    scale_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if scale_left == 0.0 or scale_right == 0.0:
        return None
    return cov / (scale_left * scale_right)


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end + 1) / 2.0
        for index in range(start, end + 1):
            ranks[order[index]] = rank
        start = end + 1
    return ranks


def spearman(left: list[float], right: list[float]) -> float | None:
    """Spearman correlation with average ranks for ties."""
    if len(left) != len(right):
        raise ValueError("frequency vectors must have the same vocabulary length")
    return _pearson(_average_ranks(left), _average_ranks(right))


def frequency_mae(generated: list[float], human: list[float]) -> float:
    if len(generated) != len(human) or not generated:
        raise ValueError("frequency vectors must be non-empty and aligned")
    return sum(abs(a - b) for a, b in zip(generated, human)) / len(generated)


def frequency_rmse(generated: list[float], human: list[float]) -> float:
    if len(generated) != len(human) or not generated:
        raise ValueError("frequency vectors must be non-empty and aligned")
    mean_sq = sum((a - b) ** 2 for a, b in zip(generated, human)) / len(generated)
    return math.sqrt(mean_sq)


def _bin_name(length: int) -> str:
    for name, start, end in LENGTH_BINS:
        if start <= length <= end:
            return name
    return "17+"


def frequency_distance(
    generated: list[float],
    human: list[float],
    texts: tuple[str, ...],
) -> dict:
    """MAE, RMSE, and Spearman over the full vocabulary, plus diagnostics.

    The motif set is ``texts``. This function does not drop entries using the
    frequency values.
    """
    if not (len(generated) == len(human) == len(texts)):
        raise ValueError("distance vectors must stay aligned with the vocabulary")
    rows = []
    for motif, f_gen, f_human in zip(texts, generated, human):
        rows.append(
            {
                "motif": motif,
                "length": motif_length(parse_motif(motif)),
                "f_gen": f_gen,
                "f_human": f_human,
                "delta": f_gen - f_human,
                "abs": abs(f_gen - f_human),
            }
        )
    largest = min(rows, key=lambda row: (-row["abs"], row["motif"]))
    excess = min(rows, key=lambda row: (-row["delta"], row["motif"]))
    buckets: dict[str, list[float]] = {name: [] for name, _, _ in LENGTH_BINS}
    for row in rows:
        buckets[_bin_name(row["length"])].append(row["abs"])
    # A bin with no vocabulary entries has an undefined mean. It is not zero error.
    return {
        "vocabulary": len(texts),
        "mae": frequency_mae(generated, human),
        "rmse": frequency_rmse(generated, human),
        "spearman": spearman(generated, human),
        "largest_abs_deviation": {
            "motif": largest["motif"],
            "length": largest["length"],
            "f_gen": largest["f_gen"],
            "f_human": largest["f_human"],
            "delta": largest["delta"],
        },
        "largest_generated_excess": {
            "motif": excess["motif"],
            "length": excess["length"],
            "f_gen": excess["f_gen"],
            "f_human": excess["f_human"],
            "delta": excess["delta"],
        },
        "length_buckets": {
            name: (sum(values) / len(values) if values else None)
            for name, values in buckets.items()
        },
    }


def _undefined_distance(scope_texts: tuple[str, ...], generated_obs: int, human_obs: int) -> dict:
    return {
        "vocabulary": len(scope_texts),
        "observations_generated": generated_obs,
        "observations_human": human_obs,
        "mae": None,
        "rmse": None,
        "spearman": None,
        "largest_abs_deviation": None,
        "largest_generated_excess": None,
        "length_buckets": None,
    }


def scope_frequency_distance(
    generated: list[float],
    human: list[float],
    texts: tuple[str, ...],
    *,
    generated_observations: int,
    human_observations: int,
) -> dict:
    """Full-list distance. A zero observation count leaves the scope undefined."""
    if generated_observations == 0 or human_observations == 0:
        return _undefined_distance(texts, generated_observations, human_observations)
    distance = frequency_distance(generated, human, texts)
    distance["observations_generated"] = generated_observations
    distance["observations_human"] = human_observations
    return distance


def corpus_distance(generated: MotifCorpus, human: MotifCorpus) -> dict:
    if generated.vocab.texts != human.vocab.texts:
        raise MotifVocabError("generated and human motif lists differ")
    return {
        scope: scope_frequency_distance(
            generated.frequency(scope),
            human.frequency(scope),
            generated.vocab.texts[scope],
            generated_observations=generated.observations[scope],
            human_observations=human.observations[scope],
        )
        for scope in SCOPES
    }
