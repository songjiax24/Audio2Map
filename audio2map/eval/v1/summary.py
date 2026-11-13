"""Aggregate live and rescored results into the v1 evaluation summary."""

from __future__ import annotations

from collections import Counter, defaultdict

from audio2map.eval.motif import motif_length
from audio2map.eval.motif_freq import MotifCorpus, corpus_distance, scope_frequency_distance
from audio2map.eval.note_match import NoteMatchStats
from audio2map.eval.v1.metrics import diversity, event_distribution, plausibility
from audio2map.eval.v1.motifs import FORMAT, motif_sequences, window_counts
from audio2map.eval.v1.statistics import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    chart_values,
    condition_responses,
    describe,
    mean_available,
    mean_interval,
)
from audio2map.eval.v1.teacher import summarize_teacher
from audio2map.features.cond.v1.vec import COND_SOURCE_FIELDS
from audio2map.generate.v1.notes import notes_from_events
from audio2map.generate.v1.sample import VALID


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["scenario"])].append(row)
    result = {"teacher": {}, "constrained": {}, "unconstrained": {}}
    for (mode, scenario), group in grouped.items():
        if mode == "teacher":
            result[mode][scenario] = summarize_teacher([row["metrics"] for row in group])
        elif mode == "unconstrained":
            counts = Counter(row["validity"] for row in group)
            result[mode][scenario] = {
                "samples": len(group),
                "counts": dict(counts),
                "valid_rate": counts[VALID] / len(group),
            }
        else:
            errors = {}
            for field in COND_SOURCE_FIELDS:
                values = [row["condition_error"][field] for row in group]
                errors[field] = {
                    "mae": mean_available(values),
                    "observations": sum(v is not None for v in values),
                    "sample_distribution": describe(values),
                    "chart_distribution": describe(list(chart_values(
                        group, lambda row: row["condition_error"][field],
                    ).values())),
                }
            result[mode][scenario] = {
                "samples": len(group),
                "charts": len({row["relpath"] for row in group}),
                "match": {
                    **match_summary(
                        [row["match"]["all"] for row in group],
                        chart_ids=[row["relpath"] for row in group],
                    ),
                    "by_kind": {
                        kind: match_summary(
                            [row["match"]["by_kind"][kind] for row in group],
                            chart_ids=[row["relpath"] for row in group],
                        )
                        for kind in ("tap", "head", "tail")
                    },
                    "by_lane": {
                        str(lane): match_summary(
                            [row["match"]["by_lane"][str(lane)] for row in group],
                            chart_ids=[row["relpath"] for row in group],
                        )
                        for lane in range(4)
                    },
                    "views": {
                        unit: {
                            scope: {
                                kind: match_summary(
                                    [row["match"][unit][scope][kind] for row in group],
                                    chart_ids=[row["relpath"] for row in group],
                                )
                                for kind in group[0]["match"][unit][scope]
                            }
                            for scope in ("lane_aware", "lane_agnostic")
                        }
                        for unit in ("event", "note")
                        if all(unit in row["match"] for row in group)
                    },
                },
                "condition_error": errors,
                "plausibility": plausibility_summary(group),
                "distribution": merge_distributions([row["distribution"] for row in group]),
            }
    return result


def merge_distributions(rows: list[dict]) -> dict:
    result = {"endpoints": sum(row["endpoints"] for row in rows)}
    for name in ("notes", "taps", "holds", "rows"):
        result[name] = sum(row[name] for row in rows) if all(name in row for row in rows) else None
    for name in ("denominator", "position", "hold_grid_span", "hold_beat_span", "row_pattern"):
        counter = Counter()
        for row in rows:
            counter.update(row.get(name) or {})
        result[name] = dict(counter)
    result["lanes"] = {
        str(lane): {
            kind: sum(row["lanes"][str(lane)][kind] for row in rows)
            for kind in ("tap", "head", "tail")
        }
        for lane in range(4)
    }
    return result


def match_summary(matches: list[dict], *, chart_ids: list[str] | None = None) -> dict:
    micro = NoteMatchStats(
        **{name: sum(match[name] for match in matches) for name in ("tp", "fp", "fn")}
    )
    by_chart = chart_values(
        [{"relpath": key, "f1": row["f1"]} for key, row in zip(chart_ids or [], matches)],
        lambda row: row["f1"],
    )
    return {
        "micro": {
            "tp": micro.tp,
            "fp": micro.fp,
            "fn": micro.fn,
            "precision": micro.precision,
            "recall": micro.recall,
            "f1": micro.f1,
        },
        "macro_f1": mean_available(match["f1"] for match in matches),
        "sample_distribution": describe([match["f1"] for match in matches]),
        "chart_distribution": describe(list(by_chart.values())),
        "chart_mean_ci": mean_interval(list(by_chart.values())),
        "reference_support": sum(match["tp"] + match["fn"] for match in matches),
        "charts_with_reference": len({
            key for key, row in zip(chart_ids or [], matches) if row["tp"] + row["fn"] > 0
        }),
    }


def plausibility_summary(rows: list[dict]) -> dict:
    counts = {
        name: sum(row["plausibility"][name] for row in rows)
        for name in ("concurrent", "almost_concurrent", "duplicate_start", "nonpositive_hold")
    }
    counts.update(
        {
            f"charts_with_{name}": len(
                {row["relpath"] for row in rows if row["plausibility"][name] > 0}
            )
            for name in ("concurrent", "almost_concurrent")
        }
    )
    counts["empty_samples"] = sum(row.get("notes") == 0 for row in rows)
    counts["sample_rates"] = {
        name: sum(row["plausibility"][name] > 0 for row in rows) / len(rows) if rows else None
        for name in ("concurrent", "almost_concurrent", "duplicate_start", "nonpositive_hold")
    }
    return counts


class EvaluationCollection:
    def __init__(self, vocabulary, scenarios) -> None:
        self.vocabulary = vocabulary
        self.human = MotifCorpus(vocabulary) if vocabulary else None
        self.generated = {name: MotifCorpus(vocabulary) for name in scenarios} if vocabulary else {}
        self.diversities = {}
        self.references = []
        self.windows = {}

    def _motifs(self, name, corpus, tokens, grid):
        corpus.add_sequences(motif_sequences(tokens))
        counts = window_counts(tokens, grid, self.vocabulary)
        totals = self.windows.setdefault(name, {})
        for scope, lengths in counts.items():
            by_length = totals.setdefault(scope, {})
            for length, row in lengths.items():
                by_length.setdefault(length, Counter()).update(row)

    def add_reference(self, relpath, tokens, events, grid, meta=None):
        stats = event_distribution(events, grid, tokens)
        notes, _warnings = notes_from_events(events)
        self.references.append({
            "relpath": relpath, "distribution": stats, "notes": len(notes),
            "audio_duration_ms": grid.t_audio_ms,
            "plausibility": plausibility(notes),
            "meta": meta.to_dict() if meta is not None else None,
        })
        if self.human is not None:
            self._motifs("human", self.human, tokens, grid)

    def add_generation(self, row, tokens, grid):
        if row["scenario"] in self.generated:
            self._motifs(row["scenario"], self.generated[row["scenario"]], tokens, grid)

    def add_diversity(self, relpath, samples, tokens):
        result = diversity(samples)
        if self.vocabulary is not None:
            corpora = {}
            for seed, sequence in tokens.items():
                corpus = MotifCorpus(self.vocabulary)
                corpus.add_sequences(motif_sequences(sequence))
                corpora[seed] = corpus
            for pair in result["pairs"]:
                left, right = [corpora[seed] for seed in pair["seeds"]]
                pair["motif"] = {
                    scope: scope_frequency_distance(
                        left.frequency(scope), right.frequency(scope), self.vocabulary.texts[scope],
                        generated_observations=left.observations[scope],
                        human_observations=right.observations[scope],
                    )
                    for scope in self.vocabulary.texts
                }
        self.diversities[relpath] = result

    def _coverage(self):
        result = {}
        for name, scopes in self.windows.items():
            corpus = self.human if name == "human" else self.generated[name]
            result[name] = {}
            for scope, lengths in scopes.items():
                hits = Counter()
                for key, count in corpus.counts[scope].items():
                    hits[str(motif_length(key))] += count
                result[name][scope] = {
                    length: {
                        **dict(row), "matched_windows": hits[length],
                        "coverage": hits[length] / row["windows"] if row["windows"] else None,
                    }
                    for length, row in lengths.items()
                }
        return result

    def finish(self, rows, *, run, pool, elapsed_s, reused_records, peak_allocated_gib):
        return {
            "run": run, "pool": pool, **summarize(rows),
            "diversity": self.diversities,
            "references": self.references,
            "reference_distribution": merge_distributions(
                [row["distribution"] for row in self.references]
            ),
            "reference_plausibility": plausibility_summary(self.references),
            "condition_response": condition_responses(rows, seed=run["seeds"][0]),
            "motif": {
                name: corpus_distance(corpus, self.human)
                for name, corpus in self.generated.items() if corpus.charts
            } if self.human is not None else {},
            "motif_coordinates": FORMAT,
            "analysis": {
                "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
                "confidence_level": CONFIDENCE_LEVEL, "resampling_unit": "chart",
                "f1_role": "Positive evidence of reference reproduction; no total quality score.",
            },
            "motif_windows": self._coverage(),
            "elapsed_s": elapsed_s, "reused_records": reused_records,
            "peak_allocated_gib": peak_allocated_gib,
        }
